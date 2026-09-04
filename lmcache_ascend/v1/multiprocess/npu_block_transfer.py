# SPDX-License-Identifier: Apache-2.0
"""Fused ``multi_layer_block_kv_transfer`` for the lmcache_driven NPU path.

Replaces the Python ``torch_ops`` fallback with
``c_ops.multi_layer_kv_transfer`` (``multi_layer_kv_transfer_kernel_v3``).
Unsupported groups fall back to ``torch_ops.multi_layer_block_kv_transfer``.

The annotation on ``lmcache_objects_ptrs`` is kept as
``list[int] | list[torch.Tensor]`` so
``_detect_block_transfer_accepts_tensor`` stays True.
"""

from __future__ import annotations

from typing import Any

import torch

import lmcache.lmcache_native as lmcache_native
from lmcache.logging import init_logger
from lmcache.v1.platform.ops_types import PageBufferShapeDesc
from lmcache.v1.platform.torch_ops import (
    _tensor_from_ptr,
    _valid_block_range_indices,
    multi_layer_block_kv_transfer as _torch_ops_block_kv,
)

from lmcache_ascend.v1.kv_format import KVCacheFormat
from lmcache_ascend.v1.multiprocess.npu_gather import (
    _build_slot_mapping,
    _descriptor_signature,
)

logger = init_logger(__name__)

EngineKVFormat = lmcache_native.EngineKVFormat
TransferDirection = lmcache_native.TransferDirection

_SUPPORTED_SINGLE_PLANE = frozenset(
    {
        EngineKVFormat.NL_X_NB_BS_NH_CS,
        EngineKVFormat.NL_X_NB_BS_HS,
    }
)
_SUPPORTED_TUPLE = frozenset({EngineKVFormat.NL_X_TWO_X_NB_BS_HS})

_ptr_table_cache: dict[tuple[int, ...], torch.Tensor] = {}

#: Incremented once per fused kernel launch. Read by server_transfer_trace.
fused_kernel_launches: int = 0
last_fallback_reason: str = ""


def _device_is_npu(device: torch.device | str) -> bool:
    name = device.type if isinstance(device, torch.device) else str(device)
    return name.startswith("npu")


def _is_310p() -> bool:
    """Avoid importing ``npu_connectors`` (circular during ``_patch_ops``)."""
    try:
        from lmcache_ascend import _build_info

        return str(_build_info.__soc_version__).lower().startswith("ascend310p")
    except Exception:
        return False


def _as_layers(paged: Any) -> list[Any]:
    if isinstance(paged, torch.Tensor):
        return [paged]
    return list(paged)


def _planes_of(layer: Any) -> list[torch.Tensor]:
    if isinstance(layer, torch.Tensor):
        return [layer]
    return [t for t in layer if isinstance(t, torch.Tensor)]


def _dim0_byte_stride(tensor: torch.Tensor) -> int:
    return int(tensor.stride(0)) * int(tensor.element_size())


def _planes_share_block_byte_stride(layers: list[Any]) -> bool:
    planes = _planes_of(layers[0])
    if len(planes) <= 1:
        return True
    expected = _dim0_byte_stride(planes[0])
    for layer in layers:
        for plane in _planes_of(layer):
            if _dim0_byte_stride(plane) != expected:
                return False
    return True


def _plane_is_dense_width(plane: torch.Tensor) -> bool:
    """Token row inside a block must be packed; dim-0 may carry block padding."""
    if plane.numel() == 0:
        return True
    if plane.dim() < 2:
        return False
    packed = 1
    for dim in range(plane.dim() - 1, 0, -1):
        if int(plane.stride(dim)) != packed:
            return False
        packed *= int(plane.shape[dim])
    return True


def _staging_aligned(ptr: int) -> bool:
    return ptr != 0 and (ptr % 32) == 0


def _fallback_reason(
    paged_layers: list[Any],
    lmcache_objects_ptrs: list[int] | list[torch.Tensor],
    device: torch.device | str,
    engine_kv_format: EngineKVFormat,
) -> str:
    """Host-only eligibility. Alignment/UB layout is the kernel's problem."""
    if not _device_is_npu(device):
        return "non-npu"
    if _is_310p():
        return "310p"
    if engine_kv_format not in _SUPPORTED_SINGLE_PLANE | _SUPPORTED_TUPLE:
        return (
            f"unsupported_format:{getattr(engine_kv_format, 'name', engine_kv_format)}"
        )
    if not paged_layers:
        return "empty_paged"
    planes0 = _planes_of(paged_layers[0])
    if engine_kv_format in _SUPPORTED_SINGLE_PLANE and len(planes0) != 1:
        return f"num_planes:{len(planes0)}"
    if engine_kv_format in _SUPPORTED_TUPLE and len(planes0) != 2:
        return f"tuple_num_planes:{len(planes0)}"
    if not _planes_share_block_byte_stride(paged_layers):
        return "disagreeing_plane_strides"
    for layer in paged_layers:
        for plane in _planes_of(layer):
            if not _plane_is_dense_width(plane):
                return "non_dense_width"
    for obj in lmcache_objects_ptrs:
        ptr = int(obj.data_ptr()) if isinstance(obj, torch.Tensor) else int(obj)
        if not _staging_aligned(ptr):
            return "staging_unaligned"
    return ""


def _ptr_table_for(paged_layers: list[Any], device: torch.device) -> torch.Tensor:
    kv_dict = {str(i): layer for i, layer in enumerate(paged_layers)}
    sig = _descriptor_signature(kv_dict)
    cached = _ptr_table_cache.get(sig)
    if cached is not None:
        return cached
    ptrs: list[int] = []
    for layer in paged_layers:
        for plane in _planes_of(layer):
            ptrs.append(int(plane.data_ptr()))
    table = torch.tensor(ptrs, dtype=torch.int64, device=device)
    _ptr_table_cache[sig] = table
    return table


def _block_ids_list(block_ids: torch.Tensor | list[int]) -> list[int]:
    if isinstance(block_ids, torch.Tensor):
        return [int(x) for x in block_ids.detach().cpu().reshape(-1).tolist()]
    return [int(x) for x in block_ids]


def _uint8_staging(
    obj: int | torch.Tensor,
    nl: int,
    n_tokens: int,
    hidden_bytes: int,
    device: torch.device | str,
) -> torch.Tensor | None:
    """View one full chunk as contiguous ``[1, nl, n_tokens, hidden_bytes]`` uint8.

    Prefix/tail skips are expressed as ``-1`` slots, not a token-window slice:
    ``[L, T, H]`` is not a contiguous ``[L, n, H]`` subregion when ``n < T``.
    """
    expected = nl * n_tokens * hidden_bytes
    if isinstance(obj, torch.Tensor):
        if not obj.is_contiguous():
            return None
        nbytes = int(obj.numel()) * int(obj.element_size())
        if nbytes < expected:
            return None
        try:
            return obj.view(torch.uint8).reshape(-1)[:expected].reshape(
                1, nl, n_tokens, hidden_bytes
            )
        except RuntimeError:
            return None
    ptr = int(obj)
    if ptr == 0:
        return None
    return _tensor_from_ptr(
        ptr, (1, nl, n_tokens, hidden_bytes), torch.uint8, device
    )


def _padded_slot_mapping(
    chunk_bids: list[int],
    block_size: int,
    chunk_tokens: int,
    offset_in_object: int,
    device: torch.device | str,
) -> torch.Tensor:
    """Slot map of length ``chunk_tokens``; skipped prefix/tail tokens are -1."""
    slots = torch.full(
        (chunk_tokens,), -1, dtype=torch.int64, device=device
    )
    if not chunk_bids:
        return slots
    real = _build_slot_mapping(chunk_bids, block_size, torch.device(device))
    n_tokens = int(real.numel())
    end = offset_in_object + n_tokens
    if offset_in_object < 0 or end > chunk_tokens:
        raise ValueError(
            f"slot window [{offset_in_object}:{end}) exceeds chunk_tokens="
            f"{chunk_tokens}"
        )
    slots[offset_in_object:end] = real
    return slots


def _kernel_stride_elems(
    shape_desc: PageBufferShapeDesc,
    engine_kv_format: EngineKVFormat | None = None,
) -> int:
    """Stride in int8 units (bytes). 0 when a single-plane group is tight.

    Tuple groups must not collapse ``stride == bs*nh*hs`` to 0: ``hs`` is the
    packed row (sum of planes), while each paged plane is padded to the pool
    block step (KG0: 4160 B vs latent width 128 B).
    """
    stride = int(getattr(shape_desc, "block_stride_elems", 0) or 0)
    if stride <= 0:
        return 0
    elem = int(shape_desc.element_size)
    if engine_kv_format in _SUPPORTED_TUPLE:
        return stride * elem
    tight = int(shape_desc.bs) * int(shape_desc.nh) * int(shape_desc.hs)
    if stride == tight:
        return 0
    return stride * elem


def _launch_chunk(
    paged_layers: list[Any],
    staging: torch.Tensor,
    slot_mapping: torch.Tensor,
    device: torch.device | str,
    shape_desc: PageBufferShapeDesc,
    engine_kv_format: EngineKVFormat,
    is_d2h: bool,
) -> None:
    global fused_kernel_launches
    import lmcache_ascend.c_ops as c_ops

    planes0 = _planes_of(paged_layers[0])
    page_buffer_size = int(shape_desc.nb) * int(shape_desc.bs)
    block_size = int(shape_desc.bs)
    stride_elems = _kernel_stride_elems(shape_desc, engine_kv_format)
    ptr_table = _ptr_table_for(paged_layers, torch.device(device))

    if engine_kv_format in _SUPPORTED_TUPLE:
        plane_bytes = [
            int(p.shape[-1]) * int(p.element_size()) for p in planes0
        ]
        k_bytes = plane_bytes[0]
        v_bytes = plane_bytes[1] if len(plane_bytes) > 1 else 0
        kvcache_format_raw = int(KVCacheFormat.MLA_KV.value)
        use_mla = True
        lmc_row_elems = int(staging.shape[-1])
    else:
        k_bytes = 0
        v_bytes = 0
        kvcache_format_raw = int(KVCacheFormat.MERGED_KV.value)
        use_mla = False
        lmc_row_elems = 0

    # direction: True = paged -> LMC (D2H gather), False = LMC -> paged.
    c_ops.multi_layer_kv_transfer(
        key_value=staging,
        key_value_ptrs=ptr_table,
        slot_mapping=slot_mapping,
        paged_memory_device=torch.device(device),
        page_buffer_size=page_buffer_size,
        direction=bool(is_d2h),
        use_mla=use_mla,
        kvcache_format_raw=kvcache_format_raw,
        k_hidden_dims=k_bytes,
        v_hidden_dims=v_bytes,
        dsa_hidden_dims=0,
        dsa_c8_scale_plane_bytes=0,
        paged_kv_block_size=block_size,
        block_stride_elems=stride_elems,
        lmc_row_elems=lmc_row_elems,
    )
    fused_kernel_launches += 1


def multi_layer_block_kv_transfer(
    paged_buffer_ptrs_tensor: "torch.Tensor | list",
    lmcache_objects_ptrs: list[int] | list[torch.Tensor],
    block_ids: torch.Tensor | list[int],
    device: torch.device | str,
    direction: TransferDirection,
    shape_desc: PageBufferShapeDesc,
    lmcache_chunk_size: int,
    engine_kv_format: EngineKVFormat,
    skip_prefix_n_blocks: int,
) -> None:
    """Native NPU block transfer; falls back to ``torch_ops`` per group."""
    global last_fallback_reason
    last_fallback_reason = ""

    paged_layers = _as_layers(paged_buffer_ptrs_tensor)
    reason = _fallback_reason(
        paged_layers, lmcache_objects_ptrs, device, engine_kv_format
    )
    if reason:
        last_fallback_reason = reason
        logger.debug(
            "npu_block_transfer fallback to torch_ops: %s fmt=%s",
            reason,
            getattr(engine_kv_format, "name", engine_kv_format),
        )
        _torch_ops_block_kv(
            paged_buffer_ptrs_tensor,
            lmcache_objects_ptrs,
            block_ids,
            device,
            direction,
            shape_desc,
            lmcache_chunk_size,
            engine_kv_format,
            skip_prefix_n_blocks,
        )
        return

    is_d2h = int(direction) == int(TransferDirection.D2H)
    block_size = int(shape_desc.bs)
    nl = int(shape_desc.nl)
    hidden_bytes = int(shape_desc.nh) * int(shape_desc.hs) * int(
        shape_desc.element_size
    )
    if engine_kv_format in _SUPPORTED_TUPLE:
        hidden_bytes = sum(
            int(p.shape[-1]) * int(p.element_size())
            for p in _planes_of(paged_layers[0])
        )

    bids = _block_ids_list(block_ids)
    n_block_ids = len(bids)
    blocks_per_object = lmcache_chunk_size // block_size
    chunk_tokens = int(lmcache_chunk_size)

    for object_idx, obj in enumerate(lmcache_objects_ptrs):
        valid = _valid_block_range_indices(
            object_idx,
            n_block_ids,
            blocks_per_object,
            block_size,
            skip_prefix_n_blocks,
        )
        if valid is None:
            continue
        idx_start, idx_end, offset_in_object = valid
        chunk_bids = bids[idx_start:idx_end]
        slot_mapping = _padded_slot_mapping(
            chunk_bids,
            block_size,
            chunk_tokens,
            offset_in_object,
            device,
        )
        staging = _uint8_staging(
            obj,
            nl=nl,
            n_tokens=chunk_tokens,
            hidden_bytes=hidden_bytes,
            device=device,
        )
        if staging is None:
            last_fallback_reason = "staging_unviewable"
            logger.debug(
                "npu_block_transfer fallback to torch_ops: %s",
                last_fallback_reason,
            )
            _torch_ops_block_kv(
                paged_buffer_ptrs_tensor,
                lmcache_objects_ptrs,
                block_ids,
                device,
                direction,
                shape_desc,
                lmcache_chunk_size,
                engine_kv_format,
                skip_prefix_n_blocks,
            )
            return
        _launch_chunk(
            paged_layers,
            staging,
            slot_mapping,
            device,
            shape_desc,
            engine_kv_format,
            is_d2h,
        )
