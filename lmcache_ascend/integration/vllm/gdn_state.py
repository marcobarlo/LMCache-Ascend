# SPDX-License-Identifier: Apache-2.0
"""Present Ascend GDN state slices as format-17 planes.

vLLM-Ascend registers a gated-delta layer as a list of contiguous slices
(conv, SSM) carved out of one int8 pool. Each slice's block stride is its
own byte size, not the padded page, so the CUDA page-view edit does not
apply. The NPU block kernel only copies format 16 and format 17, so each
slice is re-viewed here as one uint8 ``[num_blocks, block_size, 1, width]``
plane. Planes of one layer stay distinct: they are not packed into a single
blob and they do not join the attention ``(K, V)`` group.

``block_size * width`` equals the bytes of one block. ``block_size`` is the
largest divisor of the scheduler block size that also divides every plane,
so a stored block copies the whole slice, and each row stays within the
kernel's per-segment UB budget. The plane's dim-0 stride is that slice's
own block stride.
"""

# Future
from __future__ import annotations

# Standard
from collections.abc import Mapping, Sequence
from typing import Any
import logging
import sys

# Third Party
import torch

logger = logging.getLogger(__name__)

# DataCopy alignment. Matches ``kGmAlignBytes`` in the block kernel.
_GM_ALIGN_BYTES = 32
# Default per-segment UB. Queue depth is 1, so the segment is the whole
# ``kBlockTransferUbBytes`` budget (128 KiB). A row at this size still passes
# ``align_up32(payload) <= ub_segment`` on that built-in budget.
_MAX_ROW_BYTES = 128 * 1024

_INSTALLED = False


def block_bytes(state: torch.Tensor) -> int:
    """Return the dense per-block byte size of a contiguous state slice.

    Args:
        state: One GDN state with the block count on dim 0.

    Returns:
        Bytes in one block, equal to ``stride(0) * element_size``.

    Raises:
        ValueError: If the slice is empty or not densely packed per block.
    """
    num_blocks = int(state.shape[0])
    if num_blocks <= 0:
        raise ValueError("GDN state has no blocks to copy")
    if not state.is_contiguous():
        raise ValueError(
            "GDN state must be contiguous so its block stride is the "
            f"slice itself, got stride {tuple(state.stride())}"
        )
    nbytes = int(state.numel()) * int(state.element_size()) // num_blocks
    stride_bytes = int(state.stride(0)) * int(state.element_size())
    if stride_bytes != nbytes:
        raise ValueError(
            f"GDN state block stride is {stride_bytes} bytes but the dense "
            f"block is {nbytes} bytes"
        )
    return nbytes


def choose_block_size(byte_counts: Sequence[int], tokens_per_block: int) -> int:
    """Pick a shared plane block size that copies every state in full.

    Args:
        byte_counts: Per-block byte size of each state in the layer.
        tokens_per_block: Scheduler tokens covered by one block id.

    Returns:
        The largest ``block_size`` that divides ``tokens_per_block`` and
        every byte count, with each row at most ``_MAX_ROW_BYTES``.

    Raises:
        ValueError: If a state is not 32-byte aligned or no block size fits.
    """
    if tokens_per_block <= 0:
        raise ValueError(
            f"tokens_per_block must be positive, got {tokens_per_block}"
        )
    for nbytes in byte_counts:
        if nbytes <= 0 or nbytes % _GM_ALIGN_BYTES != 0:
            raise ValueError(
                f"GDN state block is {nbytes} bytes; the NPU copy needs a "
                f"positive multiple of {_GM_ALIGN_BYTES}"
            )
    best = 0
    divisor = 1
    while divisor * divisor <= tokens_per_block:
        if tokens_per_block % divisor == 0:
            for candidate in (divisor, tokens_per_block // divisor):
                if any(nbytes % candidate != 0 for nbytes in byte_counts):
                    continue
                if any(nbytes // candidate > _MAX_ROW_BYTES for nbytes in byte_counts):
                    continue
                if candidate > best:
                    best = candidate
        divisor += 1
    if best == 0:
        raise ValueError(
            "no block size divides tokens_per_block="
            f"{tokens_per_block} and state bytes {list(byte_counts)} with "
            f"each row <= {_MAX_ROW_BYTES} bytes"
        )
    return best


def tile_uint8_planes(state: torch.Tensor, block_size: int) -> tuple[torch.Tensor, ...]:
    """Re-view one state slice as one or more uint8 format-17 planes.

    ``block_size * width`` tiles cover the block exactly. A state whose row
    exceeds the UB budget is split into consecutive row tiles; each tile
    keeps the slice's block stride (``stride(0)``).

    Args:
        state: Contiguous state with the block count on dim 0.
        block_size: Plane block size. Must divide the block's byte size.

    Returns:
        Planes of shape ``[num_blocks, block_size, 1, tile_width]``.
    """
    nbytes = block_bytes(state)
    if nbytes % block_size != 0:
        raise ValueError(
            f"block size {block_size} does not divide the {nbytes}-byte state"
        )
    width = nbytes // block_size
    num_blocks = int(state.shape[0])
    raw = state.view(torch.uint8)
    base = int(raw.storage_offset())
    planes: list[torch.Tensor] = []
    consumed = 0
    while consumed < width:
        tile = min(width - consumed, _MAX_ROW_BYTES)
        planes.append(
            raw.as_strided(
                (num_blocks, block_size, 1, tile),
                (nbytes, tile, tile, 1),
                storage_offset=base + consumed * block_size,
            )
        )
        consumed += tile
    return tuple(planes)


def _layer_specs(kv_cache_config: Any) -> dict[str, Any]:
    specs: dict[str, Any] = {}
    if kv_cache_config is None:
        return specs
    for group in getattr(kv_cache_config, "kv_cache_groups", ()) or ():
        per_layer = getattr(group.kv_cache_spec, "kv_cache_specs", None)
        for name in group.layer_names:
            specs[name] = per_layer[name] if per_layer else group.kv_cache_spec
    return specs


def _is_mamba(spec: Any) -> bool:
    # Third Party
    from vllm.v1.kv_cache_interface import KVCacheSpecKind, get_kv_cache_spec_kind

    return get_kv_cache_spec_kind(spec) == KVCacheSpecKind.MAMBA


def _is_attention_pair(entry: Any) -> bool:
    if not isinstance(entry, (list, tuple)) or len(entry) != 2:
        return False
    key, value = entry
    return (
        isinstance(key, torch.Tensor)
        and isinstance(value, torch.Tensor)
        and key.ndim == 4
        and tuple(key.shape) == tuple(value.shape)
    )


def _review_attention_pair(entry: Any, logical_block_size: int) -> Any:
    """Re-view a ``(K, V)`` pair when the kernel block size is finer.

    Leaves the pair untouched when the kernel block already matches, or
    when the logical block is not an exact multiple of the kernel block.
    """
    if not _is_attention_pair(entry):
        return entry
    key = entry[0]
    kernel_block_size = int(key.shape[1])
    if (
        kernel_block_size == logical_block_size
        or logical_block_size % kernel_block_size != 0
    ):
        return entry
    ratio = logical_block_size // kernel_block_size
    num_kernel_pages = int(key.shape[0])
    if (
        num_kernel_pages % ratio != 0
        or not key.is_contiguous()
        or not entry[1].is_contiguous()
    ):
        return entry
    num_heads = int(key.shape[2])
    head_size = int(key.shape[3])
    num_blocks = num_kernel_pages // ratio
    viewed = tuple(
        plane.view(num_blocks, logical_block_size, num_heads, head_size)
        for plane in entry
    )
    if isinstance(entry, list):
        return list(viewed)
    return viewed


def _already_uint8_planes(states: Sequence[torch.Tensor]) -> bool:
    return all(
        state.dtype == torch.uint8
        and state.ndim == 4
        and int(state.shape[2]) == 1
        for state in states
    )


def present_registered_caches(
    kv_cache_config: Any,
    kv_caches: Mapping[str, Any],
) -> dict[str, Any]:
    """Return caches with GDN slices as uint8 planes and attention re-viewed.

    Args:
        kv_cache_config: vLLM ``KVCacheConfig``. ``None`` returns a copy.
        kv_caches: Registered tensors keyed by layer name.

    Returns:
        A new dict. Attention ``(K, V)`` pairs stay on the format-16 path.
        Each remaining Mamba state list becomes a tuple of uint8 planes.
    """
    if kv_cache_config is None:
        return dict(kv_caches)
    specs = _layer_specs(kv_cache_config)
    presented: dict[str, Any] = {}
    for name, entry in kv_caches.items():
        spec = specs.get(name)
        if spec is None:
            presented[name] = entry
            continue
        if _is_mamba(spec) and isinstance(entry, (list, tuple)):
            states = [state for state in entry if isinstance(state, torch.Tensor)]
            if len(states) != len(entry) or not states or _already_uint8_planes(states):
                presented[name] = entry
                continue
            num_blocks = {int(state.shape[0]) for state in states}
            if len(num_blocks) != 1:
                raise ValueError(
                    f"layer {name} GDN states cover different block counts "
                    f"{sorted(num_blocks)}"
                )
            counts = [block_bytes(state) for state in states]
            block_size = choose_block_size(counts, int(spec.block_size))
            planes: list[torch.Tensor] = []
            for state in states:
                planes.extend(tile_uint8_planes(state, block_size))
            if len(planes) > 4:
                raise ValueError(
                    f"layer {name} needs {len(planes)} uint8 planes; the NPU "
                    "kernel copies at most 4"
                )
            logger.info(
                "GDN layer %s: %d uint8 planes, block_size=%d, rows=%s",
                name,
                len(planes),
                block_size,
                [int(plane.shape[-1]) for plane in planes],
            )
            presented[name] = tuple(planes)
            continue
        logical = getattr(spec, "block_size", None)
        if logical is not None and _is_attention_pair(entry):
            presented[name] = _review_attention_pair(entry, int(logical))
            continue
        presented[name] = entry
    return presented


def install_gdn_state_planes() -> None:
    """Wrap KV-cache registration so Ascend GDN slices are format-17 planes.

    The connector binds ``apply_kv_cache_group_edits`` by name at import, so
    both the defining module and an already-imported connector are updated.
    """
    global _INSTALLED
    if _INSTALLED:
        return
    # First Party
    import lmcache.integration.vllm.kv_cache_group_edits as edits

    original = edits.apply_kv_cache_group_edits
    if getattr(original, "_lmcache_ascend_gdn_planes", False):
        _INSTALLED = True
        return

    def apply_kv_cache_group_edits(kv_cache_config, kv_caches, layout_hints):
        edited = original(kv_cache_config, kv_caches, layout_hints)
        return present_registered_caches(kv_cache_config, edited)

    apply_kv_cache_group_edits._lmcache_ascend_gdn_planes = True  # type: ignore[attr-defined]
    edits.apply_kv_cache_group_edits = apply_kv_cache_group_edits
    connector = sys.modules.get("lmcache.integration.vllm.lmcache_mp_connector")
    if connector is not None and hasattr(connector, "apply_kv_cache_group_edits"):
        connector.apply_kv_cache_group_edits = apply_kv_cache_group_edits
    _INSTALLED = True
