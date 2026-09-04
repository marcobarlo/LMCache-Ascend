# SPDX-License-Identifier: Apache-2.0
"""Fused NPU ``multi_layer_block_kv_transfer`` vs ``torch_ops``.

The primary correctness case is ``bs=2, hs=512, stride=1040``: ignoring
``blockStrideElems`` changes the result. Guard tests run on CPU.
"""

from __future__ import annotations

import inspect

import pytest
import torch

import lmcache.lmcache_native as lmcache_native
from lmcache_ascend.v1.multiprocess import npu_block_transfer as nbt
from lmcache_ascend.v1.multiprocess import npu_gather
from lmcache_ascend.v1.multiprocess import server_transfer_trace as stt

EngineKVFormat = lmcache_native.EngineKVFormat
TransferDirection = lmcache_native.TransferDirection
PageBufferShapeDesc = lmcache_native.PageBufferShapeDesc

_NPU_AVAILABLE = hasattr(torch, "npu") and torch.npu.is_available()
requires_npu = pytest.mark.skipif(not _NPU_AVAILABLE, reason="requires Ascend NPU")


def _kernel_has_stride_args() -> bool:
    try:
        import lmcache_ascend.c_ops as c_ops

        doc = c_ops.multi_layer_kv_transfer.__doc__ or ""
        return "block_stride_elems" in doc
    except Exception:
        return False


requires_stride_kernel = pytest.mark.skipif(
    not _kernel_has_stride_args(),
    reason="c_ops.multi_layer_kv_transfer missing block_stride_elems; rebuild",
)


def _shape_desc(
    *,
    nl: int,
    nb: int,
    bs: int,
    nh: int,
    hs: int,
    element_size: int,
    block_stride_elems: int,
    kv_size: int = 1,
) -> PageBufferShapeDesc:
    sd = PageBufferShapeDesc()
    sd.nl = nl
    sd.nb = nb
    sd.bs = bs
    sd.nh = nh
    sd.hs = hs
    sd.element_size = element_size
    sd.block_stride_elems = block_stride_elems
    sd.kv_size = kv_size
    return sd


def _padded_mla_layers(
    nl: int,
    nb: int,
    bs: int,
    hs: int,
    stride: int,
    device: str,
    dtype: torch.dtype,
    fill_unique: bool = True,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Per-layer views into a padded ``[NB, stride]`` pool, shape ``[NB, BS, HS]``."""
    tight = bs * hs
    pools: list[torch.Tensor] = []
    layers: list[torch.Tensor] = []
    for layer_i in range(nl):
        pool = torch.full((nb, stride), -1.0, dtype=dtype, device=device)
        view = pool[:, :tight].view(nb, bs, hs)
        if fill_unique:
            view.copy_(
                torch.arange(nb * tight, dtype=dtype, device=device).view(nb, bs, hs)
                + layer_i * 10_000
            )
        pools.append(pool)
        layers.append(view)
    return pools, layers


def _clone_padded(
    pools: list[torch.Tensor], bs: int, hs: int
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    tight = bs * hs
    cloned = [p.clone() for p in pools]
    layers = [p[:, :tight].view(p.shape[0], bs, hs) for p in cloned]
    return cloned, layers


def _run_both(
    layers_a: list,
    layers_b: list,
    objs_a: list[torch.Tensor],
    objs_b: list[torch.Tensor],
    block_ids: list[int],
    device: str,
    direction: TransferDirection,
    shape_desc: PageBufferShapeDesc,
    chunk_tokens: int,
    fmt: EngineKVFormat,
    skip_prefix_n_blocks: int = 0,
) -> None:
    nbt._torch_ops_block_kv(
        layers_a,
        objs_a,
        block_ids,
        device,
        direction,
        shape_desc,
        chunk_tokens,
        fmt,
        skip_prefix_n_blocks,
    )
    nbt.multi_layer_block_kv_transfer(
        layers_b,
        objs_b,
        block_ids,
        device,
        direction,
        shape_desc,
        chunk_tokens,
        fmt,
        skip_prefix_n_blocks,
    )


# ---------------------------------------------------------------------------
# CPU / no-NPU tests
# ---------------------------------------------------------------------------


def test_shim_registered_on_c_ops_with_tensor_annotation():
    fn = nbt.multi_layer_block_kv_transfer
    assert fn.__module__.startswith("lmcache_ascend")
    assert "torch_ops" not in fn.__module__
    ann = str(inspect.signature(fn).parameters["lmcache_objects_ptrs"].annotation)
    assert "Tensor" in ann
    import lmcache_ascend.c_ops as c_ops

    wrapped = c_ops.multi_layer_block_kv_transfer
    inner = getattr(wrapped, "__wrapped__", wrapped)
    assert inner.__module__.startswith("lmcache_ascend")
    assert "npu_block_transfer" in inner.__module__


def test_kernel_stride_elems_kg6_and_tight_and_kg0():
    kg6 = _shape_desc(
        nl=21, nb=8, bs=2, nh=1, hs=512, element_size=4, block_stride_elems=1040
    )
    assert nbt._kernel_stride_elems(kg6, EngineKVFormat.NL_X_NB_BS_HS) == 1040 * 4

    kg5 = _shape_desc(
        nl=21, nb=8, bs=2, nh=1, hs=2048, element_size=4, block_stride_elems=8192
    )
    assert nbt._kernel_stride_elems(kg5, EngineKVFormat.NL_X_NB_BS_HS) == 8192 * 4

    tight = _shape_desc(
        nl=4, nb=8, bs=32, nh=1, hs=512, element_size=2, block_stride_elems=16384
    )
    assert nbt._kernel_stride_elems(tight, EngineKVFormat.NL_X_NB_BS_HS) == 0

    kg0 = _shape_desc(
        nl=21, nb=8, bs=32, nh=1, hs=130, element_size=1, block_stride_elems=4160
    )
    assert nbt._kernel_stride_elems(kg0, EngineKVFormat.NL_X_TWO_X_NB_BS_HS) == 4160
    assert nbt._kernel_stride_elems(kg0, EngineKVFormat.NL_X_NB_BS_HS) == 0


def test_fallback_non_npu():
    layers = [torch.zeros(2, 2, 8)]
    objs = [torch.zeros(1, 4, 8)]
    reason = nbt._fallback_reason(
        layers, objs, "cpu", EngineKVFormat.NL_X_NB_BS_HS
    )
    assert reason == "non-npu"


def test_fallback_unsupported_format():
    layers = [torch.zeros(2, 2, 2, 8)]
    objs = [torch.zeros(2, 1, 4, 16)]
    reason = nbt._fallback_reason(
        layers, objs, "npu:0", EngineKVFormat.NL_X_TWO_NB_BS_NH_HS
    )
    assert reason.startswith("unsupported_format")


def test_fallback_disagreeing_plane_strides():
    p0 = torch.zeros(2, 32, 128)
    p1 = torch.zeros(2, 32, 2)
    layers = [(p0, p1)]
    objs = [torch.zeros(1, 32, 130, dtype=torch.uint8)]
    reason = nbt._fallback_reason(
        layers, objs, "npu:0", EngineKVFormat.NL_X_TWO_X_NB_BS_HS
    )
    assert reason == "disagreeing_plane_strides"


def test_fallback_non_dense_width():
    plane = torch.zeros(2, 4, 8).transpose(-1, -2)
    reason = nbt._fallback_reason(
        [plane],
        [torch.zeros(1, 8, 8)],
        "npu:0",
        EngineKVFormat.NL_X_NB_BS_HS,
    )
    assert reason == "non_dense_width"


def test_fallback_num_planes_on_single_plane_format():
    layers = [(torch.zeros(2, 4, 8), torch.zeros(2, 4, 8))]
    reason = nbt._fallback_reason(
        layers,
        [torch.zeros(1, 8, 8)],
        "npu:0",
        EngineKVFormat.NL_X_NB_BS_HS,
    )
    assert reason.startswith("num_planes")


def test_fallback_unaligned_mla_scale_plane():
    """KG0 2 B scale is no longer a host fallback; v3 copies it in-kernel."""
    pool = torch.zeros(2, 4160, dtype=torch.uint8)
    latent = pool[:, : 32 * 128].view(2, 32, 128)
    scale = pool[:, 32 * 128 :].view(torch.float16).view(2, 32, 1)
    reason = nbt._fallback_reason(
        [(latent, scale)],
        [torch.zeros(1, 32, 130, dtype=torch.uint8)],
        "npu:0",
        EngineKVFormat.NL_X_TWO_X_NB_BS_HS,
    )
    assert reason == ""


def test_fallback_staging_unaligned():
    plane = torch.zeros(2, 4, 8)
    blob = torch.zeros(64, dtype=torch.uint8)
    misaligned = blob[1:]
    reason = nbt._fallback_reason(
        [plane],
        [misaligned],
        "npu:0",
        EngineKVFormat.NL_X_NB_BS_HS,
    )
    assert reason == "staging_unaligned"


@requires_npu
@requires_stride_kernel
@pytest.mark.parametrize("direction", [TransferDirection.D2H, TransferDirection.H2D])
def test_aligned_mla_tuple_packed_row_matches_torch_ops(direction: TransferDirection):
    """Phase 2 kernel path: 128 B + 32 B planes, 160 B token row (32 B aligned)."""
    device = "npu"
    nl, nb, bs = 2, 4, 2
    w0, w1 = 128, 32
    hidden_bytes = w0 + w1
    stride = bs * hidden_bytes
    chunk = 4
    fmt = EngineKVFormat.NL_X_TWO_X_NB_BS_HS
    sd = _shape_desc(
        nl=nl,
        nb=nb,
        bs=bs,
        nh=1,
        hs=hidden_bytes,
        element_size=1,
        block_stride_elems=stride,
        kv_size=1,
    )
    pools: list[torch.Tensor] = []
    for layer_i in range(nl):
        pool = torch.arange(
            nb * stride, dtype=torch.int32, device=device
        ).to(torch.uint8).view(nb, stride)
        pool.add_(layer_i)
        pools.append(pool)

    def views(ps: list[torch.Tensor]) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return [
            (p[:, : bs * w0].view(nb, bs, w0), p[:, bs * w0 :].view(nb, bs, w1))
            for p in ps
        ]

    pools_a = [p.clone() for p in pools]
    pools_b = [p.clone() for p in pools]
    layers_a = views(pools_a)
    layers_b = views(pools_b)
    if direction == TransferDirection.D2H:
        objs_a = [torch.zeros(nl, chunk, hidden_bytes, dtype=torch.uint8, device=device)]
        objs_b = [torch.zeros(nl, chunk, hidden_bytes, dtype=torch.uint8, device=device)]
    else:
        src = (
            torch.arange(nl * chunk * hidden_bytes, dtype=torch.int32, device=device)
            .to(torch.uint8)
            .view(nl, chunk, hidden_bytes)
        )
        objs_a = [src.clone()]
        objs_b = [src.clone()]
        for p in pools_a + pools_b:
            p.zero_()

    nbt.last_fallback_reason = ""
    nbt.fused_kernel_launches = 0
    _run_both(
        layers_a,
        layers_b,
        objs_a,
        objs_b,
        [0, 1],
        device,
        direction,
        sd,
        chunk,
        fmt,
    )
    torch.npu.synchronize()
    assert nbt.last_fallback_reason == ""
    assert nbt.fused_kernel_launches == 1
    if direction == TransferDirection.D2H:
        assert torch.equal(objs_a[0].cpu(), objs_b[0].cpu())
    else:
        for pa, pb in zip(pools_a, pools_b):
            assert torch.equal(pa.cpu(), pb.cpu())

    plane = torch.zeros(2, 4, 8)
    blob = torch.zeros(64, dtype=torch.uint8)
    misaligned = blob[1:]
    reason = nbt._fallback_reason(
        [plane],
        [misaligned],
        "npu:0",
        EngineKVFormat.NL_X_NB_BS_HS,
    )
    assert reason == "staging_unaligned"


def test_fallback_cpu_transfer_matches_torch_ops():
    """CPU device hits the torch_ops fallback and still copies correctly."""
    nl, nb, bs, hs = 2, 4, 2, 8
    dtype = torch.float32
    layers = [
        torch.arange(nb * bs * hs, dtype=dtype).view(nb, bs, hs) + i * 100
        for i in range(nl)
    ]
    sd = _shape_desc(
        nl=nl, nb=nb, bs=bs, nh=1, hs=hs, element_size=4, block_stride_elems=bs * hs
    )
    chunk = 4
    objs_a = [torch.zeros(nl, chunk, hs, dtype=dtype)]
    objs_b = [torch.zeros(nl, chunk, hs, dtype=dtype)]
    bids = [0, 1]
    nbt._torch_ops_block_kv(
        [t.clone() for t in layers],
        objs_a,
        bids,
        "cpu",
        TransferDirection.D2H,
        sd,
        chunk,
        EngineKVFormat.NL_X_NB_BS_HS,
        0,
    )
    nbt.last_fallback_reason = ""
    nbt.multi_layer_block_kv_transfer(
        [t.clone() for t in layers],
        objs_b,
        bids,
        "cpu",
        TransferDirection.D2H,
        sd,
        chunk,
        EngineKVFormat.NL_X_NB_BS_HS,
        0,
    )
    assert nbt.last_fallback_reason == "non-npu"
    assert torch.equal(objs_a[0], objs_b[0])


def test_trace_counters_include_fused_launches():
    stt.reset_counters()
    snap = stt.snapshot_counters()
    assert "block_kv_fused_launches" in snap
    assert snap["block_kv_fused_launches"] == 0.0


def test_npu_gather_slot_mapping_unchanged():
    sm = npu_gather._build_slot_mapping([3, 7], block_size=2, device="cpu")
    assert sm.tolist() == [6, 7, 14, 15]


def test_multi_layer_kv_transfer_new_args_defaulted():
    import lmcache_ascend.c_ops as c_ops

    doc = c_ops.multi_layer_kv_transfer.__doc__ or ""
    if "block_stride_elems" not in doc:
        pytest.skip("extension not rebuilt with block_stride_elems")
    assert "lmc_row_elems" in doc
    assert "paged_kv_block_size" in doc


# ---------------------------------------------------------------------------
# NPU equality vs torch_ops
# ---------------------------------------------------------------------------


@requires_npu
@requires_stride_kernel
@pytest.mark.parametrize("direction", [TransferDirection.D2H, TransferDirection.H2D])
def test_kg6_stride_1040_matches_torch_ops(direction: TransferDirection):
    """Decisive case: bs=2 hs=512 es=4 stride=1040 (tight 1024)."""
    device = "npu"
    nl, nb, bs, hs, stride = 2, 4, 2, 512, 1040
    dtype = torch.float32
    chunk = 4
    fmt = EngineKVFormat.NL_X_NB_BS_HS
    sd = _shape_desc(
        nl=nl, nb=nb, bs=bs, nh=1, hs=hs, element_size=4, block_stride_elems=stride
    )
    pools, _layers = _padded_mla_layers(nl, nb, bs, hs, stride, device, dtype)
    pools_a, layers_a = _clone_padded(pools, bs, hs)
    pools_b, layers_b = _clone_padded(pools, bs, hs)
    if direction == TransferDirection.D2H:
        objs_a = [torch.zeros(nl, chunk, hs, dtype=dtype, device=device)]
        objs_b = [torch.zeros(nl, chunk, hs, dtype=dtype, device=device)]
    else:
        src = torch.arange(nl * chunk * hs, dtype=dtype, device=device).view(
            nl, chunk, hs
        )
        objs_a = [src.clone()]
        objs_b = [src.clone()]
        for p in pools_a + pools_b:
            p.fill_(-1.0)
    nbt.fused_kernel_launches = 0
    nbt.last_fallback_reason = ""
    _run_both(
        layers_a,
        layers_b,
        objs_a,
        objs_b,
        [0, 1],
        device,
        direction,
        sd,
        chunk,
        fmt,
    )
    torch.npu.synchronize()
    assert nbt.last_fallback_reason == ""
    assert nbt.fused_kernel_launches == 1
    if direction == TransferDirection.D2H:
        assert torch.equal(objs_a[0].cpu(), objs_b[0].cpu())
        # Padding must not leak into the packed token row.
        assert not torch.any(objs_b[0].cpu() == -1)
    else:
        for la, lb in zip(layers_a, layers_b):
            assert torch.equal(la.cpu(), lb.cpu())
        for pa, pb in zip(pools_a, pools_b):
            assert torch.equal(pa.cpu(), pb.cpu())


@requires_npu
@requires_stride_kernel
@pytest.mark.parametrize("direction", [TransferDirection.D2H, TransferDirection.H2D])
def test_kg5_stride_8192_matches_torch_ops(direction: TransferDirection):
    """Padded but divisible: bs=2 hs=2048 stride=8192 (tight 4096)."""
    device = "npu"
    nl, nb, bs, hs, stride = 2, 4, 2, 2048, 8192
    dtype = torch.float32
    chunk = 4
    fmt = EngineKVFormat.NL_X_NB_BS_HS
    sd = _shape_desc(
        nl=nl, nb=nb, bs=bs, nh=1, hs=hs, element_size=4, block_stride_elems=stride
    )
    pools, _layers = _padded_mla_layers(nl, nb, bs, hs, stride, device, dtype)
    pools_a, layers_a = _clone_padded(pools, bs, hs)
    pools_b, layers_b = _clone_padded(pools, bs, hs)
    if direction == TransferDirection.D2H:
        objs_a = [torch.zeros(nl, chunk, hs, dtype=dtype, device=device)]
        objs_b = [torch.zeros(nl, chunk, hs, dtype=dtype, device=device)]
    else:
        src = torch.arange(nl * chunk * hs, dtype=dtype, device=device).view(
            nl, chunk, hs
        )
        objs_a = [src.clone()]
        objs_b = [src.clone()]
        for p in pools_a + pools_b:
            p.fill_(-1.0)
    nbt.last_fallback_reason = ""
    _run_both(
        layers_a,
        layers_b,
        objs_a,
        objs_b,
        [0, 1],
        device,
        direction,
        sd,
        chunk,
        fmt,
    )
    torch.npu.synchronize()
    assert nbt.last_fallback_reason == ""
    if direction == TransferDirection.D2H:
        assert torch.equal(objs_a[0].cpu(), objs_b[0].cpu())
    else:
        for la, lb in zip(layers_a, layers_b):
            assert torch.equal(la.cpu(), lb.cpu())


@requires_npu
@requires_stride_kernel
def test_skip_prefix_and_tight_bf16_roundtrip():
    device = "npu"
    nl, nb, bs, hs = 2, 4, 2, 32
    dtype = torch.bfloat16
    stride = bs * hs
    chunk = 4
    fmt = EngineKVFormat.NL_X_NB_BS_HS
    sd = _shape_desc(
        nl=nl, nb=nb, bs=bs, nh=1, hs=hs, element_size=2, block_stride_elems=stride
    )
    _, layers = _padded_mla_layers(nl, nb, bs, hs, stride, device, dtype)
    layers_a = [t.clone() for t in layers]
    layers_b = [t.clone() for t in layers]
    objs_a = [torch.zeros(nl, chunk, hs, dtype=dtype, device=device)]
    objs_b = [torch.zeros(nl, chunk, hs, dtype=dtype, device=device)]
    nbt.last_fallback_reason = ""
    _run_both(
        layers_a,
        layers_b,
        objs_a,
        objs_b,
        [0, 1],
        device,
        TransferDirection.D2H,
        sd,
        chunk,
        fmt,
        skip_prefix_n_blocks=1,
    )
    torch.npu.synchronize()
    assert nbt.last_fallback_reason == ""
    assert torch.equal(objs_a[0].cpu(), objs_b[0].cpu())
    # Prefix tokens of the object stay zero.
    assert torch.all(objs_b[0][:, :bs, :].cpu() == 0)


@requires_npu
@requires_stride_kernel
@pytest.mark.parametrize("direction", [TransferDirection.D2H, TransferDirection.H2D])
def test_kg0_mla_tuple_packed_row_matches_torch_ops(direction: TransferDirection):
    """KG0 geometry: 128 B int8 + 2 B fp16. v3 fuses the 2-byte scale plane."""
    device = "npu"
    nl, nb, bs = 2, 4, 32
    latent_w, scale_w = 128, 1
    block_bytes = 4160
    hidden_bytes = 130
    chunk = 32
    fmt = EngineKVFormat.NL_X_TWO_X_NB_BS_HS
    sd = _shape_desc(
        nl=nl,
        nb=nb,
        bs=bs,
        nh=1,
        hs=hidden_bytes,
        element_size=1,
        block_stride_elems=block_bytes,
        kv_size=1,
    )

    def make_layers() -> tuple[list[torch.Tensor], list[tuple[torch.Tensor, torch.Tensor]]]:
        pools: list[torch.Tensor] = []
        layers: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer_i in range(nl):
            pool = torch.zeros(nb, block_bytes, dtype=torch.uint8, device=device)
            latent = pool[:, : bs * latent_w].view(nb, bs, latent_w)
            scale = pool[:, bs * latent_w :].view(torch.float16).view(nb, bs, scale_w)
            latent.copy_(
                (
                    torch.arange(nb * bs * latent_w, device=device, dtype=torch.int32)
                    % 251
                )
                .to(torch.uint8)
                .view(nb, bs, latent_w)
                + layer_i
            )
            scale.copy_(
                torch.arange(nb * bs, device=device, dtype=torch.float16).view(
                    nb, bs, scale_w
                )
                + layer_i
            )
            pools.append(pool)
            layers.append((latent, scale))
        return pools, layers

    pools, _layers = make_layers()
    pools_a = [p.clone() for p in pools]
    pools_b = [p.clone() for p in pools]

    def views(ps: list[torch.Tensor]) -> list[tuple[torch.Tensor, torch.Tensor]]:
        out: list[tuple[torch.Tensor, torch.Tensor]] = []
        for p in ps:
            latent = p[:, : bs * latent_w].view(nb, bs, latent_w)
            scale = p[:, bs * latent_w :].view(torch.float16).view(nb, bs, scale_w)
            out.append((latent, scale))
        return out

    layers_a = views(pools_a)
    layers_b = views(pools_b)
    if direction == TransferDirection.D2H:
        objs_a = [torch.zeros(nl, chunk, hidden_bytes, dtype=torch.uint8, device=device)]
        objs_b = [torch.zeros(nl, chunk, hidden_bytes, dtype=torch.uint8, device=device)]
    else:
        # Scale bytes must be valid fp16; torch_ops index_copy_ views that
        # plane as float16 and NPU canonicalizes NaN/Inf bit patterns.
        src = torch.zeros(nl, chunk, hidden_bytes, dtype=torch.uint8, device=device)
        src[:, :, :latent_w] = (
            torch.arange(nl * chunk * latent_w, dtype=torch.int32, device=device)
            .to(torch.uint8)
            .view(nl, chunk, latent_w)
        )
        src[:, :, latent_w:] = (
            torch.arange(nl * chunk, device=device, dtype=torch.float16)
            .view(nl, chunk, 1)
            .view(torch.uint8)
            .view(nl, chunk, 2)
        )
        objs_a = [src.clone()]
        objs_b = [src.clone()]
        for p in pools_a + pools_b:
            p.zero_()

    nbt.last_fallback_reason = ""
    nbt.fused_kernel_launches = 0
    _run_both(
        layers_a,
        layers_b,
        objs_a,
        objs_b,
        [0],
        device,
        direction,
        sd,
        chunk,
        fmt,
    )
    torch.npu.synchronize()
    assert nbt.last_fallback_reason == ""
    assert nbt.fused_kernel_launches == 1
    if direction == TransferDirection.D2H:
        assert torch.equal(objs_a[0].cpu(), objs_b[0].cpu())
    else:
        for pa, pb in zip(pools_a, pools_b):
            assert torch.equal(pa.cpu(), pb.cpu())


@requires_npu
@requires_stride_kernel
def test_kg0_skip_prefix_matches_torch_ops():
    """Whole-block skip_prefix on KG0 still fuses; prefix tokens stay zero."""
    device = "npu"
    nl, nb, bs = 2, 4, 32
    latent_w, scale_w = 128, 1
    block_bytes = 4160
    hidden_bytes = 130
    chunk = 64
    fmt = EngineKVFormat.NL_X_TWO_X_NB_BS_HS
    sd = _shape_desc(
        nl=nl,
        nb=nb,
        bs=bs,
        nh=1,
        hs=hidden_bytes,
        element_size=1,
        block_stride_elems=block_bytes,
        kv_size=1,
    )
    pools: list[torch.Tensor] = []
    for layer_i in range(nl):
        pool = torch.zeros(nb, block_bytes, dtype=torch.uint8, device=device)
        latent = pool[:, : bs * latent_w].view(nb, bs, latent_w)
        scale = pool[:, bs * latent_w :].view(torch.float16).view(nb, bs, scale_w)
        latent.copy_(
            (torch.arange(nb * bs * latent_w, device=device, dtype=torch.int32) % 251)
            .to(torch.uint8)
            .view(nb, bs, latent_w)
            + layer_i
        )
        scale.copy_(
            torch.arange(nb * bs, device=device, dtype=torch.float16).view(nb, bs, scale_w)
            + layer_i
        )
        pools.append(pool)

    def views(ps: list[torch.Tensor]) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return [
            (
                p[:, : bs * latent_w].view(nb, bs, latent_w),
                p[:, bs * latent_w :].view(torch.float16).view(nb, bs, scale_w),
            )
            for p in ps
        ]

    pools_a = [p.clone() for p in pools]
    pools_b = [p.clone() for p in pools]
    objs_a = [torch.zeros(nl, chunk, hidden_bytes, dtype=torch.uint8, device=device)]
    objs_b = [torch.zeros(nl, chunk, hidden_bytes, dtype=torch.uint8, device=device)]
    nbt.last_fallback_reason = ""
    nbt.fused_kernel_launches = 0
    _run_both(
        views(pools_a),
        views(pools_b),
        objs_a,
        objs_b,
        [0, 1],
        device,
        TransferDirection.D2H,
        sd,
        chunk,
        fmt,
        skip_prefix_n_blocks=1,
    )
    torch.npu.synchronize()
    assert nbt.last_fallback_reason == ""
    assert nbt.fused_kernel_launches == 1
    assert torch.equal(objs_a[0].cpu(), objs_b[0].cpu())
    assert torch.all(objs_b[0][:, :bs, :].cpu() == 0)
