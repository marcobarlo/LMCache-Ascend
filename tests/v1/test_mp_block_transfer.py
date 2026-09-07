# SPDX-License-Identifier: Apache-2.0
"""Direct correctness tests for the block-level MP-mode AscendC kernel.

These tests intentionally bypass the MP server and its CPU staging buffers.  They
exercise the device operator itself: the interleaved per-layer ``[K, V, ...]``
pointer table, its phase-1 multi-object launch loop, prefix-block skipping, and
an engine tensor whose dim-0 block stride contains padding.
"""

from __future__ import annotations

import pytest
import torch

import lmcache.lmcache_native as native
import lmcache_ascend.c_ops as lmc_ops


def _npu_available() -> bool:
    return hasattr(torch, "npu") and torch.npu.is_available()


def _shape_desc(
    *,
    nl: int,
    nb: int,
    bs: int,
    nh: int,
    hs: int,
    element_size: int,
    block_stride_elems: int,
) -> object:
    desc = native.PageBufferShapeDesc()
    desc.kv_size = 2
    desc.nl = nl
    desc.nb = nb
    desc.bs = bs
    desc.nh = nh
    desc.hs = hs
    desc.element_size = element_size
    desc.block_stride_elems = block_stride_elems
    desc.dtype = torch.float16
    return desc


def _make_paged_tensor(
    *,
    nb: int,
    bs: int,
    nh: int,
    hs: int,
    padded: bool,
    offset: int,
    device: torch.device,
) -> torch.Tensor:
    """Create a distinct fp16 paged tensor, optionally with dim-0 padding."""
    if padded:
        storage = torch.empty((nb, bs + 1, nh, hs), dtype=torch.float16, device=device)
        tensor = storage[:, :bs]
    else:
        tensor = torch.empty((nb, bs, nh, hs), dtype=torch.float16, device=device)
    values = torch.arange(
        tensor.numel(), dtype=torch.float32, device=device
    ).reshape_as(tensor)
    tensor.copy_((values + offset).to(torch.float16))
    return tensor


@pytest.mark.skipif(not _npu_available(), reason="Ascend NPU required")
@pytest.mark.parametrize("padded", [False, True])
def test_multi_layer_block_transfer_round_trip_with_prefix_skip(padded: bool) -> None:
    """D2H then H2D restores only non-skipped blocks for two MP objects."""
    device = torch.device("npu:0")
    nl, nb, bs, nh, hs = 2, 8, 4, 2, 8
    chunk = 2 * bs
    blocks_per_object = chunk // bs
    num_objects = 2
    hidden = nh * hs

    # Each layer contributes an interleaved K/V entry, exactly as vLLM-Ascend
    # SEPARATE_KV does. The selected block IDs are deliberately non-contiguous.
    paged: list[tuple[torch.Tensor, torch.Tensor]] = []
    ptrs: list[int] = []
    for layer in range(nl):
        key = _make_paged_tensor(
            nb=nb,
            bs=bs,
            nh=nh,
            hs=hs,
            padded=padded,
            offset=10_000 * layer,
            device=device,
        )
        value = _make_paged_tensor(
            nb=nb,
            bs=bs,
            nh=nh,
            hs=hs,
            padded=padded,
            offset=10_000 * layer + 5_000,
            device=device,
        )
        paged.append((key, value))
        ptrs.extend((key.data_ptr(), value.data_ptr()))

    pointer_table = torch.tensor(ptrs, dtype=torch.int64, device=device)
    block_ids = torch.tensor([1, 3, 4, 6], dtype=torch.int64, device=device)
    stride = (bs + 1 if padded else bs) * hidden
    desc = _shape_desc(
        nl=nl,
        nb=nb,
        bs=bs,
        nh=nh,
        hs=hs,
        element_size=torch.empty((), dtype=torch.float16).element_size(),
        block_stride_elems=stride if padded else 0,
    )
    objects = [
        torch.zeros((2, nl, chunk, hidden), dtype=torch.float16, device=device)
        for _ in range(num_objects)
    ]

    # Store engine -> LMCache. Skip the first block in *each* object.
    lmc_ops.multi_layer_block_kv_transfer(
        pointer_table,
        [obj.data_ptr() for obj in objects],
        block_ids,
        device,
        native.TransferDirection.D2H,
        desc,
        chunk,
        native.EngineKVFormat.NL_X_TWO_X_NB_BS_NH_HS,
        1,
    )
    torch.npu.synchronize()

    block_ids_cpu = block_ids.cpu().tolist()
    for object_idx, obj in enumerate(objects):
        assert torch.count_nonzero(obj[:, :, :bs]) == 0
        for local_block in range(1, blocks_per_object):
            engine_block = block_ids_cpu[object_idx * blocks_per_object + local_block]
            token_slice = slice(local_block * bs, (local_block + 1) * bs)
            for layer, (key, value) in enumerate(paged):
                torch.testing.assert_close(
                    obj[0, layer, token_slice].reshape_as(key[engine_block]),
                    key[engine_block],
                )
                torch.testing.assert_close(
                    obj[1, layer, token_slice].reshape_as(value[engine_block]),
                    value[engine_block],
                )

    # Clear the engine then restore. Prefix blocks remain clear; the remaining
    # block in each object is restored byte-for-byte from the 2LTD object.
    for key, value in paged:
        key.zero_()
        value.zero_()
    lmc_ops.multi_layer_block_kv_transfer(
        pointer_table,
        [obj.data_ptr() for obj in objects],
        block_ids,
        device,
        native.TransferDirection.H2D,
        desc,
        chunk,
        native.EngineKVFormat.NL_X_TWO_X_NB_BS_NH_HS,
        1,
    )
    torch.npu.synchronize()

    for object_idx in range(num_objects):
        skipped_block = block_ids_cpu[object_idx * blocks_per_object]
        restored_block = block_ids_cpu[object_idx * blocks_per_object + 1]
        for layer, (key, value) in enumerate(paged):
            assert torch.count_nonzero(key[skipped_block]) == 0
            assert torch.count_nonzero(value[skipped_block]) == 0
            torch.testing.assert_close(
                key[restored_block],
                objects[object_idx][0, layer, bs:].reshape_as(key[0]),
            )
            torch.testing.assert_close(
                value[restored_block],
                objects[object_idx][1, layer, bs:].reshape_as(value[0]),
            )


def _kg0_shape_desc(*, nl: int, nb: int, bs: int) -> object:
    desc = native.PageBufferShapeDesc()
    desc.kv_size = 1
    desc.nl = nl
    desc.nb = nb
    desc.bs = bs
    desc.nh = 1
    desc.hs = 130
    desc.element_size = 1
    desc.block_stride_elems = 4160
    desc.dtype = torch.int8
    return desc


def _kg0_layers(
    *, nl: int, nb: int, bs: int, device: torch.device
) -> tuple[list[torch.Tensor], list[tuple[torch.Tensor, torch.Tensor]]]:
    pools: list[torch.Tensor] = []
    layers: list[tuple[torch.Tensor, torch.Tensor]] = []
    latent_w = 128
    for layer_i in range(nl):
        pool = torch.zeros(nb, 4160, dtype=torch.uint8, device=device)
        latent = pool[:, : bs * latent_w].view(nb, bs, latent_w)
        scale = pool[:, bs * latent_w :].view(torch.float16).view(nb, bs, 1)
        latent.copy_(
            (torch.arange(nb * bs * latent_w, device=device, dtype=torch.int32) % 251)
            .to(torch.uint8)
            .view(nb, bs, latent_w)
            + layer_i
        )
        scale.copy_(
            torch.arange(nb * bs, device=device, dtype=torch.float16).view(nb, bs, 1)
            + layer_i
        )
        pools.append(pool)
        layers.append((latent, scale))
    return pools, layers


@pytest.mark.skipif(not _npu_available(), reason="Ascend NPU required")
@pytest.mark.parametrize("direction_d2h", [True, False])
def test_kg0_packed_row_matches_torch_ops(direction_d2h: bool) -> None:
    """KG0: 128 B int8 latent + 2 B fp16 scale, packed 130 B LMC row."""
    torch_ops = pytest.importorskip("lmcache.v1.platform.torch_ops")

    device = torch.device("npu:0")
    nl, nb, bs = 2, 4, 32
    hidden_bytes, chunk = 130, 32
    fmt = native.EngineKVFormat.NL_X_TWO_X_NB_BS_HS
    desc = _kg0_shape_desc(nl=nl, nb=nb, bs=bs)

    pools_a, layers_a = _kg0_layers(nl=nl, nb=nb, bs=bs, device=device)
    pools_b, layers_b = _kg0_layers(nl=nl, nb=nb, bs=bs, device=device)
    ptrs = [
        p for latent, scale in layers_a for p in (latent.data_ptr(), scale.data_ptr())
    ]
    pointer_table = torch.tensor(ptrs, dtype=torch.int64, device=device)
    block_ids = torch.tensor([0], dtype=torch.int64, device=device)

    if direction_d2h:
        obj_a = torch.zeros((nl, chunk, hidden_bytes), dtype=torch.uint8, device=device)
        obj_b = torch.zeros((nl, chunk, hidden_bytes), dtype=torch.uint8, device=device)
        direction = native.TransferDirection.D2H
        torch_dir = native.TransferDirection.D2H
    else:
        src = torch.zeros((nl, chunk, hidden_bytes), dtype=torch.uint8, device=device)
        src[:, :, :128] = (
            torch.arange(nl * chunk * 128, dtype=torch.int32, device=device)
            .to(torch.uint8)
            .view(nl, chunk, 128)
        )
        src[:, :, 128:] = (
            torch.arange(nl * chunk, device=device, dtype=torch.float16)
            .view(nl, chunk, 1)
            .view(torch.uint8)
            .view(nl, chunk, 2)
        )
        obj_a = src.clone()
        obj_b = src.clone()
        for pool in pools_a + pools_b:
            pool.zero_()
        direction = native.TransferDirection.H2D
        torch_dir = native.TransferDirection.H2D

    lmc_ops.multi_layer_block_kv_transfer(
        pointer_table,
        [obj_a.data_ptr()],
        block_ids,
        device,
        direction,
        desc,
        chunk,
        fmt,
        0,
    )
    torch_ops.multi_layer_block_kv_transfer(
        layers_b,
        [obj_b],
        block_ids,
        device,
        torch_dir,
        desc,
        chunk,
        native.EngineKVFormat.NL_X_TWO_X_NB_BS_HS,
        0,
    )
    torch.npu.synchronize()
    if direction_d2h:
        assert torch.equal(obj_a.cpu(), obj_b.cpu())
        assert obj_a.shape[-1] == 130
    else:
        for pa, pb in zip(pools_a, pools_b, strict=True):
            assert torch.equal(pa.cpu(), pb.cpu())


def test_page_buffer_shape_desc_accepts_dtype() -> None:
    """MP register sets desc.dtype; C++ pybind must not reject the attribute."""
    assert lmc_ops.PageBufferShapeDesc is native.PageBufferShapeDesc
    desc = native.PageBufferShapeDesc()
    desc.dtype = torch.int8
    assert desc.dtype is torch.int8


def test_object_group_plan_is_bound() -> None:
    """CUDA plan op is exported; MP store/load uses it instead of per-KG Run()."""
    fn = getattr(lmc_ops, "execute_object_group_transfer", None)
    assert fn is not None
    assert callable(fn)
    assert hasattr(lmc_ops, "KernelGroupSpec")
    assert hasattr(lmc_ops, "BatchStep")
    assert hasattr(lmc_ops, "LaunchVar")
    assert hasattr(lmc_ops, "StagingCopy")


@pytest.mark.skipif(not _npu_available(), reason="Ascend NPU required")
def test_kg0_nested_layer_list_matches_pointer_table() -> None:
    """Direct API: nested list and int64 table of length 2*nl match."""
    device = torch.device("npu:0")
    nl, nb, bs, chunk = 2, 4, 32, 32
    desc = _kg0_shape_desc(nl=nl, nb=nb, bs=bs)
    _pools, layers = _kg0_layers(nl=nl, nb=nb, bs=bs, device=device)
    ptrs = [
        p for latent, scale in layers for p in (latent.data_ptr(), scale.data_ptr())
    ]
    pointer_table = torch.tensor(ptrs, dtype=torch.int64, device=device)
    block_ids = torch.tensor([0], dtype=torch.int64, device=device)
    obj_nested = torch.zeros((nl, chunk, 130), dtype=torch.uint8, device=device)
    obj_ptrs = torch.zeros((nl, chunk, 130), dtype=torch.uint8, device=device)
    fmt = native.EngineKVFormat.NL_X_TWO_X_NB_BS_HS
    kwargs = dict(
        block_ids=block_ids,
        device=device,
        direction=native.TransferDirection.D2H,
        shape_desc=desc,
        lmcache_chunk_size=chunk,
        engine_kv_format=fmt,
        skip_prefix_n_blocks=0,
    )
    lmc_ops.multi_layer_block_kv_transfer(layers, [obj_nested.data_ptr()], **kwargs)
    lmc_ops.multi_layer_block_kv_transfer(
        pointer_table, [obj_ptrs.data_ptr()], **kwargs
    )
    torch.npu.synchronize()
    assert torch.equal(obj_nested.cpu(), obj_ptrs.cpu())
    assert obj_nested.shape[-1] == 130


@pytest.mark.skipif(not _npu_available(), reason="Ascend NPU required")
def test_kg0_live_chunk_does_not_abort() -> None:
    """DSv4 KG0 store geometry: 21 layers, 32 blocks, 1024-slot packed row."""
    device = torch.device("npu:0")
    nl, nb, bs, n_blocks = 21, 32, 32, 32
    chunk = n_blocks * bs
    desc = _kg0_shape_desc(nl=nl, nb=nb, bs=bs)
    _pools, layers = _kg0_layers(nl=nl, nb=nb, bs=bs, device=device)
    obj = torch.zeros((nl, chunk, 130), dtype=torch.uint8, device=device)
    block_ids = torch.arange(n_blocks, dtype=torch.int64, device=device)
    lmc_ops.multi_layer_block_kv_transfer(
        layers,
        [obj.data_ptr()],
        block_ids,
        device,
        native.TransferDirection.D2H,
        desc,
        chunk,
        native.EngineKVFormat.NL_X_TWO_X_NB_BS_HS,
        0,
    )
    torch.npu.synchronize()
    assert obj.shape == (nl, chunk, 130)
    assert int(obj.sum().cpu()) > 0


@pytest.mark.skipif(not _npu_available(), reason="Ascend NPU required")
def test_kg0_d2h_into_aclrt_pinned_host_does_not_abort() -> None:
    """Live MP store writes packed 130 B rows into L1 (aclrtMallocHost)."""
    device = torch.device("npu:0")
    nl, nb, bs, n_blocks = 21, 32, 32, 32
    chunk = n_blocks * bs
    desc = _kg0_shape_desc(nl=nl, nb=nb, bs=bs)
    pools, layers = _kg0_layers(nl=nl, nb=nb, bs=bs, device=device)
    nbytes = nl * chunk * 130
    host_ptr = int(lmc_ops.alloc_pinned_ptr(nbytes, 0))
    block_ids = torch.arange(n_blocks, dtype=torch.int64, device=device)
    fmt = native.EngineKVFormat.NL_X_TWO_X_NB_BS_HS
    try:
        lmc_ops.multi_layer_block_kv_transfer(
            layers,
            [host_ptr],
            block_ids,
            device,
            native.TransferDirection.D2H,
            desc,
            chunk,
            fmt,
            0,
        )
        torch.npu.synchronize()
        for pool in pools:
            pool.zero_()
        lmc_ops.multi_layer_block_kv_transfer(
            layers,
            [host_ptr],
            block_ids,
            device,
            native.TransferDirection.H2D,
            desc,
            chunk,
            fmt,
            0,
        )
        torch.npu.synchronize()
        assert int(torch.stack(pools).sum().cpu()) > 0
    finally:
        lmc_ops.free_pinned_ptr(host_ptr)


@pytest.mark.skipif(not _npu_available(), reason="Ascend NPU required")
def test_kg0_d2h_into_cpu_shm_like_host_does_not_abort() -> None:
    """Live L1 is MixedMemoryAllocator SHM, not aclrtMallocHost."""
    device = torch.device("npu:0")
    nl, nb, bs, n_blocks = 21, 32, 32, 32
    chunk = n_blocks * bs
    desc = _kg0_shape_desc(nl=nl, nb=nb, bs=bs)
    pools, layers = _kg0_layers(nl=nl, nb=nb, bs=bs, device=device)
    obj = torch.zeros((nl, chunk, 130), dtype=torch.uint8)
    block_ids = torch.arange(n_blocks, dtype=torch.int64, device=device)
    fmt = native.EngineKVFormat.NL_X_TWO_X_NB_BS_HS
    lmc_ops.multi_layer_block_kv_transfer(
        layers,
        [obj.data_ptr()],
        block_ids,
        device,
        native.TransferDirection.D2H,
        desc,
        chunk,
        fmt,
        0,
    )
    torch.npu.synchronize()
    assert int(obj.sum()) > 0
    for pool in pools:
        pool.zero_()
    lmc_ops.multi_layer_block_kv_transfer(
        layers,
        [obj.data_ptr()],
        block_ids,
        device,
        native.TransferDirection.H2D,
        desc,
        chunk,
        fmt,
        0,
    )
    torch.npu.synchronize()
    assert int(torch.stack(pools).sum().cpu()) > 0


@pytest.mark.skipif(not _npu_available(), reason="Ascend NPU required")
def test_kg0_native_desc_from_affinity_pool_does_not_abort() -> None:
    """Live MP: native PageBufferShapeDesc + enums on AffinityThreadPool worker."""
    from lmcache.v1.multiprocess.affinity_pool import AffinityThreadPool

    device = torch.device("npu:0")
    nl, nb, bs = 2, 4, 32
    chunk = bs
    desc = _kg0_shape_desc(nl=nl, nb=nb, bs=bs)
    _pools, layers = _kg0_layers(nl=nl, nb=nb, bs=bs, device=device)
    obj = torch.zeros((nl, chunk, 130), dtype=torch.uint8)
    block_ids = torch.tensor([0], dtype=torch.int64, device=device)

    def _store() -> None:
        lmc_ops.multi_layer_block_kv_transfer(
            layers,
            [obj.data_ptr()],
            block_ids,
            device,
            native.TransferDirection.D2H,
            desc,
            chunk,
            native.EngineKVFormat.NL_X_TWO_X_NB_BS_HS,
            0,
        )
        torch.npu.synchronize()

    pool = AffinityThreadPool(max_workers=1, thread_name_prefix="lmcache")
    try:
        pool.submit(_store, affinity_key=0).result(timeout=60)
    finally:
        pool.shutdown(wait=True)
    assert int(obj.sum()) > 0


@pytest.mark.skipif(not _npu_available(), reason="Ascend NPU required")
def test_torch_check_raises_python_exception() -> None:
    """Host TORCH_CHECK must not abort the process (GIL held)."""
    device = torch.device("npu:0")
    desc = _kg0_shape_desc(nl=2, nb=4, bs=32)
    obj = torch.zeros((2, 32, 130), dtype=torch.uint8, device=device)
    block_ids = torch.tensor([0], dtype=torch.int64, device=device)
    bad_ptrs = torch.tensor([obj.data_ptr()], dtype=torch.int64, device=device)
    with pytest.raises(RuntimeError, match="paged_buffer_ptrs_tensor"):
        lmc_ops.multi_layer_block_kv_transfer(
            bad_ptrs,
            [obj.data_ptr()],
            block_ids,
            device,
            native.TransferDirection.D2H,
            desc,
            32,
            native.EngineKVFormat.NL_X_TWO_X_NB_BS_HS,
            0,
        )


def _nh_cs_shape_desc(
    *,
    nl: int,
    nb: int,
    bs: int,
    hs: int,
    dtype: torch.dtype,
    block_stride_elems: int = 0,
) -> object:
    desc = native.PageBufferShapeDesc()
    desc.kv_size = 1
    desc.nl = nl
    desc.nb = nb
    desc.bs = bs
    desc.nh = 1
    desc.hs = hs
    desc.element_size = torch.empty((), dtype=dtype).element_size()
    desc.block_stride_elems = block_stride_elems
    desc.dtype = dtype
    return desc


def _nh_cs_layers(
    *,
    nl: int,
    nb: int,
    bs: int,
    hs: int,
    dtype: torch.dtype,
    device: torch.device,
    offset: int = 0,
) -> list[torch.Tensor]:
    layers: list[torch.Tensor] = []
    for layer_i in range(nl):
        values = torch.arange(nb * bs * hs, dtype=torch.float32, device=device).view(
            nb, bs, 1, hs
        )
        tensor = (values + offset + 1000 * layer_i).to(dtype)
        layers.append(tensor)
    return layers


@pytest.mark.skipif(not _npu_available(), reason="Ascend NPU required")
@pytest.mark.parametrize("direction_d2h", [True, False])
def test_nh_cs_tight_bf16_matches_torch_ops(direction_d2h: bool) -> None:
    """Fused NH_CS: tight bf16 [NB, BS, 1, 512] vs torch_ops."""
    torch_ops = pytest.importorskip("lmcache.v1.platform.torch_ops")

    device = torch.device("npu:0")
    nl, nb, bs, hs = 2, 4, 4, 512
    chunk = bs
    dtype = torch.bfloat16
    fmt = native.EngineKVFormat.NL_X_NB_BS_NH_CS
    desc = _nh_cs_shape_desc(nl=nl, nb=nb, bs=bs, hs=hs, dtype=dtype)
    layers_a = _nh_cs_layers(nl=nl, nb=nb, bs=bs, hs=hs, dtype=dtype, device=device)
    layers_b = _nh_cs_layers(nl=nl, nb=nb, bs=bs, hs=hs, dtype=dtype, device=device)
    pointer_table = torch.tensor(
        [int(t.data_ptr()) for t in layers_a], dtype=torch.int64, device=device
    )
    block_ids = torch.tensor([1], dtype=torch.int64, device=device)

    if direction_d2h:
        obj_a = torch.zeros((nl, chunk, hs), dtype=dtype, device=device)
        obj_b = torch.zeros((nl, chunk, hs), dtype=dtype, device=device)
        direction = native.TransferDirection.D2H
    else:
        src = (
            torch.arange(nl * chunk * hs, dtype=torch.float32, device=device)
            .view(nl, chunk, hs)
            .to(dtype)
        )
        obj_a = src.clone()
        obj_b = src.clone()
        for layer in layers_a + layers_b:
            layer.zero_()
        direction = native.TransferDirection.H2D

    lmc_ops.multi_layer_block_kv_transfer(
        pointer_table,
        [obj_a.data_ptr()],
        block_ids,
        device,
        direction,
        desc,
        chunk,
        fmt,
        0,
    )
    torch_ops.multi_layer_block_kv_transfer(
        layers_b,
        [obj_b],
        block_ids,
        device,
        direction,
        desc,
        chunk,
        fmt,
        0,
    )
    torch.npu.synchronize()
    if direction_d2h:
        assert torch.equal(obj_a.cpu(), obj_b.cpu())
    else:
        for a, b in zip(layers_a, layers_b, strict=True):
            assert torch.equal(a.cpu(), b.cpu())


@pytest.mark.skipif(not _npu_available(), reason="Ascend NPU required")
def test_nh_cs_padded_fp32_matches_torch_ops() -> None:
    """Dim-0 padded fp32: stride 8192 with bs=2, hs=2048 (tight=4096)."""
    torch_ops = pytest.importorskip("lmcache.v1.platform.torch_ops")

    device = torch.device("npu:0")
    nl, nb, bs, hs = 2, 4, 2, 2048
    stride, chunk = 8192, bs
    dtype = torch.float32
    fmt = native.EngineKVFormat.NL_X_NB_BS_NH_CS
    desc = _nh_cs_shape_desc(
        nl=nl, nb=nb, bs=bs, hs=hs, dtype=dtype, block_stride_elems=stride
    )
    layers_a: list[torch.Tensor] = []
    layers_b: list[torch.Tensor] = []
    pools_a: list[torch.Tensor] = []
    for layer_i in range(nl):
        pool_a = torch.full((nb, stride), 999.0, dtype=dtype, device=device)
        pool_b = pool_a.clone()
        view_a = pool_a[:, : bs * hs].view(nb, bs, 1, hs)
        view_b = pool_b[:, : bs * hs].view(nb, bs, 1, hs)
        values = (
            torch.arange(nb * bs * hs, dtype=dtype, device=device).view(nb, bs, 1, hs)
            + 1000 * layer_i
        )
        view_a.copy_(values)
        view_b.copy_(values)
        pools_a.append(pool_a)
        layers_a.append(view_a)
        layers_b.append(view_b)

    pointer_table = torch.tensor(
        [int(t.data_ptr()) for t in layers_a], dtype=torch.int64, device=device
    )
    block_ids = torch.tensor([1], dtype=torch.int64, device=device)
    obj_a = torch.zeros((nl, chunk, hs), dtype=dtype, device=device)
    obj_b = torch.zeros((nl, chunk, hs), dtype=dtype, device=device)
    lmc_ops.multi_layer_block_kv_transfer(
        pointer_table,
        [obj_a.data_ptr()],
        block_ids,
        device,
        native.TransferDirection.D2H,
        desc,
        chunk,
        fmt,
        0,
    )
    torch_ops.multi_layer_block_kv_transfer(
        layers_b,
        [obj_b],
        block_ids,
        device,
        native.TransferDirection.D2H,
        desc,
        chunk,
        fmt,
        0,
    )
    torch.npu.synchronize()
    assert torch.equal(obj_a.cpu(), obj_b.cpu())
    assert not torch.any(obj_a.cpu() == 999.0)
    for pool in pools_a:
        assert torch.all(pool[:, bs * hs :] == 999.0)

    for view in layers_a + layers_b:
        view.zero_()
    lmc_ops.multi_layer_block_kv_transfer(
        pointer_table,
        [obj_a.data_ptr()],
        block_ids,
        device,
        native.TransferDirection.H2D,
        desc,
        chunk,
        fmt,
        0,
    )
    torch_ops.multi_layer_block_kv_transfer(
        layers_b,
        [obj_b],
        block_ids,
        device,
        native.TransferDirection.H2D,
        desc,
        chunk,
        fmt,
        0,
    )
    torch.npu.synchronize()
    for a, b in zip(layers_a, layers_b, strict=True):
        assert torch.equal(a.cpu(), b.cpu())
    for pool in pools_a:
        assert torch.all(pool[:, bs * hs :] == 999.0)


@pytest.mark.skipif(not _npu_available(), reason="Ascend NPU required")
def test_nh_cs_nested_layer_list_matches_pointer_table() -> None:
    """Direct API: nested list and int64 table of length nl match."""
    device = torch.device("npu:0")
    nl, nb, bs, hs = 2, 4, 4, 512
    chunk = bs
    dtype = torch.bfloat16
    fmt = native.EngineKVFormat.NL_X_NB_BS_NH_CS
    desc = _nh_cs_shape_desc(nl=nl, nb=nb, bs=bs, hs=hs, dtype=dtype)
    layers = _nh_cs_layers(nl=nl, nb=nb, bs=bs, hs=hs, dtype=dtype, device=device)
    pointer_table = torch.tensor(
        [int(t.data_ptr()) for t in layers], dtype=torch.int64, device=device
    )
    assert pointer_table.numel() == nl
    block_ids = torch.tensor([0], dtype=torch.int64, device=device)
    obj_nested = torch.zeros((nl, chunk, hs), dtype=dtype, device=device)
    obj_ptrs = torch.zeros((nl, chunk, hs), dtype=dtype, device=device)
    kwargs = dict(
        block_ids=block_ids,
        device=device,
        direction=native.TransferDirection.D2H,
        shape_desc=desc,
        lmcache_chunk_size=chunk,
        engine_kv_format=fmt,
        skip_prefix_n_blocks=0,
    )
    lmc_ops.multi_layer_block_kv_transfer(layers, [obj_nested.data_ptr()], **kwargs)
    lmc_ops.multi_layer_block_kv_transfer(
        pointer_table, [obj_ptrs.data_ptr()], **kwargs
    )
    torch.npu.synchronize()
    assert torch.equal(obj_nested.cpu(), obj_ptrs.cpu())


@pytest.mark.skipif(not _npu_available(), reason="Ascend NPU required")
@pytest.mark.parametrize("direction_d2h", [True, False])
def test_object_group_plan_two_groups_matches_direct_launches(
    direction_d2h: bool,
) -> None:
    """One execute_object_group_transfer equals two direct kernel launches."""
    device = torch.device("npu:0")
    nl, nb, bs = 2, 4, 32
    chunk = bs
    desc17 = _kg0_shape_desc(nl=nl, nb=nb, bs=bs)
    _pools_d, layers17_d = _kg0_layers(nl=nl, nb=nb, bs=bs, device=device)
    _pools_p, layers17_p = _kg0_layers(nl=nl, nb=nb, bs=bs, device=device)
    table17_d = torch.tensor(
        [
            p
            for latent, scale in layers17_d
            for p in (latent.data_ptr(), scale.data_ptr())
        ],
        dtype=torch.int64,
        device=device,
    )
    table17_p = torch.tensor(
        [
            p
            for latent, scale in layers17_p
            for p in (latent.data_ptr(), scale.data_ptr())
        ],
        dtype=torch.int64,
        device=device,
    )
    obj17_d = torch.zeros((nl, chunk, 130), dtype=torch.uint8, device=device)
    obj17_p = torch.zeros((nl, chunk, 130), dtype=torch.uint8, device=device)

    hs, dtype = 512, torch.bfloat16
    desc13 = _nh_cs_shape_desc(nl=nl, nb=nb, bs=bs, hs=hs, dtype=dtype)
    layers13_d = _nh_cs_layers(nl=nl, nb=nb, bs=bs, hs=hs, dtype=dtype, device=device)
    layers13_p = _nh_cs_layers(nl=nl, nb=nb, bs=bs, hs=hs, dtype=dtype, device=device)
    table13_d = torch.tensor(
        [int(t.data_ptr()) for t in layers13_d], dtype=torch.int64, device=device
    )
    table13_p = torch.tensor(
        [int(t.data_ptr()) for t in layers13_p], dtype=torch.int64, device=device
    )
    obj13_d = torch.zeros((nl, chunk, hs), dtype=dtype, device=device)
    obj13_p = torch.zeros((nl, chunk, hs), dtype=dtype, device=device)

    block_ids = torch.tensor([0], dtype=torch.int64, device=device)
    fmt17 = native.EngineKVFormat.NL_X_TWO_X_NB_BS_HS
    fmt13 = native.EngineKVFormat.NL_X_NB_BS_NH_CS
    direction = (
        native.TransferDirection.D2H if direction_d2h else native.TransferDirection.H2D
    )

    if not direction_d2h:
        obj17_d[:, :, :128] = (
            torch.arange(nl * chunk * 128, dtype=torch.int32, device=device)
            .to(torch.uint8)
            .view(nl, chunk, 128)
        )
        obj17_d[:, :, 128:] = (
            torch.arange(nl * chunk, device=device, dtype=torch.float16)
            .view(nl, chunk, 1)
            .view(torch.uint8)
        )
        obj17_p.copy_(obj17_d)
        src13 = (
            torch.arange(nl * chunk * hs, dtype=torch.float32, device=device)
            .view(nl, chunk, hs)
            .to(dtype)
        )
        obj13_d.copy_(src13)
        obj13_p.copy_(src13)
        for latent, scale in layers17_d + layers17_p:
            latent.zero_()
            scale.zero_()
        for layer in layers13_d + layers13_p:
            layer.zero_()

    kwargs17 = dict(
        block_ids=block_ids,
        device=device,
        direction=direction,
        shape_desc=desc17,
        lmcache_chunk_size=chunk,
        engine_kv_format=fmt17,
        skip_prefix_n_blocks=0,
    )
    kwargs13 = dict(
        block_ids=block_ids,
        device=device,
        direction=direction,
        shape_desc=desc13,
        lmcache_chunk_size=chunk,
        engine_kv_format=fmt13,
        skip_prefix_n_blocks=0,
    )
    lmc_ops.multi_layer_block_kv_transfer(table17_d, [obj17_d.data_ptr()], **kwargs17)
    lmc_ops.multi_layer_block_kv_transfer(table13_d, [obj13_d.data_ptr()], **kwargs13)
    spec17 = lmc_ops.KernelGroupSpec(
        table17_p.data_ptr(),
        [obj17_p.data_ptr()],
        desc17,
        chunk,
        int(fmt17),
        block_ids.data_ptr(),
        block_ids.numel(),
    )
    spec13 = lmc_ops.KernelGroupSpec(
        table13_p.data_ptr(),
        [obj13_p.data_ptr()],
        desc13,
        chunk,
        int(fmt13),
        block_ids.data_ptr(),
        block_ids.numel(),
    )
    step = lmc_ops.BatchStep(
        [],
        [
            lmc_ops.LaunchVar(0, 0, 1, 1, 0),
            lmc_ops.LaunchVar(1, 0, 1, 1, 0),
        ],
    )
    lmc_ops.execute_object_group_transfer(
        int(direction),
        device,
        1 << 26,
        [spec17, spec13],
        [step],
    )
    torch.npu.synchronize()
    if direction_d2h:
        assert torch.equal(obj17_d.cpu(), obj17_p.cpu())
        assert torch.equal(obj13_d.cpu(), obj13_p.cpu())
        return
    for (lat_a, sc_a), (lat_b, sc_b) in zip(layers17_d, layers17_p, strict=True):
        assert torch.equal(lat_a.cpu(), lat_b.cpu())
        assert torch.equal(sc_a.cpu(), sc_b.cpu())
    for a, b in zip(layers13_d, layers13_p, strict=True):
        assert torch.equal(a.cpu(), b.cpu())


@pytest.mark.skipif(not _npu_available(), reason="Ascend NPU required")
def test_lmcache_memcpy_async_cpu_host_round_trip() -> None:
    """Plan-path staging: C++ aclrtMemcpyAsync of pageable/SHM-like host."""
    device = torch.device("npu:0")
    nbytes = 64 * 1024
    host = torch.arange(nbytes, dtype=torch.uint8)
    dev = torch.zeros(nbytes, dtype=torch.uint8, device=device)
    lmc_ops.lmcache_memcpy_async(
        int(dev.data_ptr()),
        int(host.data_ptr()),
        nbytes,
        lmc_ops.TransferDirection.H2D,
        0,
        4096,
    )
    torch.npu.synchronize()
    assert torch.equal(dev.cpu(), host)
    host_back = torch.zeros(nbytes, dtype=torch.uint8)
    lmc_ops.lmcache_memcpy_async(
        int(host_back.data_ptr()),
        int(dev.data_ptr()),
        nbytes,
        lmc_ops.TransferDirection.D2H,
        0,
        4096,
    )
    torch.npu.synchronize()
    assert torch.equal(host_back, host)


@pytest.mark.skipif(not _npu_available(), reason="Ascend NPU required")
def test_object_group_plan_staging_from_affinity_pool() -> None:
    """Live store: StagingCopy + execute on AffinityThreadPool (not main)."""
    from lmcache.v1.multiprocess.affinity_pool import AffinityThreadPool

    device = torch.device("npu:0")
    nl, nb, bs = 2, 4, 32
    chunk = bs
    desc = _kg0_shape_desc(nl=nl, nb=nb, bs=bs)
    _pools, layers = _kg0_layers(nl=nl, nb=nb, bs=bs, device=device)
    table = torch.tensor(
        [p for latent, scale in layers for p in (latent.data_ptr(), scale.data_ptr())],
        dtype=torch.int64,
        device=device,
    )
    temp = torch.zeros((nl, chunk, 130), dtype=torch.uint8, device=device)
    host = torch.zeros((nl, chunk, 130), dtype=torch.uint8)
    block_ids = torch.tensor([0], dtype=torch.int64, device=device)
    fmt = native.EngineKVFormat.NL_X_TWO_X_NB_BS_HS
    spec = lmc_ops.KernelGroupSpec(
        table.data_ptr(),
        [temp.data_ptr()],
        desc,
        chunk,
        int(fmt),
        block_ids.data_ptr(),
        block_ids.numel(),
    )
    staging = [
        lmc_ops.StagingCopy(int(host.data_ptr()), int(temp.data_ptr()), host.nbytes, 0)
    ]
    step = lmc_ops.BatchStep(staging, [lmc_ops.LaunchVar(0, 0, 1, 1, 0)])

    def _store() -> None:
        lmc_ops.execute_object_group_transfer(
            int(native.TransferDirection.D2H),
            device,
            1 << 26,
            [spec],
            [step],
        )
        torch.npu.synchronize()

    pool = AffinityThreadPool(max_workers=1, thread_name_prefix="lmcache")
    try:
        pool.submit(_store, affinity_key=0).result(timeout=60)
    finally:
        pool.shutdown(wait=True)
    assert int(host.sum()) > 0
