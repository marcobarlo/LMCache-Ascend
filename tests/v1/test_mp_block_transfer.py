# SPDX-License-Identifier: Apache-2.0
"""Direct correctness tests for the block-level MP-mode AscendC kernel.

Bypasses the MP server. Each generic test asserts an observable contract;
layouts, hosts, directions, and thread contexts are parametrizations of that
contract — not separate smoke tests.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from contextlib import ExitStack, contextmanager
from typing import Any

import pytest
import torch

import lmcache.lmcache_native as native
import lmcache_ascend.c_ops as lmc_ops

requires_npu = pytest.mark.skipif(
    not (hasattr(torch, "npu") and torch.npu.is_available()),
    reason="Ascend NPU required",
)

KG0_FMT = native.EngineKVFormat.NL_X_TWO_X_NB_BS_HS
NH_CS_FMT = native.EngineKVFormat.NL_X_NB_BS_NH_CS
SEP_KV_FMT = native.EngineKVFormat.NL_X_TWO_X_NB_BS_NH_HS
D2H = native.TransferDirection.D2H
H2D = native.TransferDirection.H2D


def _shape_desc(
    *,
    kv_size: int,
    nl: int,
    nb: int,
    bs: int,
    nh: int,
    hs: int,
    dtype: torch.dtype,
    block_stride_elems: int = 0,
) -> object:
    desc = native.PageBufferShapeDesc()
    desc.kv_size = kv_size
    desc.nl = nl
    desc.nb = nb
    desc.bs = bs
    desc.nh = nh
    desc.hs = hs
    desc.element_size = torch.empty((), dtype=dtype).element_size()
    desc.block_stride_elems = block_stride_elems
    desc.dtype = dtype
    return desc


def _kg0_desc(*, nl: int, nb: int, bs: int) -> object:
    return _shape_desc(
        kv_size=1,
        nl=nl,
        nb=nb,
        bs=bs,
        nh=1,
        hs=130,
        dtype=torch.int8,
        block_stride_elems=4160,
    )


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


def _kg0_packed_object(
    nl: int, chunk: int, device: torch.device | None
) -> torch.Tensor:
    obj = torch.zeros((nl, chunk, 130), dtype=torch.uint8, device=device)
    obj[:, :, :128] = (
        torch.arange(nl * chunk * 128, dtype=torch.int32, device=device)
        .to(torch.uint8)
        .view(nl, chunk, 128)
    )
    obj[:, :, 128:] = (
        torch.arange(nl * chunk, device=device, dtype=torch.float16)
        .view(nl, chunk, 1)
        .view(torch.uint8)
    )
    return obj


def _nh_cs_layers(
    *,
    nl: int,
    nb: int,
    bs: int,
    hs: int,
    dtype: torch.dtype,
    device: torch.device,
) -> list[torch.Tensor]:
    layers: list[torch.Tensor] = []
    for layer_i in range(nl):
        values = torch.arange(nb * bs * hs, dtype=torch.float32, device=device).view(
            nb, bs, 1, hs
        )
        layers.append((values + 1000 * layer_i).to(dtype))
    return layers


def _nh_cs_padded(
    *, nl: int, nb: int, bs: int, hs: int, stride: int, device: torch.device
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    dtype = torch.float32
    pools: list[torch.Tensor] = []
    layers: list[torch.Tensor] = []
    for layer_i in range(nl):
        pool = torch.full((nb, stride), 999.0, dtype=dtype, device=device)
        view = pool[:, : bs * hs].view(nb, bs, 1, hs)
        view.copy_(
            torch.arange(nb * bs * hs, dtype=dtype, device=device).view(nb, bs, 1, hs)
            + 1000 * layer_i
        )
        pools.append(pool)
        layers.append(view)
    return pools, layers


def _pointer_table(layers: Sequence[Any], device: torch.device) -> torch.Tensor:
    ptrs: list[int] = []
    for item in layers:
        if isinstance(item, tuple):
            ptrs.extend(int(t.data_ptr()) for t in item)
        else:
            ptrs.append(int(item.data_ptr()))
    return torch.tensor(ptrs, dtype=torch.int64, device=device)


def _zero_engine(layers: Sequence[Any]) -> None:
    for item in layers:
        if isinstance(item, tuple):
            for tensor in item:
                tensor.zero_()
        else:
            item.zero_()


def _assert_engine_equal(left: Sequence[Any], right: Sequence[Any]) -> None:
    for a, b in zip(left, right, strict=True):
        if isinstance(a, tuple):
            for ta, tb in zip(a, b, strict=True):
                assert torch.equal(ta.cpu(), tb.cpu())
        else:
            assert torch.equal(a.cpu(), b.cpu())


def _transfer(
    paged: object,
    obj_ptrs: list[int],
    block_ids: torch.Tensor,
    device: torch.device,
    direction: object,
    desc: object,
    chunk: int,
    fmt: object,
    skip: int = 0,
) -> None:
    lmc_ops.multi_layer_block_kv_transfer(
        paged, obj_ptrs, block_ids, device, direction, desc, chunk, fmt, skip
    )
    torch.npu.synchronize()


def _run(fn: Callable[[], None], *, affinity: bool) -> None:
    if not affinity:
        fn()
        return
    from lmcache.v1.multiprocess.affinity_pool import AffinityThreadPool

    pool = AffinityThreadPool(max_workers=1, thread_name_prefix="lmcache")
    try:
        pool.submit(fn, affinity_key=0).result(timeout=120)
    finally:
        pool.shutdown(wait=True)


@contextmanager
def _host_object(
    kind: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> Iterator[tuple[list[int], torch.Tensor | None]]:
    nbytes = int(torch.empty(shape, dtype=dtype).nbytes)
    if kind == "npu":
        obj = torch.zeros(shape, dtype=dtype, device=device)
        yield [int(obj.data_ptr())], obj
        return
    if kind in ("cpu", "affinity"):
        obj = torch.zeros(shape, dtype=dtype)
        yield [int(obj.data_ptr())], obj
        return
    if kind == "pinned":
        ptr = int(lmc_ops.alloc_pinned_ptr(nbytes, 0))
        try:
            yield [ptr], None
        finally:
            lmc_ops.free_pinned_ptr(ptr)
        return
    raise ValueError(kind)


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
    if padded:
        storage = torch.empty((nb, bs + 1, nh, hs), dtype=torch.float16, device=device)
        tensor = storage[:, :bs]
    else:
        tensor = torch.empty((nb, bs, nh, hs), dtype=torch.float16, device=device)
    values = torch.arange(tensor.numel(), dtype=torch.float32, device=device).reshape_as(
        tensor
    )
    tensor.copy_((values + offset).to(torch.float16))
    return tensor


def _clone_engine(layers: Sequence[Any]) -> list[Any]:
    cloned: list[Any] = []
    for item in layers:
        if isinstance(item, tuple):
            cloned.append(tuple(t.clone() for t in item))
        else:
            cloned.append(item.clone())
    return cloned


def _engine_nb(layers: Sequence[Any]) -> int:
    first = layers[0][0] if isinstance(layers[0], tuple) else layers[0]
    return int(first.shape[0])


def _transferred_block_ids(ids: list[int], num_objects: int, skip: int) -> set[int]:
    bpo = len(ids) // num_objects
    return {
        ids[obj_i * bpo + local]
        for obj_i in range(num_objects)
        for local in range(skip, bpo)
    }


def _assert_block_zero(layers: Sequence[Any], bid: int) -> None:
    for item in layers:
        for tensor in item if isinstance(item, tuple) else (item,):
            assert torch.count_nonzero(tensor[bid]) == 0


def _assert_block_equal(left: Sequence[Any], right: Sequence[Any], bid: int) -> None:
    for a, b in zip(left, right, strict=True):
        planes_a = a if isinstance(a, tuple) else (a,)
        planes_b = b if isinstance(b, tuple) else (b,)
        for ta, tb in zip(planes_a, planes_b, strict=True):
            if ta.dtype in (torch.float16, torch.bfloat16):
                torch.testing.assert_close(ta[bid], tb[bid])
            else:
                assert torch.equal(ta[bid].cpu(), tb[bid].cpu())


def _kg0_packed_block(
    latent: torch.Tensor, scale: torch.Tensor, bid: int, bs: int
) -> torch.Tensor:
    return torch.cat(
        [
            latent[bid].reshape(bs, 128),
            scale[bid].contiguous().view(torch.uint8).reshape(bs, 2),
        ],
        dim=-1,
    )


def _assert_d2h_object(
    objects: Sequence[torch.Tensor],
    layers: Sequence[Any],
    ids: list[int],
    *,
    skip: int,
    bs: int,
    num_objects: int,
) -> None:
    bpo = len(ids) // num_objects
    for obj_i, obj in enumerate(objects):
        for local in range(bpo):
            sl = slice(local * bs, (local + 1) * bs)
            bid = ids[obj_i * bpo + local]
            if local < skip:
                prefix = obj[:, :, sl] if obj.dim() == 4 else obj[:, sl]
                assert torch.count_nonzero(prefix) == 0
                continue
            if obj.dim() == 4:
                for layer, (key, value) in enumerate(layers):
                    torch.testing.assert_close(
                        obj[0, layer, sl].reshape_as(key[bid]), key[bid]
                    )
                    torch.testing.assert_close(
                        obj[1, layer, sl].reshape_as(value[bid]), value[bid]
                    )
            else:
                for layer, (latent, scale) in enumerate(layers):
                    assert torch.equal(
                        obj[layer, sl].cpu(),
                        _kg0_packed_block(latent, scale, bid, bs).cpu(),
                    )


def _assert_h2d_engine(
    layers: Sequence[Any],
    golden: Sequence[Any],
    ids: list[int],
    *,
    skip: int,
    num_objects: int,
) -> None:
    transferred = _transferred_block_ids(ids, num_objects, skip)
    for bid in range(_engine_nb(layers)):
        if bid in transferred:
            _assert_block_equal(layers, golden, bid)
        else:
            _assert_block_zero(layers, bid)


@contextmanager
def _host_objects(
    kind: str,
    shapes: Sequence[tuple[int, ...]],
    dtype: torch.dtype,
    device: torch.device,
) -> Iterator[tuple[list[int], list[torch.Tensor | None]]]:
    with ExitStack() as stack:
        allocated = [
            stack.enter_context(_host_object(kind, shape, dtype, device))
            for shape in shapes
        ]
        ptrs: list[int] = []
        tensors: list[torch.Tensor | None] = []
        for part, tensor in allocated:
            ptrs.extend(part)
            tensors.append(tensor)
        yield ptrs, tensors


def _build_roundtrip_engine(
    layout: str,
    *,
    padded: bool,
    nl: int,
    device: torch.device,
) -> dict[str, Any]:
    if layout == "sep_kv":
        nb, bs, nh, hs = 8, 4, 2, 8
        hidden = nh * hs
        layers: list[Any] = []
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
            layers.append((key, value))
        stride = (bs + 1 if padded else bs) * hidden
        return dict(
            fmt=SEP_KV_FMT,
            desc=_shape_desc(
                kv_size=2,
                nl=nl,
                nb=nb,
                bs=bs,
                nh=nh,
                hs=hs,
                dtype=torch.float16,
                block_stride_elems=stride if padded else 0,
            ),
            layers=layers,
            table=_pointer_table(layers, device),
            obj_dtype=torch.float16,
            obj_tail=(nl, hidden),
            kv_leading=True,
            bs=bs,
        )
    nb, bs = 32, 32
    _pools, layers = _kg0_layers(nl=nl, nb=nb, bs=bs, device=device)
    return dict(
        fmt=KG0_FMT,
        desc=_kg0_desc(nl=nl, nb=nb, bs=bs),
        layers=layers,
        table=_pointer_table(layers, device),
        obj_dtype=torch.uint8,
        obj_tail=(nl, 130),
        kv_leading=False,
        bs=bs,
    )


# ---------------------------------------------------------------------------
# Contract: D2H then H2D restores unskipped selected blocks only.
# ---------------------------------------------------------------------------


def _roundtrip_cases() -> list[Any]:
    cases = [
        pytest.param(
            "sep_kv", "npu", False, 1, 2, 2, [1, 3, 4, 6], id="sep-kv-tight-skip"
        ),
        pytest.param(
            "sep_kv", "npu", True, 1, 2, 2, [1, 3, 4, 6], id="sep-kv-padded-skip"
        ),
        pytest.param("kg0", "npu", False, 1, 2, 1, [1, 3], id="kg0-npu-skip"),
    ]
    for host in ("npu", "cpu", "pinned", "affinity"):
        cases.append(
            pytest.param("kg0", host, False, 0, 2, 1, [0], id=f"kg0-{host}-mini")
        )
        cases.append(
            pytest.param(
                "kg0", host, False, 0, 21, 1, list(range(32)), id=f"kg0-{host}-live"
            )
        )
    return cases


@requires_npu
@pytest.mark.parametrize(
    "layout,host,padded,skip,nl,num_objects,block_ids",
    _roundtrip_cases(),
)
def test_roundtrip_restores_unskipped_selected_blocks(
    layout: str,
    host: str,
    padded: bool,
    skip: int,
    nl: int,
    num_objects: int,
    block_ids: list[int],
) -> None:
    device = torch.device("npu:0")
    spec = _build_roundtrip_engine(layout, padded=padded, nl=nl, device=device)
    bs = spec["bs"]
    bpo = len(block_ids) // num_objects
    chunk = bpo * bs
    if spec["kv_leading"]:
        obj_shape: tuple[int, ...] = (2, spec["obj_tail"][0], chunk, spec["obj_tail"][1])
    else:
        obj_shape = (spec["obj_tail"][0], chunk, spec["obj_tail"][1])
    ids = torch.tensor(block_ids, dtype=torch.int64, device=device)
    golden = _clone_engine(spec["layers"])
    with _host_objects(
        host, [obj_shape] * num_objects, spec["obj_dtype"], device
    ) as (ptrs, obj_tensors):

        def _d2h() -> None:
            _transfer(
                spec["table"],
                ptrs,
                ids,
                device,
                D2H,
                spec["desc"],
                chunk,
                spec["fmt"],
                skip=skip,
            )

        def _h2d() -> None:
            _transfer(
                spec["table"],
                ptrs,
                ids,
                device,
                H2D,
                spec["desc"],
                chunk,
                spec["fmt"],
                skip=skip,
            )

        _run(_d2h, affinity=host == "affinity")
        visible = [t for t in obj_tensors if t is not None]
        if len(visible) == num_objects:
            _assert_d2h_object(
                visible,
                spec["layers"],
                block_ids,
                skip=skip,
                bs=bs,
                num_objects=num_objects,
            )
        _zero_engine(spec["layers"])
        _run(_h2d, affinity=host == "affinity")
    _assert_h2d_engine(
        spec["layers"], golden, block_ids, skip=skip, num_objects=num_objects
    )


# ---------------------------------------------------------------------------
# Contract: native kernel matches torch_ops for each engine layout × direction.
# ---------------------------------------------------------------------------


def _layout_pair(layout: str, device: torch.device) -> dict[str, Any]:
    if layout == "kg0":
        nl, nb, bs, chunk = 2, 4, 32, 32
        pools_a, layers_a = _kg0_layers(nl=nl, nb=nb, bs=bs, device=device)
        _pools_b, layers_b = _kg0_layers(nl=nl, nb=nb, bs=bs, device=device)
        return dict(
            fmt=KG0_FMT,
            desc=_kg0_desc(nl=nl, nb=nb, bs=bs),
            layers_a=layers_a,
            layers_b=layers_b,
            table=_pointer_table(layers_a, device),
            obj_shape=(nl, chunk, 130),
            obj_dtype=torch.uint8,
            block_ids=torch.tensor([0], dtype=torch.int64, device=device),
            chunk=chunk,
            nl=nl,
            pools_a=pools_a,
            pad_width=None,
        )
    if layout == "nh_cs":
        nl, nb, bs, hs, chunk = 2, 4, 4, 512, 4
        dtype = torch.bfloat16
        layers_a = _nh_cs_layers(nl=nl, nb=nb, bs=bs, hs=hs, dtype=dtype, device=device)
        layers_b = _nh_cs_layers(nl=nl, nb=nb, bs=bs, hs=hs, dtype=dtype, device=device)
        return dict(
            fmt=NH_CS_FMT,
            desc=_shape_desc(
                kv_size=1, nl=nl, nb=nb, bs=bs, nh=1, hs=hs, dtype=dtype
            ),
            layers_a=layers_a,
            layers_b=layers_b,
            table=_pointer_table(layers_a, device),
            obj_shape=(nl, chunk, hs),
            obj_dtype=dtype,
            block_ids=torch.tensor([1], dtype=torch.int64, device=device),
            chunk=chunk,
            nl=nl,
            pools_a=None,
            pad_width=None,
        )
    if layout == "nh_cs_padded":
        nl, nb, bs, hs, stride, chunk = 2, 4, 2, 2048, 8192, 2
        dtype = torch.float32
        pools_a, layers_a = _nh_cs_padded(
            nl=nl, nb=nb, bs=bs, hs=hs, stride=stride, device=device
        )
        pools_b, layers_b = _nh_cs_padded(
            nl=nl, nb=nb, bs=bs, hs=hs, stride=stride, device=device
        )
        return dict(
            fmt=NH_CS_FMT,
            desc=_shape_desc(
                kv_size=1,
                nl=nl,
                nb=nb,
                bs=bs,
                nh=1,
                hs=hs,
                dtype=dtype,
                block_stride_elems=stride,
            ),
            layers_a=layers_a,
            layers_b=layers_b,
            table=_pointer_table(layers_a, device),
            obj_shape=(nl, chunk, hs),
            obj_dtype=dtype,
            block_ids=torch.tensor([1], dtype=torch.int64, device=device),
            chunk=chunk,
            nl=nl,
            pools_a=pools_a,
            pad_width=bs * hs,
        )
    raise ValueError(layout)


def _fill_src_object(layout: str, spec: dict[str, Any], device: torch.device) -> torch.Tensor:
    nl, chunk = spec["nl"], spec["chunk"]
    if layout == "kg0":
        return _kg0_packed_object(nl, chunk, device)
    src = torch.arange(
        nl * chunk * spec["obj_shape"][-1], dtype=torch.float32, device=device
    ).view(nl, chunk, spec["obj_shape"][-1])
    return src.to(spec["obj_dtype"])


@requires_npu
@pytest.mark.parametrize("layout", ["kg0", "nh_cs", "nh_cs_padded"])
@pytest.mark.parametrize("direction_d2h", [True, False], ids=["d2h", "h2d"])
def test_native_matches_torch_ops(layout: str, direction_d2h: bool) -> None:
    torch_ops = pytest.importorskip("lmcache.v1.platform.torch_ops")
    device = torch.device("npu:0")
    spec = _layout_pair(layout, device)
    if direction_d2h:
        obj_a = torch.zeros(spec["obj_shape"], dtype=spec["obj_dtype"], device=device)
        obj_b = obj_a.clone()
        direction = D2H
    else:
        src = _fill_src_object(layout, spec, device)
        obj_a, obj_b = src.clone(), src.clone()
        _zero_engine(spec["layers_a"])
        _zero_engine(spec["layers_b"])
        direction = H2D

    _transfer(
        spec["table"],
        [int(obj_a.data_ptr())],
        spec["block_ids"],
        device,
        direction,
        spec["desc"],
        spec["chunk"],
        spec["fmt"],
    )
    torch_ops.multi_layer_block_kv_transfer(
        spec["layers_b"],
        [obj_b],
        spec["block_ids"],
        device,
        direction,
        spec["desc"],
        spec["chunk"],
        spec["fmt"],
        0,
    )
    torch.npu.synchronize()
    if direction_d2h:
        assert torch.equal(obj_a.cpu(), obj_b.cpu())
        if spec["pad_width"] is not None:
            assert not torch.any(obj_a.cpu() == 999.0)
    else:
        _assert_engine_equal(spec["layers_a"], spec["layers_b"])
    if spec["pools_a"] is not None and spec["pad_width"] is not None:
        for pool in spec["pools_a"]:
            assert torch.all(pool[:, spec["pad_width"] :] == 999.0)


# ---------------------------------------------------------------------------
# Contract: nested layer list and int64 pointer table are equivalent APIs.
# ---------------------------------------------------------------------------


@requires_npu
@pytest.mark.parametrize("layout", ["kg0", "nh_cs"])
@pytest.mark.parametrize("direction_d2h", [True, False], ids=["d2h", "h2d"])
def test_nested_layers_match_pointer_table(layout: str, direction_d2h: bool) -> None:
    device = torch.device("npu:0")
    spec = _layout_pair(layout, device)
    table_b = _pointer_table(spec["layers_b"], device)
    if direction_d2h:
        obj_n = torch.zeros(spec["obj_shape"], dtype=spec["obj_dtype"], device=device)
        obj_p = obj_n.clone()
        direction = D2H
        paged_n, paged_p = spec["layers_a"], spec["table"]
        dest_n, dest_p = [int(obj_n.data_ptr())], [int(obj_p.data_ptr())]
    else:
        src = _fill_src_object(layout, spec, device)
        obj_n, obj_p = src.clone(), src.clone()
        _zero_engine(spec["layers_a"])
        _zero_engine(spec["layers_b"])
        direction = H2D
        paged_n, paged_p = spec["layers_a"], table_b
        dest_n, dest_p = [int(obj_n.data_ptr())], [int(obj_p.data_ptr())]

    kwargs = dict(
        block_ids=spec["block_ids"],
        device=device,
        direction=direction,
        desc=spec["desc"],
        chunk=spec["chunk"],
        fmt=spec["fmt"],
    )
    _transfer(paged_n, dest_n, **kwargs)
    _transfer(paged_p, dest_p, **kwargs)
    if direction_d2h:
        assert torch.equal(obj_n.cpu(), obj_p.cpu())
    else:
        _assert_engine_equal(spec["layers_a"], spec["layers_b"])


# ---------------------------------------------------------------------------
# Contract: invalid pointer-table length is a Python RuntimeError, not abort.
# ---------------------------------------------------------------------------


@requires_npu
def test_torch_check_raises_python_exception() -> None:
    device = torch.device("npu:0")
    desc = _kg0_desc(nl=2, nb=4, bs=32)
    obj = torch.zeros((2, 32, 130), dtype=torch.uint8, device=device)
    block_ids = torch.tensor([0], dtype=torch.int64, device=device)
    bad_ptrs = torch.tensor([obj.data_ptr()], dtype=torch.int64, device=device)
    with pytest.raises(RuntimeError, match="paged_buffer_ptrs_tensor"):
        lmc_ops.multi_layer_block_kv_transfer(
            bad_ptrs,
            [obj.data_ptr()],
            block_ids,
            device,
            D2H,
            desc,
            32,
            KG0_FMT,
            0,
        )


# ---------------------------------------------------------------------------
# Contract: object-group plan equals the corresponding direct launch(es).
# ---------------------------------------------------------------------------


@requires_npu
@pytest.mark.parametrize("direction_d2h", [True, False], ids=["d2h", "h2d"])
def test_object_group_plan_matches_direct_launches(direction_d2h: bool) -> None:
    assert lmc_ops.PageBufferShapeDesc is native.PageBufferShapeDesc
    device = torch.device("npu:0")
    nl, nb, bs, chunk = 2, 4, 32, 32
    desc17 = _kg0_desc(nl=nl, nb=nb, bs=bs)
    assert desc17.dtype is torch.int8
    _pa, layers17_d = _kg0_layers(nl=nl, nb=nb, bs=bs, device=device)
    _pb, layers17_p = _kg0_layers(nl=nl, nb=nb, bs=bs, device=device)
    table17_d = _pointer_table(layers17_d, device)
    table17_p = _pointer_table(layers17_p, device)
    obj17_d = torch.zeros((nl, chunk, 130), dtype=torch.uint8, device=device)
    obj17_p = obj17_d.clone()

    hs, dtype = 512, torch.bfloat16
    desc13 = _shape_desc(kv_size=1, nl=nl, nb=nb, bs=bs, nh=1, hs=hs, dtype=dtype)
    layers13_d = _nh_cs_layers(nl=nl, nb=nb, bs=bs, hs=hs, dtype=dtype, device=device)
    layers13_p = _nh_cs_layers(nl=nl, nb=nb, bs=bs, hs=hs, dtype=dtype, device=device)
    table13_d = _pointer_table(layers13_d, device)
    table13_p = _pointer_table(layers13_p, device)
    obj13_d = torch.zeros((nl, chunk, hs), dtype=dtype, device=device)
    obj13_p = obj13_d.clone()
    block_ids = torch.tensor([0], dtype=torch.int64, device=device)
    direction = D2H if direction_d2h else H2D

    if not direction_d2h:
        packed = _kg0_packed_object(nl, chunk, device)
        obj17_d.copy_(packed)
        obj17_p.copy_(packed)
        src13 = (
            torch.arange(nl * chunk * hs, dtype=torch.float32, device=device)
            .view(nl, chunk, hs)
            .to(dtype)
        )
        obj13_d.copy_(src13)
        obj13_p.copy_(src13)
        _zero_engine(layers17_d)
        _zero_engine(layers17_p)
        _zero_engine(layers13_d)
        _zero_engine(layers13_p)

    _transfer(
        table17_d,
        [int(obj17_d.data_ptr())],
        block_ids,
        device,
        direction,
        desc17,
        chunk,
        KG0_FMT,
    )
    _transfer(
        table13_d,
        [int(obj13_d.data_ptr())],
        block_ids,
        device,
        direction,
        desc13,
        chunk,
        NH_CS_FMT,
    )
    spec17 = lmc_ops.KernelGroupSpec(
        table17_p.data_ptr(),
        [obj17_p.data_ptr()],
        desc17,
        chunk,
        int(KG0_FMT),
        block_ids.data_ptr(),
        block_ids.numel(),
    )
    spec13 = lmc_ops.KernelGroupSpec(
        table13_p.data_ptr(),
        [obj13_p.data_ptr()],
        desc13,
        chunk,
        int(NH_CS_FMT),
        block_ids.data_ptr(),
        block_ids.numel(),
    )
    step = lmc_ops.BatchStep(
        [],
        [lmc_ops.LaunchVar(0, 0, 1, 1, 0), lmc_ops.LaunchVar(1, 0, 1, 1, 0)],
    )
    lmc_ops.execute_object_group_transfer(
        int(direction), device, 1 << 26, [spec17, spec13], [step]
    )
    torch.npu.synchronize()
    if direction_d2h:
        assert torch.equal(obj17_d.cpu(), obj17_p.cpu())
        assert torch.equal(obj13_d.cpu(), obj13_p.cpu())
        return
    _assert_engine_equal(layers17_d, layers17_p)
    _assert_engine_equal(layers13_d, layers13_p)


@requires_npu
@pytest.mark.parametrize("affinity", [False, True], ids=["main", "affinity"])
def test_plan_staging_d2h_matches_direct(affinity: bool) -> None:
    device = torch.device("npu:0")
    nl, nb, bs, chunk = 2, 4, 32, 32
    desc = _kg0_desc(nl=nl, nb=nb, bs=bs)
    _pools, layers = _kg0_layers(nl=nl, nb=nb, bs=bs, device=device)
    table = _pointer_table(layers, device)
    golden = torch.zeros((nl, chunk, 130), dtype=torch.uint8, device=device)
    temp = golden.clone()
    host = torch.zeros((nl, chunk, 130), dtype=torch.uint8)
    block_ids = torch.tensor([0], dtype=torch.int64, device=device)
    _transfer(table, [int(golden.data_ptr())], block_ids, device, D2H, desc, chunk, KG0_FMT)
    spec = lmc_ops.KernelGroupSpec(
        table.data_ptr(),
        [temp.data_ptr()],
        desc,
        chunk,
        int(KG0_FMT),
        block_ids.data_ptr(),
        block_ids.numel(),
    )
    staging = [
        lmc_ops.StagingCopy(int(host.data_ptr()), int(temp.data_ptr()), host.nbytes, 0)
    ]
    step = lmc_ops.BatchStep(staging, [lmc_ops.LaunchVar(0, 0, 1, 1, 0)])

    def _store() -> None:
        lmc_ops.execute_object_group_transfer(
            int(D2H), device, 1 << 26, [spec], [step]
        )
        torch.npu.synchronize()

    _run(_store, affinity=affinity)
    assert torch.equal(host, golden.cpu())


# ---------------------------------------------------------------------------
# Contract: lmcache_memcpy_async is a correct H2D/D2H of pageable host memory.
# ---------------------------------------------------------------------------


@requires_npu
def test_lmcache_memcpy_async_host_roundtrip() -> None:
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
