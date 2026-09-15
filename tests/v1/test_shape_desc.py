# SPDX-License-Identifier: Apache-2.0
"""NpuPageBufferShapeDesc plane-slot contract (no NPU)."""

from __future__ import annotations

import pytest
import torch

from lmcache.v1.platform.npu.shape_desc import NpuPageBufferShapeDesc
from lmcache_ascend.v1.shape_desc import (
    attach_tuple_block_strides,
    attach_tuple_planes,
    has_packed_engine_strides,
    is_packed_two_plane,
)


def _desc(*, hs: int, element_size: int, bs: int = 32) -> NpuPageBufferShapeDesc:
    desc = NpuPageBufferShapeDesc()
    desc.kv_size = 1
    desc.nl = 1
    desc.nb = 4
    desc.bs = bs
    desc.nh = 1
    desc.hs = hs
    desc.element_size = element_size
    desc.block_stride_elems = 0
    return desc


def test_unset_planes_are_not_packed() -> None:
    desc = NpuPageBufferShapeDesc()
    desc.hs = 130
    desc.element_size = 1
    desc.bs = 32
    desc.block_stride_elems = 0
    assert desc.num_planes == 0
    assert desc.plane_slot_bytes == ()
    assert desc.plane_block_stride_bytes == ()
    assert not is_packed_two_plane(desc)
    assert not has_packed_engine_strides(desc)


def test_g0_128_plus_2_is_packed() -> None:
    desc = _desc(hs=130, element_size=1)
    attach_tuple_planes(desc, (128, 2))
    assert desc.num_planes == 2
    assert desc.plane_slot_bytes == (128, 2)
    assert is_packed_two_plane(desc)


def test_hs130_without_planes_is_not_packed() -> None:
    desc = _desc(hs=130, element_size=1)
    assert not is_packed_two_plane(desc)


def test_np1_dense_row_is_not_packed() -> None:
    desc = _desc(hs=512, element_size=2)
    attach_tuple_planes(desc, (1024,))
    assert desc.num_planes == 1
    assert not is_packed_two_plane(desc)


def test_mla_512_64_bf16_is_not_packed() -> None:
    desc = _desc(hs=576, element_size=2)
    attach_tuple_planes(desc, (1024, 128))
    assert desc.num_planes == 2
    assert not is_packed_two_plane(desc)


def test_dsa_three_plane_is_not_packed() -> None:
    desc = _desc(hs=704, element_size=2)
    attach_tuple_planes(desc, (1024, 128, 256))
    assert desc.num_planes == 3
    assert not is_packed_two_plane(desc)


def test_inflated_hs_does_not_pack() -> None:
    desc = _desc(hs=160, element_size=1)
    attach_tuple_planes(desc, (128, 2))
    assert not is_packed_two_plane(desc)


def test_plane_widths_side_channel_is_packed() -> None:
    desc = _desc(hs=130, element_size=1)
    desc.plane_widths = (128, 1)
    desc.plane_dtypes = (torch.int8, torch.float16)
    assert is_packed_two_plane(desc)


def test_attach_unequal_block_strides_is_packed_native() -> None:
    desc = _desc(hs=130, element_size=1)
    attach_tuple_planes(desc, (128, 2))
    attach_tuple_block_strides(desc, (4160, 96))
    assert desc.plane_block_stride_bytes == (4160, 96)
    assert is_packed_two_plane(desc)
    assert has_packed_engine_strides(desc)


def test_attach_rejects_more_than_four_planes() -> None:
    desc = NpuPageBufferShapeDesc()
    with pytest.raises(ValueError, match="exceeds max"):
        attach_tuple_planes(desc, (1, 2, 3, 4, 5))
