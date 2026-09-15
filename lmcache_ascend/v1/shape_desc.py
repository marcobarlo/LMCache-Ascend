# SPDX-License-Identifier: Apache-2.0
"""Ascend helpers for fmt-17 extras on the plugin PageBufferShapeDesc.

CUDA's compiled struct is unchanged. After bind_native, factory/grouping
construct ``device_ops.PageBufferShapeDesc()`` (``c_ops``). NPU kernels
take that class by value.
"""

from __future__ import annotations

from collections.abc import Sequence

_MAX_PLANES = 4


def plane_slot_bytes_of(desc: object) -> tuple[int, ...]:
    """Return per-token plane widths in bytes.

    Prefers ``plane_slot_bytes``. If unset, derives from MP's
    ``plane_widths * plane_dtypes.itemsize`` (fmt-17 side channel).
    """
    slots = tuple(int(b) for b in (getattr(desc, "plane_slot_bytes", ()) or ()))
    if slots:
        return slots
    widths = tuple(getattr(desc, "plane_widths", ()) or ())
    if not widths:
        return ()
    dtypes = tuple(getattr(desc, "plane_dtypes", ()) or ())
    out: list[int] = []
    default_itemsize = int(getattr(desc, "element_size", 1) or 1)
    for i, width in enumerate(widths):
        itemsize = (
            int(dtypes[i].itemsize) if i < len(dtypes) else default_itemsize
        )
        out.append(int(width) * itemsize)
    return tuple(out)


def is_packed_two_plane(desc: object) -> bool:
    """True when ``desc`` is the thin-tail packed-MLA layout (G0).

    Requires two planes, 32 B-aligned latent, scale in ``(0, 32)``,
    and LMC row bytes equal to the sum of the two planes.
    """
    planes = plane_slot_bytes_of(desc)
    n_planes = int(getattr(desc, "num_planes", 0) or 0) or len(planes)
    if n_planes != 2 or len(planes) < 2:
        return False
    p0, p1 = int(planes[0]), int(planes[1])
    row = int(desc.hs) * int(desc.element_size)
    bs = int(desc.bs)
    return (
        p0 > 0
        and p0 % 32 == 0
        and 0 < p1 < 32
        and (p1 * bs) % 32 == 0
        and row == p0 + p1
    )


def attach_tuple_planes(
    desc: object,
    plane_slot_bytes: Sequence[int],
) -> None:
    """Record per-token plane widths (bytes) on ``desc``.

    ``plane_slot_bytes[i]`` is ``numel * itemsize // (nb * bs)`` for plane
    ``i``. Caps at ``_MAX_PLANES``; the packed kernel only accepts NP==2.
    """
    slots = tuple(int(b) for b in plane_slot_bytes)
    if len(slots) > _MAX_PLANES:
        raise ValueError(
            f"attach_tuple_planes: {len(slots)} planes exceeds max {_MAX_PLANES}"
        )
    desc.num_planes = len(slots)
    desc.plane_slot_bytes = slots


def attach_tuple_block_strides(
    desc: object,
    plane_block_stride_bytes: Sequence[int],
) -> None:
    """Record per-block dim-0 byte strides on ``desc``.

    ``plane_block_stride_bytes[i]`` is ``stride(0) * itemsize`` for plane
    ``i``. Packed native uses plane 0/1; 0 means fall back to the shared
    ``block_stride_elems``.
    """
    strides = tuple(int(b) for b in plane_block_stride_bytes)
    if len(strides) > _MAX_PLANES:
        raise ValueError(
            "attach_tuple_block_strides: "
            f"{len(strides)} planes exceeds max {_MAX_PLANES}"
        )
    desc.plane_block_stride_bytes = strides


def has_packed_engine_strides(desc: object) -> bool:
    """True when packed native can address engine pages.

    Shared ``block_stride_elems`` or both packed-plane byte strides.
    """
    if int(getattr(desc, "block_stride_elems", 0) or 0) > 0:
        return True
    strides = tuple(getattr(desc, "plane_block_stride_bytes", ()) or ())
    return len(strides) >= 2 and int(strides[0]) > 0 and int(strides[1]) > 0
