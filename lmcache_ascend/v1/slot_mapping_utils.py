# SPDX-License-Identifier: Apache-2.0
"""Slot-mapping helpers for multi-plane KV transfer (slice bounds, compaction)."""

# Future
from __future__ import annotations

# Standard
from typing import Sequence

# Third Party
import torch


def multi_plane_slot_slice_bounds(
    token_start: int,
    token_end: int,
    sched_g: int,
    compress_ratios: Sequence[int],
    sm_len: int,
) -> tuple[int, int]:
    """Map a global token span to compressed slot bounds in the slot-mapping tensor.

    Each scheduler group ``sched_g`` has its own compression ratio. Given logical token
    indices ``[token_start, token_end)`` returns ``(s0, s1)`` such that
    ``sm[s0:s1]`` covers exactly those tokens for that group. ``sm_len`` is the
    length of that group's full slot-mapping tensor (``len(sm)``).

    Raises:
        ValueError: If ``sched_g`` is out of range for ``compress_ratios``, if
            ``ratio < 1``, or if the computed slice ``[s0, s1)`` falls outside
            ``[0, sm_len)``.

    Returns ``(0, 0)`` when the token span is empty.
    """
    if token_end <= token_start:
        return 0, 0
    if sched_g < 0 or sched_g >= len(compress_ratios):
        raise ValueError(
            f"scheduler group {sched_g} out of range for compress_ratios "
            f"(len={len(compress_ratios)})"
        )
    ratio = int(compress_ratios[sched_g])
    if ratio < 1:
        raise ValueError(f"compress_ratios[{sched_g}] must be >= 1, got {ratio}")
    if ratio <= 1:
        s0, s1 = token_start, token_end
    else:
        s0 = token_start // ratio
        s1 = (token_end + ratio - 1) // ratio
    if s0 < 0 or s1 > sm_len:
        raise ValueError(
            f"slot slice [{s0}, {s1}) out of range for sm length {sm_len} "
            f"(range [{token_start}, {token_end}), sched_g={sched_g}, ratio={ratio})"
        )
    return s0, s1


def dense_bounds_from_prefix(
    prefix: torch.Tensor,
    s0: int,
    s1: int,
) -> tuple[int, int]:
    """Map full-``sm`` row bounds ``[s0, s1)`` to a dense filtered slice.

    ``prefix`` comes from :func:`build_filtered_slot_mappings`; ``prefix[k]`` is
    the count of non-``-1`` rows in ``sm[0:k)``. Returns
    ``(dense_start, dense_count)`` so that
    ``filtered[dense_start : dense_start + dense_count]`` equals the valid
    entries in ``sm[s0:s1]``.
    """
    if s1 <= s0:
        return 0, 0
    dense_start = int(prefix[s0])
    dense_count = int(prefix[s1]) - dense_start
    return dense_start, dense_count


def build_filtered_slot_mappings(
    slot_mappings_by_group: tuple[torch.Tensor, ...] | list[torch.Tensor],
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    """Precompute per-group compacted slot mappings and prefix lookup tables.

    For each scheduler group, strips ``-1`` (invalid) rows from ``sm`` and
    builds a prefix array where ``prefix[g][k]`` counts valid slots in
    ``sm[g][0:k)``. Enables O(1) per-chunk dense slicing via
    :func:`dense_bounds_from_prefix` instead of scanning ``-1`` on every
    transfer.
    """
    filtered: list[torch.Tensor] = []
    prefixes: list[torch.Tensor] = []
    for sched_g, sm in enumerate(slot_mappings_by_group):
        if sm.numel() == 0:
            filtered.append(sm)
            prefixes.append(torch.zeros(1, dtype=torch.int32))
            continue
        valid = (sm != -1).to(torch.int32)
        prefix = torch.cat(
            [torch.zeros(1, dtype=torch.int32), torch.cumsum(valid, dim=0)]
        )
        filtered.append(sm[sm != -1])
        prefixes.append(prefix)
    return tuple(filtered), tuple(prefixes)
