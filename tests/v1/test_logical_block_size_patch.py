# SPDX-License-Identifier: Apache-2.0
"""Wrap ``get_tokens_per_block`` for pre-/post-#13242 Ascend compressed MLA."""

from __future__ import annotations

from dataclasses import dataclass, field

import lmcache.integration.vllm.kv_cache_groups as kg
from lmcache_ascend.integration.vllm.logical_block_size import (
    install_on,
    install_overrides,
    tokens_per_block_id,
)


class AttentionSpec:
    """Name used by ``get_tokens_per_block`` DCP MRO detection."""


@dataclass(frozen=True)
class AscendMLAAttentionSpec(AttentionSpec):
    block_size: int
    compress_ratio: int = 1
    model_version: str | None = None
    storage_block_size: int | None = None

    @classmethod
    def merge(cls, specs: list["AscendMLAAttentionSpec"]) -> "AscendMLAAttentionSpec":
        return cls(block_size=specs[0].block_size)


@dataclass(frozen=True)
class AscendSlidingWindowMLASpec:
    block_size: int
    compress_ratio: int = 1
    sliding_window: int = 128
    storage_block_size: int | None = None


@dataclass
class UniformTypeKVCacheSpecs:
    block_size: int
    kv_cache_specs: dict = field(default_factory=dict)


@dataclass(frozen=True)
class MLAAttentionSpec(AttentionSpec):
    """Upstream GPU MLA: ``block_size`` is already logical tokens."""

    block_size: int
    compress_ratio: int = 1


def _gtpb(spec: object, dcp_size: int) -> int:
    install_overrides()
    return kg.get_tokens_per_block(spec, dcp_size)


def test_pre_13242_physical_block_times_compress_ratio() -> None:
    spec = AscendMLAAttentionSpec(block_size=32, compress_ratio=128)
    assert tokens_per_block_id(spec) == 4096
    assert _gtpb(spec, 1) == 4096
    assert _gtpb(spec, 2) == 8192
    swa = AscendSlidingWindowMLASpec(block_size=32, compress_ratio=4)
    assert _gtpb(swa, 1) == 128


def test_pre_13242_storage_equal_to_block_still_scales() -> None:
    spec = AscendMLAAttentionSpec(
        block_size=32, compress_ratio=128, storage_block_size=32
    )
    assert _gtpb(spec, 1) == 4096
    assert _gtpb(spec, 2) == 8192


def test_post_13242_storage_distinct_from_logical_block() -> None:
    spec = AscendMLAAttentionSpec(
        block_size=4096, compress_ratio=128, storage_block_size=32
    )
    assert tokens_per_block_id(spec) == 4096
    assert _gtpb(spec, 1) == 4096
    assert _gtpb(spec, 2) == 8192


def test_uniform_type_is_max_of_ascend_leaves() -> None:
    leaf = AscendMLAAttentionSpec(block_size=32, compress_ratio=128)
    wrapped = UniformTypeKVCacheSpecs(
        block_size=32, kv_cache_specs={"l0": leaf, "l1": leaf}
    )
    assert tokens_per_block_id(wrapped) == 4096
    assert _gtpb(wrapped, 1) == 4096
    assert _gtpb(wrapped, 2) == 8192


def test_non_ascend_mla_compress_ratio_does_not_scale() -> None:
    spec = MLAAttentionSpec(block_size=32, compress_ratio=128)
    assert tokens_per_block_id(spec) == 32
    assert _gtpb(spec, 1) == 32
    assert _gtpb(spec, 2) == 64


def test_merge_restores_dropped_compress_ratio() -> None:
    install_on((AscendMLAAttentionSpec,))
    merged = AscendMLAAttentionSpec.merge(
        [
            AscendMLAAttentionSpec(
                block_size=32, compress_ratio=128, model_version="deepseek_v4"
            )
            for _ in range(2)
        ]
    )
    assert merged.compress_ratio == 128
    assert merged.model_version == "deepseek_v4"
    assert tokens_per_block_id(merged) == 4096


def test_install_overrides_is_idempotent() -> None:
    install_overrides()
    install_overrides()
    spec = AscendMLAAttentionSpec(block_size=32, compress_ratio=128)
    assert _gtpb(spec, 1) == 4096
    assert kg.get_tokens_per_block is not tokens_per_block_id
    assert getattr(kg.get_tokens_per_block, "_lmcache_ascend_logical_block_size", False)
