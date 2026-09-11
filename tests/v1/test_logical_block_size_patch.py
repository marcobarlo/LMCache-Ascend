# SPDX-License-Identifier: Apache-2.0
"""Runtime ``logical_block_size`` patch (no vLLM-Ascend source edits)."""

from __future__ import annotations

from dataclasses import dataclass, field

from lmcache.integration.vllm.kv_cache_groups import get_tokens_per_block
from lmcache_ascend.integration.vllm.logical_block_size import install_on


class AttentionSpec:
    """Name used by ``get_tokens_per_block`` DCP MRO detection."""


@dataclass(frozen=True)
class AscendMLAAttentionSpec(AttentionSpec):
    block_size: int
    compress_ratio: int = 1
    model_version: str | None = None

    @classmethod
    def merge(cls, specs: list["AscendMLAAttentionSpec"]) -> "AscendMLAAttentionSpec":
        return cls(block_size=specs[0].block_size)


@dataclass(frozen=True)
class AscendSlidingWindowMLASpec:
    block_size: int
    compress_ratio: int = 1
    sliding_window: int = 128


@dataclass
class UniformTypeKVCacheSpecs:
    block_size: int
    kv_cache_specs: dict = field(default_factory=dict)


def test_leaf_logical_block_size_is_physical_times_compress_ratio() -> None:
    install_on((AscendMLAAttentionSpec, AscendSlidingWindowMLASpec))
    spec = AscendMLAAttentionSpec(block_size=32, compress_ratio=128)
    assert spec.logical_block_size == 4096
    swa = AscendSlidingWindowMLASpec(block_size=32, compress_ratio=4)
    assert swa.logical_block_size == 128


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
    assert merged.logical_block_size == 4096


def test_uniform_type_logical_block_size_is_max_of_leaves() -> None:
    install_on((AscendMLAAttentionSpec,), UniformTypeKVCacheSpecs)
    leaf = AscendMLAAttentionSpec(block_size=32, compress_ratio=128)
    wrapped = UniformTypeKVCacheSpecs(
        block_size=32, kv_cache_specs={"l0": leaf, "l1": leaf}
    )
    assert wrapped.logical_block_size == 4096
    assert get_tokens_per_block(wrapped, 1) == 4096
    assert get_tokens_per_block(wrapped, 2) == 8192


def test_install_on_is_idempotent() -> None:
    install_on((AscendMLAAttentionSpec,), UniformTypeKVCacheSpecs)
    install_on((AscendMLAAttentionSpec,), UniformTypeKVCacheSpecs)
    spec = AscendMLAAttentionSpec(block_size=32, compress_ratio=128)
    assert spec.logical_block_size == 4096


def test_get_tokens_per_block_reads_patched_property() -> None:
    install_on((AscendMLAAttentionSpec,))
    spec = AscendMLAAttentionSpec(block_size=32, compress_ratio=128)
    assert get_tokens_per_block(spec, 1) == 4096
    assert get_tokens_per_block(spec, 2) == 8192
