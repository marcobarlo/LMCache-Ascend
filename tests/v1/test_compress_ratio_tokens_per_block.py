# SPDX-License-Identifier: Apache-2.0
"""CPU tests for the vLLM-Ascend compress_ratio tokens_per_block overlay."""

from __future__ import annotations

from types import SimpleNamespace

from lmcache.integration.vllm.lmcache_mp_metadata import (
    LMCacheMPRequestMetadata,
    LMCacheMPRequestState,
    LMCacheMPRequestTracker,
)
from lmcache_ascend.integration.vllm.kv_cache_groups import (
    group_compress_ratio,
    group_sliding_window,
    iter_layer_specs,
    logical_tokens_per_block,
)


class _FakeSpec:
    def __init__(
        self,
        block_size: int,
        compress_ratio: int | None = None,
        sliding_window: int | None = None,
    ) -> None:
        self.block_size = block_size
        self.compress_ratio = compress_ratio
        self.sliding_window = sliding_window


class UniformTypeKVCacheSpecs:
    """Name matches vLLM's bundle class so ``iter_layer_specs`` unwraps it."""

    def __init__(self, specs: dict[str, _FakeSpec]) -> None:
        self.kv_cache_specs = specs
        self.block_size = next(iter(specs.values())).block_size


def _core_tpb(spec: object, dcp_size: int) -> int:
    return int(getattr(spec, "block_size")) * max(dcp_size, 1)


def test_group_compress_ratio_plain_spec() -> None:
    assert group_compress_ratio(_FakeSpec(8, compress_ratio=128)) == 128
    assert group_compress_ratio(_FakeSpec(8)) == 1
    assert group_compress_ratio(_FakeSpec(8, compress_ratio=0)) == 1


def test_group_compress_ratio_uniform_bundle_takes_max() -> None:
    bundle = UniformTypeKVCacheSpecs(
        {
            "a": _FakeSpec(8, compress_ratio=4),
            "b": _FakeSpec(8, compress_ratio=128),
        }
    )
    assert type(bundle).__name__ == "UniformTypeKVCacheSpecs"
    assert group_compress_ratio(bundle) == 128
    assert list(iter_layer_specs(bundle))  # non-empty


def test_logical_tokens_per_block_multiplies_ratio() -> None:
    spec = _FakeSpec(8, compress_ratio=128)
    assert (
        logical_tokens_per_block(spec, dcp_size=1, get_tokens_per_block=_core_tpb)
        == 1024
    )


def test_logical_tokens_per_block_unchanged_without_ratio() -> None:
    spec = _FakeSpec(16)
    assert (
        logical_tokens_per_block(spec, dcp_size=1, get_tokens_per_block=_core_tpb)
        == 16
    )


def test_group_sliding_window_from_bundle() -> None:
    bundle = UniformTypeKVCacheSpecs({"a": _FakeSpec(8, sliding_window=128)})
    assert group_sliding_window(bundle) == 128
    assert group_sliding_window(_FakeSpec(8)) is None


def _make_tracker(
    n_tokens: int, blocks: dict[int, int]
) -> LMCacheMPRequestTracker:
    tracker = LMCacheMPRequestTracker.__new__(LMCacheMPRequestTracker)
    tracker.request_id = "dsv4-test"
    tracker.all_token_ids = list(range(n_tokens))
    tracker.allocated_block_ids = {g: list(range(n)) for g, n in blocks.items()}
    tracker.num_scheduled_tokens = n_tokens
    tracker.num_stored_tokens = 0
    tracker.num_vllm_hit_tokens = 0
    tracker.num_lmcache_hit_tokens = 0
    tracker.state = LMCacheMPRequestState.READY
    tracker.cache_salt = ""
    tracker.mm_adjusted_prompt_ids = []
    return tracker


def test_get_store_metadata_physical_tpb_yields_zero_chunks() -> None:
    """Reproduces the pre-fix gate: physical tpb under-counts eg1."""
    tracker = _make_tracker(
        4911,
        {0: 10, 1: 1, 2: 39, 3: 39, 4: 614, 5: 154},
    )
    meta = LMCacheMPRequestMetadata.GetStoreMetadata(
        tracker,
        lmcache_tokens_per_chunk=256,
        group_tokens_per_block=[128, 128, 128, 128, 8, 32],
    )
    assert meta is None


def test_get_store_metadata_logical_tpb_stores_chunks() -> None:
    """B=8, ratios (4, 128, 1, 1, 1, 1) → logical tpb and enough chunks."""
    ntok = 4911
    tpb = [32, 1024, 8, 8, 8, 32]
    chunk = 1024
    blocks = {i: (ntok + t - 1) // t for i, t in enumerate(tpb)}
    tracker = _make_tracker(ntok, blocks)
    meta = LMCacheMPRequestMetadata.GetStoreMetadata(
        tracker,
        lmcache_tokens_per_chunk=chunk,
        group_tokens_per_block=tpb,
    )
    assert meta is not None
    assert meta.direction == "STORE"
    assert (meta.op.end - meta.op.start) >= chunk
    assert (meta.op.end - meta.op.start) % chunk == 0


def test_patch_rebinds_get_tokens_per_block() -> None:
    # Third Party
    import lmcache.integration.vllm.kv_cache_groups as kv_cache_groups
    import lmcache.integration.vllm.lmcache_mp_connector as mp_conn

    # First Party
    import lmcache_ascend
    from lmcache_ascend.integration.vllm.kv_cache_groups import (
        apply_get_tokens_per_block_patch,
    )

    lmcache_ascend._patch_kv_cache_groups()
    apply_get_tokens_per_block_patch()  # idempotent
    spec = SimpleNamespace(block_size=8, compress_ratio=128)
    assert kv_cache_groups.get_tokens_per_block(spec, 1) == 1024
    assert mp_conn.get_tokens_per_block is kv_cache_groups.get_tokens_per_block
    assert mp_conn.get_tokens_per_block(spec, 1) == 1024
