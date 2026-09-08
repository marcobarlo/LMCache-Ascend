# SPDX-License-Identifier: Apache-2.0
"""Regression tests for Ascend LMCache retrieval."""

# Standard
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

# Third Party
import torch

# First Party
from lmcache_ascend.v1.cache_engine import AscendLMCacheEngine


class _RetrieveStats:
    def profile_process_tokens(self):
        return nullcontext()

    def profile_to_gpu(self):
        return nullcontext()

    def profile_broadcast(self):
        return nullcontext()

    def time_to_retrieve(self):
        return 0.001


def _make_retrieve_engine(*, save_only_first_rank: bool, chunks) -> AscendLMCacheEngine:
    engine = object.__new__(AscendLMCacheEngine)
    stats = _RetrieveStats()
    engine.gpu_connector = SimpleNamespace(
        batched_to_gpu=Mock(),
        to_gpu=Mock(),
    )
    engine.is_healthy = lambda: True
    engine._get_req_id = lambda _kwargs: "test-request"
    engine._log_kvcache_for_check = lambda **_kwargs: None
    engine.stats_monitor = SimpleNamespace(
        on_retrieve_request=lambda _num_tokens: stats,
        on_retrieve_finished=Mock(),
    )
    engine._is_passive = lambda: False
    engine.async_loading = False
    engine.save_only_first_rank = save_only_first_rank
    engine.remove_after_retrieve = False
    engine.metadata = SimpleNamespace(
        is_first_rank=lambda: True, worker_id=0, first_rank=0
    )

    def process_tokens(tokens, mask, ret_mask, **kwargs):
        ret_mask[:] = True
        return chunks, 1024

    engine._process_tokens_internal = process_tokens
    return engine


def test_retrieve_loads_cache_to_gpu_in_normal_mode():
    """A cache hit must copy the retrieved CPU KV into the serving KV cache."""
    memory_obj_0 = SimpleNamespace(ref_count_down=Mock())
    memory_obj_1 = SimpleNamespace(ref_count_down=Mock())
    chunks = [
        (object(), memory_obj_0, 0, 256),
        (object(), memory_obj_1, 256, 512),
    ]
    engine = _make_retrieve_engine(save_only_first_rank=False, chunks=chunks)

    slot_mapping = object()
    result = engine.retrieve(
        list(range(512)),
        slot_mapping=slot_mapping,
    )

    engine.gpu_connector.batched_to_gpu.assert_called_once_with(
        [memory_obj_0, memory_obj_1],
        [0, 256],
        [256, 512],
        slot_mapping=slot_mapping,
    )
    assert torch.all(result)
    memory_obj_0.ref_count_down.assert_called_once_with()
    memory_obj_1.ref_count_down.assert_called_once_with()


def test_retrieve_save_only_first_rank_uses_sharded_pipeline():
    """Sharded retrieve must not go through batched_to_gpu."""
    memory_obj = SimpleNamespace(ref_count_down=Mock())
    chunks = [(object(), memory_obj, 0, 256)]
    engine = _make_retrieve_engine(save_only_first_rank=True, chunks=chunks)
    engine._pipelined_sharded_broadcast_and_load = Mock()

    engine.retrieve(list(range(256)), slot_mapping=object())

    engine.gpu_connector.batched_to_gpu.assert_not_called()
    engine._pipelined_sharded_broadcast_and_load.assert_called_once()
