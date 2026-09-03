# SPDX-License-Identifier: Apache-2.0
"""Regression tests for Ascend LMCache retrieval."""

# Standard
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

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


class _StreamCM:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def _make_retrieve_engine(*, save_only_first_rank: bool, chunks) -> AscendLMCacheEngine:
    engine = object.__new__(AscendLMCacheEngine)
    stats = _RetrieveStats()
    engine.gpu_connector = SimpleNamespace(
        batched_to_gpu=Mock(),
        _ensure_mp_launch_meta_for_batch=Mock(),
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


def test_sharded_pipeline_precomputes_mp_launch_meta_before_togpu():
    """_pipeline_broadcast_and_load must ensure launch meta before to_gpu."""
    engine = object.__new__(AscendLMCacheEngine)
    call_order: list[str] = []

    def ensure(starts, ends, kwargs, *, stream):
        call_order.append("ensure")
        kwargs["mp_launch_meta"] = {"ok": True}
        assert list(starts) == [0, 256]
        assert list(ends) == [256, 512]

    def submit_togpu(_ctx, _load_stream, _pending, **kwargs):
        call_order.append("togpu")
        assert kwargs.get("mp_launch_meta") == {"ok": True}

    mem_obj = SimpleNamespace(ref_count_down=Mock())
    engine.gpu_connector = SimpleNamespace(_ensure_mp_launch_meta_for_batch=ensure)
    engine.metadata = SimpleNamespace(
        worker_id=0, first_rank=0, is_first_rank=lambda: True
    )
    engine._ensure_merged_pool = Mock(return_value=True)
    engine._merged_pool = [
        torch.empty(8, dtype=torch.uint8),
        torch.empty(8, dtype=torch.uint8),
    ]
    engine._pool_scatter_ev = [Mock(), Mock()]
    engine.broadcast_stream = object()
    engine.broadcast_fn = Mock()
    engine._submit_togpu = submit_togpu
    engine._fill_shard_sender = Mock(return_value=([mem_obj], [0], [256]))

    plan = {
        "meta": [(0, 256, {}), (256, 512, {})],
        "shard_plan": [0],
        "shard_layouts": [[(0, 0, 8)]],
        "max_shard_bytes": 8,
    }
    load_stream = SimpleNamespace(synchronize=Mock(), wait_event=Mock())
    fake_npu = SimpleNamespace(
        stream=lambda *_a, **_k: _StreamCM(),
        Event=lambda: SimpleNamespace(record=Mock()),
    )

    with patch.object(torch, "npu", fake_npu, create=True):
        engine._pipeline_broadcast_and_load(
            plan,
            load_stream,
            reordered_chunks=[(object(), mem_obj, 0, 256)],
            slot_mapping=object(),
        )

    assert call_order[0] == "ensure"
    assert "togpu" in call_order
    assert call_order.index("ensure") < call_order.index("togpu")
    load_stream.synchronize.assert_called()
