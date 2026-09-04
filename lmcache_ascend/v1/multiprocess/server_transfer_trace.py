# SPDX-License-Identifier: Apache-2.0
"""Named spans + counters for lmcache_driven server-side KV transfer.

Under ``lmcache_driven`` the LMCache *server* runs:
  - ``device_ops.multi_layer_block_kv_transfer`` (npu_block_transfer / torch_ops)
  - ``device_ops.lmcache_memcpy_async`` (staging D2H/H2D via ``Tensor.copy_``)

``block_kv_fused_launches`` counts ``multi_layer_kv_transfer_kernel_v3``
calls on the shim path. Also wraps ``transfer_kv_per_object_group`` so
counters cannot miss when ``c_ops`` holds a stale unbound reference.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from contextlib import contextmanager
from functools import wraps
from pathlib import Path
from typing import Any, Iterator

import torch

from lmcache.logging import init_logger

logger = init_logger(__name__)

_installed = False
_orig_block_kv = None
_orig_memcpy = None
_orig_transfer_kv = None

_lock = threading.Lock()
_counters: dict[str, int | float] = {
    "block_kv_launches": 0,
    "block_kv_chunks": 0,
    "block_kv_copy_calls": 0,
    "block_kv_index_copy_calls": 0,
    "block_kv_fused_launches": 0,
    "memcpy_launches": 0,
    "memcpy_bytes": 0,
    "transfer_kv_calls": 0,
    "transfer_kv_objs": 0,
}


def reset_counters() -> None:
    with _lock:
        for k in _counters:
            _counters[k] = 0


def snapshot_counters() -> dict[str, float]:
    with _lock:
        return {k: float(v) for k, v in _counters.items()}


def _counter_log_path() -> Path | None:
    root = os.environ.get("LMCACHE_SERVER_PROFILER_DIR") or os.environ.get("TRACE_DIR")
    if not root:
        return None
    return Path(root) / "lmcache_transfer_counters.log"


def _emit(msg: str) -> None:
    logger.warning("%s", msg)
    print(msg, flush=True)
    path = _counter_log_path()
    if path is None:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(msg + "\n")
    except OSError:
        pass


@contextmanager
def _trace_span(name: str, stream: object = None) -> Iterator[None]:
    """CPU record_function + optional MSTX push/pop for Ascend chrome traces.

    ``mstx.mstx_range`` is a *decorator*, not a context manager. Use
    ``range_push`` / ``range_pop`` for runtime spans.
    """
    pushed = False
    try:
        import torch_npu.npu.mstx as mstx

        try:
            mstx.range_push(name)
            pushed = True
        except TypeError:
            # Some builds take (message, domain=...).
            mstx.range_push(name, domain="lmcache")
            pushed = True
        except Exception:
            pushed = False
    except Exception:
        pushed = False

    try:
        with torch.profiler.record_function(name):
            yield
    finally:
        if pushed:
            try:
                import torch_npu.npu.mstx as mstx

                mstx.range_pop()
            except Exception:
                pass


@contextmanager
def _count_tensor_copies() -> Iterator[tuple[list[int], list[int]]]:
    copy_n = [0]
    index_n = [0]
    orig_copy = torch.Tensor.copy_
    orig_index = torch.Tensor.index_copy_

    def _copy(self: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        copy_n[0] += 1
        return orig_copy(self, *args, **kwargs)

    def _index_copy(
        self: torch.Tensor, *args: object, **kwargs: object
    ) -> torch.Tensor:
        index_n[0] += 1
        return orig_index(self, *args, **kwargs)

    torch.Tensor.copy_ = _copy  # type: ignore[method-assign]
    torch.Tensor.index_copy_ = _index_copy  # type: ignore[method-assign]
    try:
        yield copy_n, index_n
    finally:
        torch.Tensor.copy_ = orig_copy  # type: ignore[method-assign]
        torch.Tensor.index_copy_ = orig_index  # type: ignore[method-assign]


def _direction_name(direction: object) -> str:
    try:
        ival = int(direction)
    except Exception:
        return str(direction)
    if ival == 0:
        return "H2D"
    if ival == 1:
        return "D2H"
    return f"dir{ival}"


def _wrap_block_kv(orig: Any) -> Any:
    @wraps(orig)
    def wrapped(*args: Any, **kwargs: Any) -> None:
        # Support both free-function and accidental bound-method call shapes.
        if args and not isinstance(args[0], (torch.Tensor, list, tuple)):
            # Likely (self, paged, objs, ...)
            self_ignored, args = args[0], args[1:]
            _ = self_ignored
        (
            paged_buffer_ptrs_tensor,
            lmcache_objects_ptrs,
            block_ids,
            device,
            direction,
            shape_desc,
            lmcache_chunk_size,
            engine_kv_format,
            skip_prefix_n_blocks,
        ) = args[:9] if len(args) >= 9 else (
            kwargs["paged_buffer_ptrs_tensor"],
            kwargs["lmcache_objects_ptrs"],
            kwargs["block_ids"],
            kwargs["device"],
            kwargs["direction"],
            kwargs["shape_desc"],
            kwargs["lmcache_chunk_size"],
            kwargs["engine_kv_format"],
            kwargs["skip_prefix_n_blocks"],
        )
        n_chunks = (
            len(lmcache_objects_ptrs)
            if isinstance(lmcache_objects_ptrs, (list, tuple))
            else 1
        )
        fmt = getattr(engine_kv_format, "name", str(engine_kv_format))
        dname = _direction_name(direction)
        span = (
            f"lmcache.multi_layer_block_kv_transfer/{fmt}/{dname}/"
            f"chunks={n_chunks}/chunk_tokens={lmcache_chunk_size}"
        )
        fused_before = 0
        try:
            from lmcache_ascend.v1.multiprocess import npu_block_transfer as _nbt

            fused_before = int(_nbt.fused_kernel_launches)
        except Exception:
            fused_before = 0
        with _trace_span(span):
            with _count_tensor_copies() as (copy_n, index_n):
                orig(
                    paged_buffer_ptrs_tensor,
                    lmcache_objects_ptrs,
                    block_ids,
                    device,
                    direction,
                    shape_desc,
                    lmcache_chunk_size,
                    engine_kv_format,
                    skip_prefix_n_blocks,
                )
        copies = copy_n[0]
        indexes = index_n[0]
        fused_delta = 0
        fallback_reason = ""
        try:
            from lmcache_ascend.v1.multiprocess import npu_block_transfer as _nbt

            fused_delta = int(_nbt.fused_kernel_launches) - fused_before
            fallback_reason = str(_nbt.last_fallback_reason or "")
        except Exception:
            fused_delta = 0
        with _lock:
            _counters["block_kv_launches"] = int(_counters["block_kv_launches"]) + 1
            _counters["block_kv_chunks"] = int(_counters["block_kv_chunks"]) + n_chunks
            _counters["block_kv_copy_calls"] = (
                int(_counters["block_kv_copy_calls"]) + copies
            )
            _counters["block_kv_index_copy_calls"] = (
                int(_counters["block_kv_index_copy_calls"]) + indexes
            )
            _counters["block_kv_fused_launches"] = (
                int(_counters["block_kv_fused_launches"]) + fused_delta
            )
        per_chunk_copy = copies / n_chunks if n_chunks else 0.0
        per_chunk_idx = indexes / n_chunks if n_chunks else 0.0
        impl = (
            "npu_block_transfer.multi_layer_kv_transfer_kernel_v3"
            if fused_delta
            else "torch_ops.Tensor.copy_|index_copy_"
        )
        kernel = (
            "multi_layer_kv_transfer_kernel_v3"
            if fused_delta
            else "aclnnInplaceCopy|index_copy"
        )
        _emit(
            f"[LMCACHE_COUNTERS] op=multi_layer_block_kv_transfer fmt={fmt} "
            f"dir={dname} chunks={n_chunks} chunk_tokens={lmcache_chunk_size} "
            f"impl={impl} kernel={kernel} fused_launches={fused_delta} "
            f"fallback={fallback_reason or '-'} "
            f"copy_calls={copies} index_copy_calls={indexes} "
            f"copy_per_chunk={per_chunk_copy:.3f} "
            f"index_copy_per_chunk={per_chunk_idx:.3f}"
        )

    return wrapped


def _wrap_memcpy(orig: Any) -> Any:
    def wrapped(
        self: Any,
        dest: int,
        src: int,
        nbytes: int,
        direction: Any,
        host_buffer_offset: int,
        host_buffer_alignments: int,
    ) -> None:
        dname = _direction_name(direction)
        span = f"lmcache.lmcache_memcpy_async/{dname}/bytes={nbytes}"
        with _trace_span(span):
            with _count_tensor_copies() as (copy_n, index_n):
                orig(
                    self,
                    dest,
                    src,
                    nbytes,
                    direction,
                    host_buffer_offset,
                    host_buffer_alignments,
                )
        copies = copy_n[0]
        with _lock:
            _counters["memcpy_launches"] = int(_counters["memcpy_launches"]) + 1
            _counters["memcpy_bytes"] = int(_counters["memcpy_bytes"]) + int(nbytes)
        _emit(
            f"[LMCACHE_COUNTERS] op=lmcache_memcpy_async dir={dname} "
            f"bytes={nbytes} impl=NpuDeviceOps.Tensor.copy_ "
            f"kernel=aclnnInplaceCopy|MEMCPY_ASYNC "
            f"copy_calls={copies} index_copy_calls={index_n[0]}"
        )

    return wrapped


def _wrap_transfer_kv(orig: Any) -> Any:
    def wrapped(*args: Any, **kwargs: Any) -> None:
        memory_objs = kwargs.get("memory_objs")
        if memory_objs is None and len(args) >= 3:
            memory_objs = args[2]
        n_objs = len(memory_objs) if memory_objs is not None else -1
        direction = kwargs.get("direction", args[6] if len(args) > 6 else "?")
        dname = _direction_name(direction)
        og = kwargs.get("object_group_id", args[3] if len(args) > 3 else "?")
        span = f"lmcache.transfer_kv_per_object_group/{dname}/og={og}/objs={n_objs}"
        with _trace_span(span):
            with _count_tensor_copies() as (copy_n, index_n):
                orig(*args, **kwargs)
        with _lock:
            _counters["transfer_kv_calls"] = int(_counters["transfer_kv_calls"]) + 1
            if n_objs > 0:
                _counters["transfer_kv_objs"] = int(_counters["transfer_kv_objs"]) + n_objs
        _emit(
            f"[LMCACHE_COUNTERS] op=transfer_kv_per_object_group dir={dname} "
            f"object_group_id={og} objs={n_objs} "
            f"host_copy_calls={copy_n[0]} host_index_copy_calls={index_n[0]} "
            f"copy_per_obj={(copy_n[0] / n_objs) if n_objs else 0:.3f}"
        )

    return wrapped


def _rebind_block_kv(wrapped: Any) -> None:
    """Rebind every known alias of multi_layer_block_kv_transfer."""
    from lmcache import device_ops
    from lmcache.v1.platform import torch_ops

    torch_ops.multi_layer_block_kv_transfer = wrapped  # type: ignore[assignment]
    object.__setattr__(device_ops, "multi_layer_block_kv_transfer", wrapped)

    # c_ops often keeps a *separate* function object copied at import time.
    try:
        import lmcache_ascend.c_ops as ascend_c_ops

        setattr(ascend_c_ops, "multi_layer_block_kv_transfer", wrapped)
    except Exception:
        pass
    try:
        import lmcache.c_ops as lm_c_ops

        setattr(lm_c_ops, "multi_layer_block_kv_transfer", wrapped)
    except Exception:
        pass


def install_server_transfer_trace() -> None:
    """Wrap server-side transfer entry points with spans and counters."""
    global _installed, _orig_block_kv, _orig_memcpy, _orig_transfer_kv
    if _installed:
        return

    from lmcache import device_ops
    from lmcache.v1.platform import torch_ops
    from lmcache.v1.platform.npu.device_ops import NpuDeviceOps

    device_ops.ensure_native()

    if _orig_block_kv is None:
        # Prefer the plugin shim. Wrapping DeviceOps.multi_layer_block_kv_transfer
        # and then rebinding torch_ops to that wrapper recurses: DeviceOps
        # delegates back to torch_ops.
        try:
            from lmcache_ascend.v1.multiprocess.npu_block_transfer import (
                multi_layer_block_kv_transfer as _npu_block_kv,
            )

            live = _npu_block_kv
        except Exception:
            live = getattr(device_ops, "multi_layer_block_kv_transfer", None)
        _orig_block_kv = live or torch_ops.multi_layer_block_kv_transfer
        _rebind_block_kv(_wrap_block_kv(_orig_block_kv))

    if _orig_memcpy is None:
        _orig_memcpy = NpuDeviceOps.lmcache_memcpy_async
        NpuDeviceOps.lmcache_memcpy_async = _wrap_memcpy(  # type: ignore[method-assign]
            _orig_memcpy
        )

    if _orig_transfer_kv is None:
        import lmcache.v1.multiprocess.modules.lmcache_driven_transfer as m

        _orig_transfer_kv = m.transfer_kv_per_object_group
        m.transfer_kv_per_object_group = _wrap_transfer_kv(  # type: ignore[assignment]
            _orig_transfer_kv
        )

    _installed = True
    logger.info(
        "Installed lmcache_driven server transfer spans/counters "
        "(block_kv + memcpy + transfer_kv_per_object_group)"
    )


def _is_lmcache_server_process() -> bool:
    argv = " ".join(sys.argv).lower()
    if "lmcache" in argv and "server" in argv:
        return True
    return os.environ.get("LMCACHE_IS_SERVER", "") in ("1", "true", "TRUE")


def _profiler_dir() -> str | None:
    return os.environ.get("LMCACHE_SERVER_PROFILER_DIR") or os.environ.get("TRACE_DIR")


def start_file_triggered_server_profiler() -> None:
    """File-triggered torch_npu profiler for the LMCache server process only."""
    if not _is_lmcache_server_process():
        logger.info("Skipping server profiler watcher (not lmcache server)")
        return
    root = _profiler_dir()
    if not root:
        logger.info("Server profiler idle (set LMCACHE_SERVER_PROFILER_DIR)")
        return
    if os.environ.get("LMCACHE_SERVER_PROFILER", "1") not in ("1", "true", "TRUE"):
        return

    t = threading.Thread(
        target=_server_profiler_loop,
        args=(root,),
        name="lmcache-server-profiler",
        daemon=True,
    )
    t.start()
    logger.info("LMCache server profiler watcher on %s", root)


def _server_profiler_loop(root: str) -> None:
    start_path = os.path.join(root, "server_profile.start")
    stop_path = os.path.join(root, "server_profile.stop")
    out_dir = os.path.join(root, "server_prof")
    os.makedirs(out_dir, exist_ok=True)
    profiler = None
    logger.warning("LMCache server profiler loop watching %s", root)

    while True:
        try:
            if profiler is None and os.path.exists(start_path):
                try:
                    os.remove(start_path)
                except OSError:
                    pass
                reset_counters()
                path = _counter_log_path()
                if path is not None:
                    try:
                        path.write_text("")
                    except OSError:
                        pass
                profiler = _make_npu_profiler(out_dir)
                if profiler is None:
                    _emit("[LMCACHE_SERVER_PROF] failed to create profiler")
                    time.sleep(0.5)
                    continue
                profiler.start()
                _emit(f"[LMCACHE_SERVER_PROF] START out={out_dir}")

            if profiler is not None and os.path.exists(stop_path):
                try:
                    os.remove(stop_path)
                except OSError:
                    pass
                try:
                    profiler.stop()
                finally:
                    profiler = None
                snap = snapshot_counters()
                _emit(f"[LMCACHE_SERVER_PROF] STOP counters={snap}")

        except Exception as exc:
            logger.exception("Server profiler loop error: %s", exc)
            _emit(f"[LMCACHE_SERVER_PROF] error: {exc}")
            profiler = None
        time.sleep(0.2)


def _make_npu_profiler(out_dir: str):
    try:
        import torch_npu
    except ImportError:
        logger.warning("torch_npu unavailable; server profiler disabled")
        return None

    try:
        experimental_config = torch_npu.profiler._ExperimentalConfig(
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
            profiler_level=torch_npu.profiler.ProfilerLevel.Level0,
            l2_cache=False,
            data_simplification=False,
            msprof_tx=True,
        )
        activities = [
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ]
        return torch_npu.profiler.profile(
            activities=activities,
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(out_dir),
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
            experimental_config=experimental_config,
        )
    except Exception as exc:
        logger.warning("Failed to create torch_npu profiler: %s", exc)
        try:
            return torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU],
                on_trace_ready=torch.profiler.tensorboard_trace_handler(out_dir),
                record_shapes=False,
                with_stack=False,
            )
        except Exception as exc2:
            logger.warning("Fallback CPU profiler failed: %s", exc2)
            return None
