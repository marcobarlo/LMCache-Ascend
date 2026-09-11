# SPDX-License-Identifier: Apache-2.0
"""Expose Ascend physical-page token span as ``logical_block_size``.

vLLM-Ascend stores ``block_size`` in physical slots; the token span is
``block_size * compress_ratio``. Upstream vLLM already uses logical tokens in
``block_size``, so LMCache core only does
``getattr(spec, "logical_block_size", spec.block_size)``.

This module attaches that property at runtime (no vLLM-Ascend source edits):
leaf Ascend MLA/SWA specs, ``UniformTypeKVCacheSpecs`` (max of leaves), and a
``merge()`` wrap so dropped ``compress_ratio`` is restored.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any
import sys

_INSTALLED = False
_GET_TOKENS_WRAPPED = False


def _leaf_logical_block_size(self: Any) -> int:
    return int(self.block_size) * int(getattr(self, "compress_ratio", 1) or 1)


def _uniform_logical_block_size(self: Any) -> int:
    inner = getattr(self, "kv_cache_specs", None) or {}
    if not inner:
        return int(self.block_size)
    return max(
        int(getattr(spec, "logical_block_size", spec.block_size))
        for spec in inner.values()
    )


def _has_logical_block_size_property(cls: type) -> bool:
    return isinstance(getattr(cls, "logical_block_size", None), property)


def _restore_dropped_merge_fields(merged: Any, specs: list[Any]) -> Any:
    first = specs[0]
    for name in ("compress_ratio", "model_version"):
        expected = getattr(first, name, None)
        if expected is None:
            continue
        if getattr(merged, name, None) != expected:
            object.__setattr__(merged, name, expected)
    return merged


def _wrap_merge(cls: type) -> None:
    orig = cls.merge
    orig_fn = getattr(orig, "__func__", orig)
    if getattr(orig_fn, "_lmcache_ascend_logical_block_size", False):
        return

    def merge(cls_: type, specs: list[Any]) -> Any:
        merged = orig_fn(cls_, specs)
        return _restore_dropped_merge_fields(merged, specs)

    merge._lmcache_ascend_logical_block_size = True  # type: ignore[attr-defined]
    cls.merge = classmethod(merge)


def install_on(
    leaf_classes: Iterable[type],
    uniform_type_cls: type | None = None,
) -> None:
    """Attach ``logical_block_size`` to the given spec classes (idempotent)."""
    for cls in leaf_classes:
        if not _has_logical_block_size_property(cls):
            cls.logical_block_size = property(_leaf_logical_block_size)
        if hasattr(cls, "merge"):
            _wrap_merge(cls)
    if uniform_type_cls is not None and not _has_logical_block_size_property(
        uniform_type_cls
    ):
        uniform_type_cls.logical_block_size = property(_uniform_logical_block_size)


def install_logical_block_size() -> bool:
    """Patch real vLLM-Ascend / vLLM spec classes if they are importable."""
    try:
        from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs
        from vllm_ascend.core.kv_cache_interface import (
            AscendMLAAttentionSpec,
            AscendSlidingWindowMLASpec,
        )
    except ImportError:
        return False
    install_on(
        (AscendMLAAttentionSpec, AscendSlidingWindowMLASpec),
        UniformTypeKVCacheSpecs,
    )
    return True


def _wrap_get_tokens_per_block() -> None:
    """Ensure the spec patch runs before the first ``get_tokens_per_block``."""
    global _GET_TOKENS_WRAPPED
    if _GET_TOKENS_WRAPPED:
        return
    try:
        import lmcache.integration.vllm.kv_cache_groups as kg
    except ImportError:
        return
    orig = kg.get_tokens_per_block
    if getattr(orig, "_lmcache_ascend_logical_block_size", False):
        _GET_TOKENS_WRAPPED = True
        return

    def get_tokens_per_block(kv_cache_spec: Any, dcp_size: int) -> int:
        install_logical_block_size()
        return orig(kv_cache_spec, dcp_size)

    get_tokens_per_block._lmcache_ascend_logical_block_size = True  # type: ignore[attr-defined]
    kg.get_tokens_per_block = get_tokens_per_block
    connector = sys.modules.get("lmcache.integration.vllm.lmcache_mp_connector")
    if connector is not None and getattr(connector, "get_tokens_per_block", None) is orig:
        connector.get_tokens_per_block = get_tokens_per_block
    _GET_TOKENS_WRAPPED = True


def install_overrides() -> None:
    """Install spec properties and wrap ``get_tokens_per_block`` (idempotent)."""
    global _INSTALLED
    install_logical_block_size()
    _wrap_get_tokens_per_block()
    _INSTALLED = True
