# SPDX-License-Identifier: Apache-2.0
"""vLLM-Ascend compress_ratio overlay for LMCache-MP group accounting.

vLLM-CUDA's ``kv_cache_spec.block_size`` is *logical* tokens per block;
LMCache derives compression as ``tokens_per_block / slots_per_block``.
vLLM-Ascend reports ``block_size`` as the *physical* slot count and keeps
compression in ``spec.compress_ratio``. Core LMCache never reads that
field, so every DSv4 group looks uncompressed (ratio 1) and
``GetStoreMetadata`` under-counts allocated tokens.

This module multiplies ``get_tokens_per_block`` by the spec's
``compress_ratio`` so the MP connector sees logical tokens per block,
matching the upstream hybrid-group contract.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any
import math

from lmcache.logging import init_logger

logger = init_logger(__name__)

_PATCHED = False


def iter_layer_specs(group_spec: Any) -> Iterator[Any]:
    """Yield per-layer specs, unwrapping ``UniformTypeKVCacheSpecs`` bundles.

    Args:
        group_spec: A vLLM KV cache spec, or a ``UniformTypeKVCacheSpecs``
            whose ``kv_cache_specs`` maps layer name to leaf spec.

    Yields:
        Leaf specs. An empty bundle yields nothing.
    """
    if type(group_spec).__name__ == "UniformTypeKVCacheSpecs":
        specs = getattr(group_spec, "kv_cache_specs", None) or {}
        yield from specs.values()
    else:
        yield group_spec


def group_compress_ratio(group_spec: Any) -> int:
    """Return the max ``compress_ratio`` across specs in one scheduler group.

    Specs without the attribute (CUDA / uncompressed) contribute 1.

    Args:
        group_spec: A vLLM KV cache spec or ``UniformTypeKVCacheSpecs`` bundle.

    Returns:
        Integer ratio >= 1.
    """
    ratios = [
        int(getattr(spec, "compress_ratio", 1) or 1)
        for spec in iter_layer_specs(group_spec)
    ]
    return max(ratios) if ratios else 1


def group_sliding_window(group_spec: Any) -> int | None:
    """Return the first non-None ``sliding_window`` in a scheduler group.

    Args:
        group_spec: A vLLM KV cache spec or ``UniformTypeKVCacheSpecs`` bundle.

    Returns:
        Window size in tokens, or ``None`` if the group is full-attention.
    """
    for spec in iter_layer_specs(group_spec):
        sw = getattr(spec, "sliding_window", None)
        if sw is not None:
            return int(sw)
    return None


def logical_tokens_per_block(
    kv_cache_spec: Any,
    dcp_size: int,
    get_tokens_per_block: Any,
) -> int:
    """Logical tokens covered by one block ID, including Ascend compression.

    Args:
        kv_cache_spec: Engine KV cache spec.
        dcp_size: Decode-context parallel size forwarded to the core helper.
        get_tokens_per_block: Unpatched core ``get_tokens_per_block``.

    Returns:
        ``core_tokens_per_block * compress_ratio``.
    """
    return get_tokens_per_block(kv_cache_spec, dcp_size) * group_compress_ratio(
        kv_cache_spec
    )


def _probe_page_dim1(value: Any) -> int | None:
    """Physical slots per block from a registered KV tensor or nested list."""
    shape = getattr(value, "shape", None)
    if shape is not None and len(shape) > 1:
        return int(shape[1])
    if isinstance(value, (list, tuple)) and value:
        return _probe_page_dim1(value[0])
    return None


def _dump_engine_group_specs(kv_cache_config: Any) -> None:
    """One-shot INFO dump of each vLLM engine group's spec fields."""
    groups = getattr(kv_cache_config, "kv_cache_groups", None) or ()
    if not groups:
        return
    for idx, group in enumerate(groups):
        spec = getattr(group, "kv_cache_spec", None)
        inners = list(iter_layer_specs(spec))
        layer_names = getattr(group, "layer_names", ())
        logger.info(
            "[ascend-mp] engine-group %d spec=%s n_layers=%d block_size=%s "
            "compress_ratio=%s sliding_window=%s inner=%s",
            idx,
            type(spec).__name__,
            len(layer_names),
            getattr(spec, "block_size", None),
            group_compress_ratio(spec),
            group_sliding_window(spec),
            [
                (
                    type(inner).__name__,
                    getattr(inner, "block_size", None),
                    int(getattr(inner, "compress_ratio", 1) or 1),
                    getattr(inner, "sliding_window", None),
                )
                for inner in inners
            ],
        )


def _dump_registered_page_dim1(kv_caches: Any) -> None:
    """Dump unique registered-page dim-1 (physical slots per block)."""
    if not isinstance(kv_caches, dict):
        return
    seen: dict[tuple[Any, ...], str] = {}
    for name, value in kv_caches.items():
        dim1 = _probe_page_dim1(value)
        tensor = value
        if isinstance(value, (list, tuple)) and value:
            tensor = value[0]
        dtype = str(getattr(tensor, "dtype", None))
        shape = tuple(getattr(tensor, "shape", ()) or ())
        key = (shape, dtype, dim1)
        if key not in seen:
            seen[key] = name
            logger.info(
                "[ascend-mp] registered page name=%s dim1=%s shape=%s dtype=%s",
                name,
                dim1,
                shape,
                dtype,
            )


def _rebind_mp_connector_helpers(kv_cache_groups: Any) -> None:
    """Rebind by-name imports on the MP connector if that module is complete.

    Do not import the connector here: plugin activation can run while
    ``lmcache_mp_connector`` is still loading (``from lmcache.banner``), and
    a re-import would see a partial module. A from-import that has not run
    yet picks up the patched source helpers automatically.
    """
    # Standard
    import sys

    mp_mod = sys.modules.get("lmcache.integration.vllm.lmcache_mp_connector")
    if mp_mod is None:
        return
    if hasattr(mp_mod, "get_tokens_per_block"):
        mp_mod.get_tokens_per_block = kv_cache_groups.get_tokens_per_block
    if hasattr(mp_mod, "create_engine_group_infos_from_vllm"):
        mp_mod.create_engine_group_infos_from_vllm = (
            kv_cache_groups.create_engine_group_infos_from_vllm
        )


def apply_get_tokens_per_block_patch() -> None:
    """Rebind ``get_tokens_per_block`` on core and MP-connector namespaces.

    Idempotent. Specs without ``compress_ratio`` keep core behaviour.
    """
    global _PATCHED
    if _PATCHED:
        return

    # Third Party
    import lmcache.integration.vllm.kv_cache_groups as kv_cache_groups

    orig = kv_cache_groups.get_tokens_per_block

    def _ascend_get_tokens_per_block(kv_cache_spec: Any, dcp_size: int = 1) -> int:
        core = orig(kv_cache_spec, dcp_size)
        logical = core * group_compress_ratio(kv_cache_spec)
        logger.info(
            "[ascend-mp] tokens_per_block spec=%s block_size=%s "
            "compress_ratio=%s sliding_window=%s core=%s logical=%s",
            type(kv_cache_spec).__name__,
            getattr(kv_cache_spec, "block_size", None),
            group_compress_ratio(kv_cache_spec),
            group_sliding_window(kv_cache_spec),
            core,
            logical,
        )
        return logical

    kv_cache_groups.get_tokens_per_block = _ascend_get_tokens_per_block  # type: ignore[assignment]

    orig_create = kv_cache_groups.create_engine_group_infos_from_vllm

    def _ascend_create_engine_group_infos_from_vllm(
        kv_cache_config: Any,
        kv_caches: Any,
        layout_hints: Any = None,
        dcp_size: int = 1,
    ) -> Any:
        _dump_engine_group_specs(kv_cache_config)
        _dump_registered_page_dim1(kv_caches)
        infos = orig_create(
            kv_cache_config,
            kv_caches,
            layout_hints=layout_hints,
            dcp_size=dcp_size,
        )
        tpb = [info.tokens_per_block for info in infos]
        positive = [t for t in tpb if t > 0]
        required = math.lcm(*positive) if positive else 0
        logger.info(
            "[ascend-mp] engine_group_infos tpb=%s sw=%s "
            "required_chunk_multiple=%s",
            tpb,
            [info.sw_size_tokens for info in infos],
            required,
        )
        return infos

    kv_cache_groups.create_engine_group_infos_from_vllm = (  # type: ignore[assignment]
        _ascend_create_engine_group_infos_from_vllm
    )
    _rebind_mp_connector_helpers(kv_cache_groups)
    _PATCHED = True
    logger.info("Patched get_tokens_per_block to include spec.compress_ratio")
