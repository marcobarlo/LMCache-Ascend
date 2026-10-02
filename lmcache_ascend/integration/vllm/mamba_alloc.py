# SPDX-License-Identifier: Apache-2.0
"""Correct Ascend's extra Mamba block after an async external KV load.

``AscendMambaManager.get_num_blocks_to_allocate`` adds one block per Mamba
group when a request has external tokens and still has new tokens to
schedule. That matches the first synchronous load, where
``allocate_new_computed_blocks`` really does allocate one more block.

An async load allocates those blocks on the way into
``WAITING_FOR_REMOTE_KVS``. The resume step still satisfies the same token
check, so Ascend adds the block again. On a small block pool the
full-sequence admission gate then fails on every schedule and the request
never runs. Skip that extra block once the request already holds Mamba
blocks. Same condition as vllm-ascend PR 15068.
"""

# Future
from __future__ import annotations

# Standard
import inspect
import logging
from typing import Any

logger = logging.getLogger(__name__)

_GUARD_ATTR = "_lmcache_ascend_mamba_ext_guard"


def _already_guarded(method: Any) -> bool:
    """Return True when this method already skips the extra block."""
    if getattr(method, _GUARD_ATTR, False):
        return True
    try:
        source = inspect.getsource(method)
    except (OSError, TypeError):
        return False
    return "has_existing_blocks" in source


def _install_on(cls: type) -> None:
    """Wrap one manager class. No-op unless it is Ascend's override."""
    if cls.__name__ != "AscendMambaManager":
        return
    original = cls.get_num_blocks_to_allocate
    if _already_guarded(original):
        return
    signature = inspect.signature(original)

    def get_num_blocks_to_allocate(self: Any, *args: Any, **kwargs: Any) -> int:
        """Drop Ascend's extra block when this request already holds blocks."""
        num_new_blocks = original(self, *args, **kwargs)
        bound = signature.bind(self, *args, **kwargs)
        bound.apply_defaults()
        request_id = bound.arguments["request_id"]
        if not self.req_to_blocks.get(request_id):
            return num_new_blocks
        new_computed_blocks = bound.arguments["new_computed_blocks"]
        total_computed_tokens = bound.arguments["total_computed_tokens"]
        num_tokens_main_model = bound.arguments.get("num_tokens_main_model")
        if num_tokens_main_model is None:
            return num_new_blocks
        block_size = self.block_size
        ascend_added_one = (
            total_computed_tokens > len(new_computed_blocks) * block_size
            and num_tokens_main_model > total_computed_tokens
        )
        if ascend_added_one and num_new_blocks > 0:
            return num_new_blocks - 1
        return num_new_blocks

    setattr(get_num_blocks_to_allocate, _GUARD_ATTR, True)
    cls.get_num_blocks_to_allocate = get_num_blocks_to_allocate  # type: ignore[method-assign]
    logger.info(
        "Ascend Mamba allocation: skip the extra external block once the "
        "request already holds blocks"
    )


def install_mamba_external_block_guard() -> None:
    """Install the resume-step guard on Ascend's Mamba manager, if present.

    vllm-ascend replaces ``MambaManager`` at import. Patch both the class
    object in ``single_type_kv_cache_manager`` and the class defined in
    ``patch_mamba_manager`` so the guard holds whichever one the scheduler
    instantiated.

    Returns:
        None. Missing vLLM or a not-yet-imported Ascend patch is ignored.
    """
    try:
        import vllm.v1.core.single_type_kv_cache_manager as manager_mod
    except ImportError:
        return
    _install_on(manager_mod.MambaManager)
    try:
        from vllm_ascend.patch.platform.patch_mamba_manager import (
            AscendMambaManager,
        )
    except ImportError:
        return
    _install_on(AscendMambaManager)
