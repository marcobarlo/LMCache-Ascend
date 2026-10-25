# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import Optional

# Third Party
from lmcache.inprocess.v1.gpu_connector.gpu_connectors import GPUConnectorInterface
from lmcache.logging import init_logger
from lmcache.utils import EngineType
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.gpu_connector.utils import LayoutHints, need_gpu_interm_buffer
from lmcache.v1.metadata import LMCacheMetadata
import torch

# First Party
from lmcache_ascend.v1.npu_connector.npu_connectors import (
    VLLMBufferLayerwiseNPUConnector,
    VLLMPagedMemLayerwiseNPUConnector,
    VLLMPagedMemNPUConnectorV2,
)

logger = init_logger(__name__)


def CreateNPUConnector(
    config: LMCacheEngineConfig,
    metadata: LMCacheMetadata,
    engine: EngineType,
    layout_hints: Optional[LayoutHints] = None,
) -> GPUConnectorInterface:
    """Factory function to create NPU connectors on Ascend.

    Replaces upstream CreateGPUConnector to return Ascend NPU-specific
    connector implementations.
    """
    use_gpu = need_gpu_interm_buffer(config)

    num_gpus = torch.npu.device_count()
    local_rank = metadata.worker_id % num_gpus
    torch.npu.set_device(local_rank)
    device = torch.device(f"npu:{local_rank}")

    if engine == EngineType.VLLM:
        if metadata.use_mla and config.use_layerwise and config.enable_blending:
            raise ValueError(
                "We haven't supported MLA with Cacheblend yet. Please disable blending."
            )

        if config.use_layerwise:
            if config.enable_blending:
                conn = VLLMBufferLayerwiseNPUConnector.from_metadata(
                    metadata, use_gpu, device, layout_hints=layout_hints
                )
            else:
                conn = VLLMPagedMemLayerwiseNPUConnector.from_metadata(
                    metadata, use_gpu, device, layout_hints=layout_hints
                )
            return conn

        if config.use_gpu_connector_v3:
            raise NotImplementedError(
                "GPU Connector v3 is not supported yet. Please contact LMCache-Ascend."
            )
        else:
            conn = VLLMPagedMemNPUConnectorV2.from_metadata(
                metadata, use_gpu, device, layout_hints=layout_hints
            )
            return conn
    elif engine == EngineType.SGLANG:
        raise NotImplementedError(
            "SGLang NPU connectors are unsupported on LMCache dev: "
            "SGLangGPUConnector was removed upstream."
        )
    else:
        raise RuntimeError(f"Unsupported engine type for Ascend: {engine}")
