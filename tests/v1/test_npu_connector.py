# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501
# Standard
from unittest.mock import patch

# Third Party
# TODO (gingfung): once we have sglang kernel, re-enable test_sglang_connector_with_gpu_and_mla
from lmcache_tests.v1.test_gpu_connector import (
    test_batched_layerwise_vllm_paged_connector_with_gpu as original_test_batched_layerwise_vllm_paged_connector_with_gpu,
)
from lmcache_tests.v1.test_gpu_connector import (
    test_layerwise_vllm_paged_connector_with_gpu as original_test_layerwise_vllm_paged_connector_with_gpu,
)
from lmcache_tests.v1.test_gpu_connector import (
    test_vllm_paged_connector_v2_to_gpu_bench as original_test_vllm_paged_connector_v2_to_gpu_bench,
)
import pytest

# First Party
from lmcache_ascend.v1.npu_connector.npu_connectors import (
    VLLMPagedMemLayerwiseNPUConnector,
    VLLMPagedMemNPUConnectorV2,
)
import lmcache_ascend.c_ops as lmc_ops


@pytest.mark.parametrize("use_npu", [True])
@pytest.mark.parametrize(
    "gpu_kv_format",
    [
        lmc_ops.GPUKVFormat.NL_X_TWO_NB_BS_NH_HS,  # vllm non-MLA flash attention
    ],
)
def test_layerwise_vllm_paged_connector_with_npu(use_npu, gpu_kv_format):
    target_patch = (
        "lmcache_tests.v1.test_gpu_connector.VLLMPagedMemLayerwiseGPUConnector"
    )

    with patch(target_patch, new=VLLMPagedMemLayerwiseNPUConnector):
        original_test_layerwise_vllm_paged_connector_with_gpu(use_npu, gpu_kv_format)


@pytest.mark.parametrize("use_npu", [True])
def test_batched_layerwise_vllm_paged_connector_with_npu(use_npu):
    target_patch = (
        "lmcache_tests.v1.test_gpu_connector.VLLMPagedMemLayerwiseGPUConnector"
    )

    with patch(target_patch, new=VLLMPagedMemLayerwiseNPUConnector):
        original_test_batched_layerwise_vllm_paged_connector_with_gpu(use_npu)


def test_vllm_paged_connector_v2_to_npu_bench(benchmark):
    target_patch = "lmcache_tests.v1.test_gpu_connector.VLLMPagedMemGPUConnectorV2"

    with patch(target_patch, new=VLLMPagedMemNPUConnectorV2):
        original_test_vllm_paged_connector_v2_to_gpu_bench(benchmark)
