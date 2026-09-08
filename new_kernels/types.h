/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cstdint>

namespace kvcache_ops {

constexpr int32_t kMultiPlaneLaunchMaxPlanes = 8;

// Host-side launch packet for templated multi-plane kernels (N in {1,2,4,8}).
struct MultiPlaneKernelLaunch {
  uint32_t blockDim;
  void *stream;
  uint8_t *pagedKVCaches;
  uint8_t *dstCacheTensor;
  uint8_t *slotMaps[kMultiPlaneLaunchMaxPlanes];
  int32_t *prefixes[kMultiPlaneLaunchMaxPlanes];
  int32_t hd[kMultiPlaneLaunchMaxPlanes];
  int32_t bs[kMultiPlaneLaunchMaxPlanes];
  int32_t pbs[kMultiPlaneLaunchMaxPlanes];
  int32_t lmcRowOff[kMultiPlaneLaunchMaxPlanes];
  int32_t ratio[kMultiPlaneLaunchMaxPlanes];
  int32_t numPlanes;
  int32_t gStart;
  int32_t gEnd;
  int32_t numLayers;
  int64_t lmcChunkLastDimBytes;
  int32_t numTokensLmcChunk;
  int64_t perLoopBuffer;
  int32_t maxTokensPerLoop;
  bool page2L;
};
enum struct AscendType {
    FP16 = 0,
    BF16 = 1,
    FP32 = 2,
    INT8 = 3,
    INT32 = 4,
    INT64 = 5,
};

enum struct KVCacheFormat : int {
    UNDEFINED = 0,
    MERGED_KV = 1,    // [2, num_blocks, block_size, num_heads, head_dim] eg: vllm0.9.2 
    SEPARATE_KV = 2,  // tuple(K, V), k/v: [num_blocks, block_size, num_heads, head_dim] eg: vllm0.11.0 
    MLA_KV = 3,       // tuple(k_cache, v_cache) with different hidden_dims for DeepSeek V2/V3
    DSA_KV = 4,       // tuple(k_cache, v_cache, dsa_k_cache) for DeepSeek V3.2 sparse attention
};
}