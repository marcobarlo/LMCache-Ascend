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

#ifndef MULTI_LAYER_MEM_KERNELS_V3_H
#define MULTI_LAYER_MEM_KERNELS_V3_H

#include "multi_layer_mem_kernels.h"

namespace kvcache_ops {

struct V3Config {
    StandardConfig common;
    int64_t perLoopBuffSize;  // buffer size in innerloop within UB
    int32_t maxTokensPerLoop; // num tokens per inner loop for transferring
    // 0 = tight slot-major (slot * hiddenDims). Non-zero is stride(0) of the
    // paged block axis, in the launched scalar_t units (LMCache
    // PageBufferShapeDesc.block_stride_elems, converted to the kernel dtype).
    int64_t blockStrideElems{0};
    // vLLM tokens per block; required when blockStrideElems > 0.
    int32_t blockSize{0};
    // 0 = plane-major LMC slabs (GetLMCBaseOffset). Non-zero = token-major
    // packed row width in scalar_t units; planes sit at prefix-sum offsets
    // inside each token row (torch_ops._transfer_per_layer_mla_tuple).
    int64_t lmcRowElems{0};
};

inline V3Config MakeV3Config(
    int64_t hiddenDims, int32_t numLayers, int64_t pageBuffSize,
    int32_t numTokensChunk, bool page2L, int32_t kvs,
    int64_t perLoopBuffSize, int32_t maxTokensPerLoop,
    int64_t blockStrideElems = 0, int32_t blockSize = 0,
    int64_t lmcRowElems = 0)
{
    V3Config cfg;
    cfg.common = {hiddenDims, numLayers, pageBuffSize, numTokensChunk, page2L, kvs};
    cfg.perLoopBuffSize = perLoopBuffSize;
    cfg.maxTokensPerLoop = maxTokensPerLoop;
    cfg.blockStrideElems = blockStrideElems;
    cfg.blockSize = blockSize;
    cfg.lmcRowElems = lmcRowElems;
    return cfg;
}

template<typename scalar_t, typename slot_t, KVCacheFormat fmt>
struct V3Launcher {
    static void Launch(
        uint32_t blockDim,
        void* stream,
        uint8_t* pagedKVCaches,
        uint8_t* dstCacheTensor,
        uint8_t* slotmappings,
        const V3Config& config,
        int64_t kHiddenDims = 0,
        int64_t vHiddenDims = 0,
        int64_t dsaHiddenDims = 0);
};

void multi_layer_kv_transfer_kernel_v3(
    kvcache_ops::AscendType type, kvcache_ops::AscendType slotType,
    const kvcache_ops::KVCacheFormat kvcacheFormat, uint32_t blockDim, void *stream,
    uint8_t *pagedKVCaches, uint8_t *dstCacheTensor, uint8_t *slotmappings,
    const int64_t hiddenDims, const int32_t kvs, const int32_t numLayers,
    const int64_t pageBuffSize, const int32_t numTokensChunk,
    const int64_t perLoopBuffer, const int32_t maxTokensPerLoop,
    const bool page2L,
    const int64_t kHiddenDims = 0, const int64_t vHiddenDims = 0,
    const int64_t dsaHiddenDims = 0,
    const int64_t blockStrideElems = 0, const int32_t blockSize = 0,
    const int64_t lmcRowElems = 0);

} // namespace kvcache_ops

#endif // MULTI_LAYER_MEM_KERNELS_V3_H
