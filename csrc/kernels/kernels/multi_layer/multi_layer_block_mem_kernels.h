// SPDX-License-Identifier: Apache-2.0

#ifndef MULTI_LAYER_BLOCK_MEM_KERNELS_H
#define MULTI_LAYER_BLOCK_MEM_KERNELS_H

#include "block_transfer_layout.h"

namespace kvcache_ops {

// Host-side launcher for the block-level multi-layer KV transfer kernel.
// One launch moves one LMCache memory object
//
// Work distribution is driven by separate_plane on the launch (see
// block_transfer_contract.md). Pointer table layout:
//   one_pointer_per_layer: ptrs[layer], plane GM at base +
//   PlaneLayout::engine_base_offset_bytes (unusual when num_planes > 1).
//   default: ptrs[layer * num_planes + plane], each plane view's data_ptr().
// Enum examples: format 2 uses one pointer per layer; 16 uses per-plane
// pointers with separate_plane; 17 packs planes with separate_plane false.

void multi_layer_block_transfer_kernel(
    uint32_t block_dim,            // min(AIV cores, ACTIVE work items =
                                   //   nl * plane_slots * (blocks - skip))
    void* stream,                  // aclrtStream (current NPU stream)
    uint8_t* paged_buffer_ptrs,    // interleaved pointer table (see above)
    uint8_t* lmcache_obj,          // LMC object device VA (maybe staged)
    uint8_t* engine_block_ids,     // int64 array: logical -> engine block
    int32_t num_blocks_per_object, // == lmcache_chunk_size / bs (host-checked)
    int32_t skip_prefix_n_blocks,  // leading blocks to skip (prefix dedup);
                                   //   == blocks -> no launch at all
    int32_t nl,                    // layers in this group
    int32_t nb,                    // engine pool block count (ID upper bound)
    int32_t bs,                    // tokens/slots per block
    bool separate_plane,           // true: (layer, plane, block) work items
    BlockTransferLayout layout,    // full byte layout (block_transfer_layout.h)
    bool to_engine,                // H2D = LMC->engine (retrieve), D2H =
                                   // engine->LMC (store); device-to-device
    int64_t ub_bytes = 0,          // usable UB staging budget, host-queried
                                   //   via GetCoreMemSize(CoreMemType::UB);
                                   //   0 = built-in default
    bool one_pointer_per_layer = false); // see pointer table note above
} // namespace kvcache_ops

#endif // MULTI_LAYER_BLOCK_MEM_KERNELS_H
