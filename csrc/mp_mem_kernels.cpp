// SPDX-License-Identifier: Apache-2.0

// C++ host layer for the block-level MP-mode KV transfer (mirrors upstream
// LMCache csrc/mp_mem_kernels.cu).

#include "mp_mem_kernels.h"

#include "kernels/multi_layer/multi_layer_block_mem_kernels.h"
#include "utils.h"

#include <acl/acl.h>

#include <algorithm>
#include <cstdint>

namespace {

// DataCopy granularity on the GM side: segment base addresses and segment
// byte lengths must both be multiples of 32B.
constexpr int64_t kGmAlignBytes = 32;
// UB budget assumed by the device kernel: the depth-2 queue must fit two
// token segments, so a single token's bytes may not exceed half of it.
constexpr int64_t kUbBudgetBytes = 128 * 1024;

// The kernel is a pure byte mover, so fp16 and bf16 share the 2-byte
// instantiation (mirrors the upstream uint16/uint32/uint4 granularity
// dispatch in multi_layer_block_kv_transfer).
kvcache_ops::AscendType ascend_type_from_element_size(int element_size) {
  switch (element_size) {
    case 1:
      return kvcache_ops::AscendType::INT8;
    case 2:
      return kvcache_ops::AscendType::FP16;
    case 4:
      return kvcache_ops::AscendType::FP32;
    default:
      TORCH_CHECK(false, "Unsupported element_size: ", element_size,
                  " (expected 1, 2 or 4)");
  }
  return kvcache_ops::AscendType::FP16;  // unreachable
}

struct PackedPlaneSpec {
  bool packed = false;
  int32_t k_plane_elems = 0;
  int32_t v_plane_elems = 0;
  int32_t lmc_row_elems = 0;
};

PackedPlaneSpec packed_planes(const PageBufferShapeDesc& shape_desc,
                              EngineKVFormat engine_kv_format) {
  PackedPlaneSpec spec;
  if (engine_kv_format == EngineKVFormat::NL_X_TWO_X_NB_BS_HS) {
    spec.packed = true;
    spec.lmc_row_elems = shape_desc.hs * shape_desc.element_size;
    spec.v_plane_elems = 2;
    spec.k_plane_elems = spec.lmc_row_elems - 2;
  }
  return spec;
}

kvcache_ops::AscendType launch_type(const PageBufferShapeDesc& shape_desc,
                                    const PackedPlaneSpec& spec) {
  if (spec.packed) {
    return kvcache_ops::AscendType::INT8;
  }
  return ascend_type_from_element_size(shape_desc.element_size);
}

// Shared validation for both entry points. All TORCH_CHECKs fire before any
// stream work is enqueued; returns blocks per object.
int validate_block_transfer(const PageBufferShapeDesc& shape_desc,
                            int64_t total_blocks, int num_objects,
                            int lmcache_chunk_size,
                            EngineKVFormat engine_kv_format,
                            int skip_prefix_n_blocks) {
  const bool separate =
      engine_kv_format == EngineKVFormat::NL_X_TWO_X_NB_BS_NH_HS;
  const bool packed =
      engine_kv_format == EngineKVFormat::NL_X_TWO_X_NB_BS_HS;
  const bool fused =
      engine_kv_format == EngineKVFormat::NL_X_NB_BS_NH_CS;
  TORCH_CHECK(separate || packed || fused,
              "LMCache-Ascend block-level MP transfer currently supports "
              "NL_X_TWO_X_NB_BS_NH_HS (16), NL_X_TWO_X_NB_BS_HS (17), and "
              "NL_X_NB_BS_NH_CS (13), got ",
              static_cast<int>(engine_kv_format));
  if (separate) {
    TORCH_CHECK(shape_desc.kv_size == 2, "SEPARATE_KV requires kv_size == 2, ",
                "got ", shape_desc.kv_size);
  } else if (fused) {
    TORCH_CHECK(shape_desc.kv_size == 1, "NH_CS fused requires kv_size == 1, ",
                "got ", shape_desc.kv_size);
  } else {
    TORCH_CHECK(shape_desc.kv_size == 1 || shape_desc.kv_size == 2,
                "packed MLA requires kv_size 1 or 2, got ", shape_desc.kv_size);
  }
  TORCH_CHECK(skip_prefix_n_blocks >= 0, "skip_prefix_n_blocks must be >= 0, ",
              "got ", skip_prefix_n_blocks);

  TORCH_CHECK(num_objects >= 1 && num_objects <= 4,
              "Expected 1-4 LMCache objects, got ", num_objects);
  TORCH_CHECK(total_blocks % num_objects == 0, "block_ids length (",
              total_blocks, ") must be divisible by num_objects (",
              num_objects, ")");
  const int num_blocks_per_object =
      static_cast<int>(total_blocks / num_objects);

  TORCH_CHECK(num_blocks_per_object * shape_desc.bs == lmcache_chunk_size,
              "blocks_per_object * block_size (",
              num_blocks_per_object * shape_desc.bs,
              ") must equal lmcache_chunk_size (", lmcache_chunk_size, ")");
  TORCH_CHECK(skip_prefix_n_blocks <= num_blocks_per_object,
              "skip_prefix_n_blocks (", skip_prefix_n_blocks,
              ") cannot exceed blocks per object (", num_blocks_per_object,
              ")");

  const PackedPlaneSpec spec = packed_planes(shape_desc, engine_kv_format);
  const int64_t engine_block_stride =
      shape_desc.block_stride_elems > 0
          ? static_cast<int64_t>(shape_desc.block_stride_elems)
          : static_cast<int64_t>(shape_desc.bs) * shape_desc.nh *
                shape_desc.hs;
  const int64_t stride_bytes =
      spec.packed ? engine_block_stride
                  : engine_block_stride * shape_desc.element_size;
  TORCH_CHECK(stride_bytes % kGmAlignBytes == 0,
              "engine block stride (", stride_bytes,
              " bytes) must be a multiple of ", kGmAlignBytes,
              " for DataCopy alignment");

  int64_t ub_token_bytes;
  if (spec.packed) {
    TORCH_CHECK(spec.lmc_row_elems > 2, "packed LMC row must exceed 2 B, got ",
                spec.lmc_row_elems);
    TORCH_CHECK(spec.k_plane_elems > 0 &&
                    spec.k_plane_elems % kGmAlignBytes == 0,
                "packed latent width (", spec.k_plane_elems,
                " bytes) must be a positive multiple of ", kGmAlignBytes);
    const int64_t latent_bytes = spec.k_plane_elems;
    const int64_t scale_aligned = (2 + 31) & ~31;
    ub_token_bytes =
        latent_bytes > scale_aligned ? latent_bytes : scale_aligned;
  } else {
    const int64_t token_bytes = static_cast<int64_t>(shape_desc.nh) *
                                shape_desc.hs * shape_desc.element_size;
    TORCH_CHECK(token_bytes > 0, "nh * hs * element_size must be positive");
    TORCH_CHECK(token_bytes % kGmAlignBytes == 0,
                "scalars_per_token * element_size (", token_bytes,
                " bytes) must be a multiple of ", kGmAlignBytes,
                " for DataCopy alignment");
    ub_token_bytes = token_bytes;
  }

  TORCH_CHECK(ub_token_bytes <= kUbBudgetBytes / 2, "token bytes (",
              ub_token_bytes, ") exceed the per-segment UB budget (",
              kUbBudgetBytes / 2, "); token-level segmentation cannot fit");

  return num_blocks_per_object;
}

// CUDA UVA lets the kernel consume host data_ptr()s. Ascend DataCopy needs a
// device VA: registered aclrtMallocHost (get_device_ptr) or a true NPU
// allocation. MixedMemoryAllocator SHM L1 is neither — stage on-device and
// aclrtMemcpy.
bool is_npu_memory_ptr(void* ptr) {
  aclrtPtrAttributes attributes{};
  const aclError ret = aclrtPointerGetAttributes(ptr, &attributes);
  if (ret != ACL_ERROR_NONE) {
    return false;
  }
  return attributes.location.type == ACL_MEM_LOCATION_TYPE_DEVICE;
}

std::vector<int64_t> device_lmc_ptrs(const std::vector<int64_t>& ptrs) {
  std::vector<int64_t> out;
  out.reserve(ptrs.size());
  for (int64_t p : ptrs) {
    void* mapped =
        get_device_ptr(reinterpret_cast<void*>(static_cast<uintptr_t>(p)));
    out.push_back(mapped != nullptr ? reinterpret_cast<int64_t>(mapped) : p);
  }
  return out;
}

struct PackedHostStage {
  torch::Tensor buf;
  int64_t host_ptr = 0;
  size_t nbytes = 0;
};

// Resolve LMCache object pointers to device VAs. Packed SHM/host objects
// that are neither registered aclrtMallocHost nor NPU memory get an on-device
// staging tensor (PackedHostStage); the caller memcpy-asyncs around the kernel.
struct PreparedLmcPtrs {
  std::vector<int64_t> kernel_obj_ptrs;
  std::vector<PackedHostStage> host_stages;
};

PreparedLmcPtrs prepare_lmc_ptrs(const std::vector<int64_t>& lmcache_objects_ptrs,
                                 const torch::Device& device,
                                 const PageBufferShapeDesc& shape_desc,
                                 int lmcache_chunk_size,
                                 const PackedPlaneSpec& spec) {
  PreparedLmcPtrs prepared;
  prepared.kernel_obj_ptrs.reserve(lmcache_objects_ptrs.size());
  const auto staging_opts =
      torch::TensorOptions().dtype(torch::kUInt8).device(device);
  for (int64_t p : lmcache_objects_ptrs) {
    void* raw = reinterpret_cast<void*>(static_cast<uintptr_t>(p));
    void* mapped = get_device_ptr(raw);
    if (mapped != nullptr) {
      prepared.kernel_obj_ptrs.push_back(reinterpret_cast<int64_t>(mapped));
      continue;
    }
    if (!spec.packed || is_npu_memory_ptr(raw)) {
      prepared.kernel_obj_ptrs.push_back(p);
      continue;
    }
    const int64_t nbytes = static_cast<int64_t>(shape_desc.nl) *
                           lmcache_chunk_size * spec.lmc_row_elems;
    PackedHostStage stage;
    stage.buf = torch::empty(
        {shape_desc.nl, lmcache_chunk_size, spec.lmc_row_elems}, staging_opts);
    stage.host_ptr = p;
    stage.nbytes = static_cast<size_t>(nbytes);
    prepared.kernel_obj_ptrs.push_back(
        reinterpret_cast<int64_t>(stage.buf.data_ptr()));
    prepared.host_stages.push_back(std::move(stage));
  }
  return prepared;
}

void launch_block_transfer_objects(
    kvcache_ops::AscendType type, uint32_t aiv_num, void* stream,
    uint8_t* paged_buffer_ptrs,
    const std::vector<int64_t>& lmcache_objects_ptrs, int64_t* block_ids_base,
    int64_t total_blocks, int num_blocks_per_object,
    const PageBufferShapeDesc& shape_desc, int lmcache_chunk_size,
    int skip_prefix_n_blocks, bool lmcache_to_engine,
    const PackedPlaneSpec& spec);

// Enqueue PackedHostStage memcpys + the block kernel(s) on ``stream``.
// No OpCommand: the caller wraps this in one Run() (direct API) or folds it
// into the object-group plan's single Run(). Returns 0 or an ACL error.
int enqueue_block_transfer(void* stream, uint32_t aiv_num,
                           uint8_t* paged_buffer_ptrs,
                           const PreparedLmcPtrs& prepared,
                           int64_t* block_ids_base, int64_t total_blocks,
                           int num_blocks_per_object,
                           const PageBufferShapeDesc& shape_desc,
                           int lmcache_chunk_size, int skip_prefix_n_blocks,
                           bool lmcache_to_engine, kvcache_ops::AscendType type,
                           const PackedPlaneSpec& spec) {
  if (lmcache_to_engine) {
    for (const auto& stage : prepared.host_stages) {
      const aclError ret = aclrtMemcpyAsync(
          stage.buf.data_ptr(), stage.nbytes,
          reinterpret_cast<const void*>(static_cast<uintptr_t>(stage.host_ptr)),
          stage.nbytes, ACL_MEMCPY_HOST_TO_DEVICE, stream);
      if (ret != ACL_ERROR_NONE) {
        return static_cast<int>(ret);
      }
    }
  }
  launch_block_transfer_objects(
      type, aiv_num, stream, paged_buffer_ptrs, prepared.kernel_obj_ptrs,
      block_ids_base, total_blocks, num_blocks_per_object, shape_desc,
      lmcache_chunk_size, skip_prefix_n_blocks, lmcache_to_engine, spec);
  if (!lmcache_to_engine) {
    for (const auto& stage : prepared.host_stages) {
      const aclError ret = aclrtMemcpyAsync(
          reinterpret_cast<void*>(static_cast<uintptr_t>(stage.host_ptr)),
          stage.nbytes, stage.buf.data_ptr(), stage.nbytes,
          ACL_MEMCPY_DEVICE_TO_HOST, stream);
      if (ret != ACL_ERROR_NONE) {
        return static_cast<int>(ret);
      }
    }
  }
  return 0;
}

// Phase-1 launch loop: one object + one block_ids slice per kernel launch
// (design doc 4.5). blockDim is clamped to the work-item count so tiny
// transfers do not spin idle cores.
void launch_block_transfer_objects(
    kvcache_ops::AscendType type, uint32_t aiv_num, void* stream,
    uint8_t* paged_buffer_ptrs,
    const std::vector<int64_t>& lmcache_objects_ptrs, int64_t* block_ids_base,
    int64_t total_blocks, int num_blocks_per_object,
    const PageBufferShapeDesc& shape_desc, int lmcache_chunk_size,
    int skip_prefix_n_blocks, bool lmcache_to_engine,
    const PackedPlaneSpec& spec) {
  // Packed MLA always has two pointer-table planes (latent+scale) even when
  // shape_desc.kv_size == 1. Fused NH_CS uses kv_size == 1 (one ptr/layer).
  const int32_t kv_size = spec.packed ? 2 : shape_desc.kv_size;
  const int64_t total_work =
      static_cast<int64_t>(shape_desc.nl) * kv_size * total_blocks;
  const uint32_t blockDim =
      static_cast<uint32_t>(std::min<int64_t>(aiv_num, total_work));
  for (int i = 0; i < static_cast<int>(lmcache_objects_ptrs.size()); ++i) {
    uint8_t* engine_block_ids = reinterpret_cast<uint8_t*>(
        block_ids_base + static_cast<int64_t>(i) * num_blocks_per_object);
    kvcache_ops::multi_layer_block_transfer_kernel(
        type, blockDim, stream, paged_buffer_ptrs,
        reinterpret_cast<uint8_t*>(lmcache_objects_ptrs[i]),
        engine_block_ids, num_blocks_per_object, skip_prefix_n_blocks,
        shape_desc.nl, shape_desc.bs, shape_desc.nh, shape_desc.hs,
        shape_desc.block_stride_elems, lmcache_chunk_size, lmcache_to_engine,
        spec.k_plane_elems, spec.v_plane_elems, spec.lmc_row_elems, kv_size);
  }
}

}  // namespace

void multi_layer_block_kv_transfer(
    const torch::Tensor& paged_buffer_ptrs_tensor,
    std::vector<int64_t> lmcache_objects_ptrs, const torch::Tensor& block_ids,
    const torch::Device& device, TransferDirection direction,
    PageBufferShapeDesc shape_desc, int lmcache_chunk_size,
    EngineKVFormat engine_kv_format, int skip_prefix_n_blocks) {
  // --- Validation ---
  const int num_objects = static_cast<int>(lmcache_objects_ptrs.size());
  const int64_t total_blocks = block_ids.size(0);
  const int num_blocks_per_object = validate_block_transfer(
      shape_desc, total_blocks, num_objects, lmcache_chunk_size,
      engine_kv_format, skip_prefix_n_blocks);

  TORCH_CHECK(paged_buffer_ptrs_tensor.scalar_type() == at::kLong,
              "paged_buffer_ptrs_tensor must be int64");
  TORCH_CHECK(paged_buffer_ptrs_tensor.is_privateuseone(),
              "paged_buffer_ptrs_tensor must live on the NPU");
  TORCH_CHECK(paged_buffer_ptrs_tensor.dim() == 1,
              "paged_buffer_ptrs_tensor must be one-dimensional");
  const PackedPlaneSpec spec =
      packed_planes(shape_desc, engine_kv_format);
  const int64_t expected_ptrs =
      static_cast<int64_t>(spec.packed ? 2 : shape_desc.kv_size) *
      shape_desc.nl;
  TORCH_CHECK(paged_buffer_ptrs_tensor.numel() == expected_ptrs,
              "paged_buffer_ptrs_tensor must contain kv_size * nl pointers: "
              "expected ", expected_ptrs, ", got ",
              paged_buffer_ptrs_tensor.numel());
  TORCH_CHECK(paged_buffer_ptrs_tensor.is_contiguous(),
              "paged_buffer_ptrs_tensor must be contiguous");
  TORCH_CHECK(block_ids.is_privateuseone(), "block_ids must live on the NPU");
  TORCH_CHECK(block_ids.scalar_type() == at::kLong,
              "block_ids must have dtype int64");
  TORCH_CHECK(block_ids.dim() == 1, "block_ids must be one-dimensional");
  TORCH_CHECK(block_ids.is_contiguous(), "block_ids must be contiguous");

  const kvcache_ops::AscendType type = launch_type(shape_desc, spec);
  const bool lmcache_to_engine = (direction == TransferDirection::H2D);

  PreparedLmcPtrs prepared = prepare_lmc_ptrs(
      lmcache_objects_ptrs, device, shape_desc, lmcache_chunk_size, spec);

  uint8_t* paged_buffer_ptrs =
      static_cast<uint8_t*>(paged_buffer_ptrs_tensor.data_ptr());
  int64_t* block_ids_base = block_ids.data_ptr<int64_t>();

  const c10::OptionalDeviceGuard device_guard(device);
  aclrtStream stream = c10_npu::getCurrentNPUStream().stream();

  at_npu::native::OpCommand cmd;
  cmd.Name("multi_layer_block_transfer_kernel");
  cmd.SetCustomHandler([type, stream, paged_buffer_ptrs, prepared,
                        block_ids_base, total_blocks, num_blocks_per_object,
                        shape_desc, lmcache_chunk_size, skip_prefix_n_blocks,
                        lmcache_to_engine, spec]() -> int {
    const char* socName = aclrtGetSocName();
    auto ascendcPlatform =
        platform_ascendc::PlatformAscendCManager::GetInstance(socName);
    const uint32_t aiv_num = ascendcPlatform->GetCoreNumAiv();
    return enqueue_block_transfer(
        stream, aiv_num, paged_buffer_ptrs, prepared, block_ids_base,
        total_blocks, num_blocks_per_object, shape_desc, lmcache_chunk_size,
        skip_prefix_n_blocks, lmcache_to_engine, type, spec);
  });
  cmd.Run();
}

void lmcache_memcpy_async_on_stream(uintptr_t dest, uintptr_t src, size_t nbytes,
                                    TransferDirection direction,
                                    size_t host_buffer_offset,
                                    size_t host_buffer_alignments,
                                    aclrtStream stream);

void execute_object_group_transfer(
    TransferDirection direction, const torch::Device& device,
    size_t host_buffer_alignment,
    const std::vector<KernelGroupSpec>& kernel_group_specs,
    const std::vector<BatchStep>& batch_steps) {
  // Set the device guard once for the whole plan so every staging copy and
  // kernel launch below is enqueued on this device's current stream, in
  // order (mirrors upstream execute_object_group_transfer).
  const c10::OptionalDeviceGuard device_guard(device);
  const bool is_h2d = (direction == TransferDirection::H2D);
  TORCH_CHECK(device.is_privateuseone(), "device must be an NPU device");

  // --- Validate the whole plan up front, before any stream work ---
  // Bounds-check every launch's block_ids slice before the kernel
  // dereferences it on device: an out-of-range offset/length would
  // otherwise be a silent out-of-bounds device read, not a clean error.
  for (const auto& step : batch_steps) {
    for (const auto& launch : step.launches) {
      TORCH_CHECK(launch.group_idx >= 0 &&
                      launch.group_idx <
                          static_cast<int>(kernel_group_specs.size()),
                  "LaunchVar.group_idx out of range: ", launch.group_idx);
      const KernelGroupSpec& group = kernel_group_specs[launch.group_idx];
      TORCH_CHECK(launch.num_objects >= 1 &&
                      launch.num_objects <=
                          static_cast<int>(group.lmcache_objects_ptrs.size()),
                  "LaunchVar.num_objects (", launch.num_objects,
                  ") exceeds available temp buffers (",
                  group.lmcache_objects_ptrs.size(), ")");
      TORCH_CHECK(launch.block_ids_offset >= 0,
                  "LaunchVar.block_ids_offset must be non-negative, got ",
                  launch.block_ids_offset);
      TORCH_CHECK(launch.total_blocks >= 0,
                  "LaunchVar.total_blocks must be non-negative, got ",
                  launch.total_blocks);
      TORCH_CHECK(launch.block_ids_offset + launch.total_blocks <=
                      group.block_ids_capacity,
                  "LaunchVar block_ids slice [", launch.block_ids_offset, ", ",
                  launch.block_ids_offset + launch.total_blocks,
                  ") exceeds block_ids capacity ", group.block_ids_capacity);
      // Full per-launch validation (format, shape, alignment) through the
      // shared checker so the plan path rejects bad launches exactly like
      // the direct entry point would.
      validate_block_transfer(group.shape_desc, launch.total_blocks,
                              launch.num_objects, group.lmcache_chunk_size,
                              group.engine_kv_format,
                              launch.skip_prefix_n_blocks);
    }
  }

  const char* socName = aclrtGetSocName();
  auto ascendcPlatform =
      platform_ascendc::PlatformAscendCManager::GetInstance(socName);
  const uint32_t aiv_num = ascendcPlatform->GetCoreNumAiv();
  const aclrtStream stream = c10_npu::getCurrentNPUStream().stream();

  at_npu::native::OpCommand cmd;
  cmd.Name("multi_layer_block_transfer_kernel");
  // Capture by value like multi_layer_block_kv_transfer: OpCommand may run
  // the handler on a worker thread; [&] deadlocks torch_npu's task queue.
  cmd.SetCustomHandler([direction, is_h2d, host_buffer_alignment, aiv_num,
                        stream, kernel_group_specs, batch_steps]() -> int {
    const auto do_staging = [&](const std::vector<StagingCopy>& staging) {
      for (const auto& copy : staging) {
        lmcache_memcpy_async_on_stream(copy.dest, copy.src, copy.nbytes,
                                       direction, copy.host_offset,
                                       host_buffer_alignment, stream);
      }
    };

    for (const auto& step : batch_steps) {
      // H2D stages CPU->NPU temp buffers before the kernel reads them; D2H
      // stages NPU->CPU after the kernel writes them. The per-step ordering
      // must be preserved because temp buffers are reused across steps.
      if (is_h2d) {
        do_staging(step.staging);
      }
      for (const auto& launch : step.launches) {
        const KernelGroupSpec& group = kernel_group_specs[launch.group_idx];
        std::vector<int64_t> lmcache_objects_ptrs = device_lmc_ptrs(
            std::vector<int64_t>(
                group.lmcache_objects_ptrs.begin(),
                group.lmcache_objects_ptrs.begin() + launch.num_objects));
        int64_t* block_ids_base = reinterpret_cast<int64_t*>(
            group.block_ids_base +
            static_cast<uintptr_t>(launch.block_ids_offset) * sizeof(int64_t));
        const PackedPlaneSpec spec =
            packed_planes(group.shape_desc, group.engine_kv_format);
        launch_block_transfer_objects(
            launch_type(group.shape_desc, spec), aiv_num, stream,
            reinterpret_cast<uint8_t*>(group.paged_buffer_ptrs),
            lmcache_objects_ptrs, block_ids_base, launch.total_blocks,
            static_cast<int>(launch.total_blocks / launch.num_objects),
            group.shape_desc, group.lmcache_chunk_size,
            launch.skip_prefix_n_blocks, is_h2d, spec);
      }
      if (!is_h2d) {
        do_staging(step.staging);
      }
    }
    return 0;
  });
  cmd.Run();
}

void lmcache_memcpy_async_on_stream(uintptr_t dest, uintptr_t src, size_t nbytes,
                                    TransferDirection direction,
                                    size_t host_buffer_offset,
                                    size_t host_buffer_alignments,
                                    aclrtStream stream) {
  // Check that host_buffer_alignments is power of two
  TORCH_CHECK(host_buffer_alignments > 0 &&
                  (host_buffer_alignments & (host_buffer_alignments - 1)) == 0,
              "host_buffer_alignments must be a non-zero power of two");

  size_t offset = 0;
  const size_t mask = host_buffer_alignments - 1;
  const aclrtMemcpyKind kind = (direction == TransferDirection::H2D)
                                    ? ACL_MEMCPY_HOST_TO_DEVICE
                                    : ACL_MEMCPY_DEVICE_TO_HOST;

  // Split the copy at the host buffer's alignment boundaries: each chunk
  // stays inside one aligned region (port of upstream lmcache_memcpy_async).
  while (offset < nbytes) {
    const size_t aligned_area_end =
        ((offset + host_buffer_offset) & ~mask) + host_buffer_alignments;
    const size_t real_end =
        std::min<size_t>(host_buffer_offset + nbytes, aligned_area_end);
    const size_t max_nbytes = real_end - offset - host_buffer_offset;

    const aclError ret = aclrtMemcpyAsync(
        reinterpret_cast<void*>(dest + offset), max_nbytes,
        reinterpret_cast<const void*>(src + offset), max_nbytes, kind, stream);
    TORCH_CHECK(ret == ACL_ERROR_NONE, "aclrtMemcpyAsync failed: ret=", ret);

    offset += max_nbytes;
  }
}

void lmcache_memcpy_async(uintptr_t dest, uintptr_t src, size_t nbytes,
                          TransferDirection direction,
                          size_t host_buffer_offset,
                          size_t host_buffer_alignments) {
  lmcache_memcpy_async_on_stream(
      dest, src, nbytes, direction, host_buffer_offset, host_buffer_alignments,
      c10_npu::getCurrentNPUStream().stream());
}
