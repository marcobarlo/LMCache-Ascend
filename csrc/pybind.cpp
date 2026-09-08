// SPDX-License-Identifier: Apache-2.0

#include "cachegen_kernels.h"
#include "dcmi_management.h"
#include "managed_mem.h"
#include "mem_alloc.h"
#include "mem_kernels.h"
#include "mp_mem_kernels.h"
#include "pac_kernels.h"
#include "pos_kernels.h"
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/csrc/autograd/python_variable.h>
#include <torch/torch.h>

namespace py = pybind11;

std::vector<torch::Tensor> normalize_kv_caches(const py::object &input) {
  if (THPVariable_Check(input.ptr())) {
    return {input.cast<torch::Tensor>()};
  } else if (py::isinstance<py::tuple>(input)) {
    return input.cast<std::vector<torch::Tensor>>();
  } else {
    throw std::runtime_error(
        "vllm_kv_caches must be a Tensor or a tuple of Tensors");
  }
}

void single_layer_kv_transfer_wrapper(torch::Tensor &lmc_key_value_cache,
                                      const py::object &vllm_kv_caches_obj,
                                      torch::Tensor &slot_mapping,
                                      bool direction, int kvcache_format_raw,
                                      bool token_major, bool vllm_two_major) {
  auto vllm_kv_caches = normalize_kv_caches(vllm_kv_caches_obj);
  single_layer_kv_transfer(lmc_key_value_cache, vllm_kv_caches, slot_mapping,
                           direction, kvcache_format_raw, token_major,
                           vllm_two_major);
}

void batched_fused_single_layer_kv_transfer_wrapper(
    std::vector<torch::Tensor> &lmc_tensors, torch::Tensor &staging_cache,
    const py::object &vllm_kv_caches_obj, torch::Tensor &slot_mapping_full,
    std::vector<int64_t> &chunk_offsets, std::vector<int64_t> &chunk_sizes,
    bool direction, int kvcache_format_raw, bool token_major,
    bool vllm_two_major) {
  auto vllm_kv_caches = normalize_kv_caches(vllm_kv_caches_obj);
  batched_fused_single_layer_kv_transfer(
      lmc_tensors, staging_cache, vllm_kv_caches, slot_mapping_full,
      chunk_offsets, chunk_sizes, direction, kvcache_format_raw, token_major,
      vllm_two_major);
}

PYBIND11_MODULE(c_ops, m) {
  m.def("get_device_ptr", [](uintptr_t ptr_addr) {
    return reinterpret_cast<uintptr_t>(
        get_device_ptr(reinterpret_cast<void *>(ptr_addr)));
  });
  m.def("register_mapping",
        [](uintptr_t host_ptr, uintptr_t dev_ptr, size_t size) {
          return reinterpret_cast<uintptr_t>(
              register_mapping(reinterpret_cast<void *>(host_ptr),
                               reinterpret_cast<void *>(dev_ptr), size));
        });
  m.def("unregister_ptr", [](uintptr_t ptr_addr) {
    return unregister_ptr(reinterpret_cast<void *>(ptr_addr));
  });
  // Trailing dim/block args default to 0 so legacy call sites (pre-DSv4) need
  // no changes; new code passes dsa_c8_scale_plane_bytes and
  // paged_kv_block_size.
  m.def("multi_layer_kv_transfer", &multi_layer_kv_transfer,
        py::arg("key_value"), py::arg("key_value_ptrs"),
        py::arg("slot_mapping"), py::arg("paged_memory_device"),
        py::arg("page_buffer_size"), py::arg("direction"), py::arg("use_mla"),
        py::arg("kvcache_format_raw"), py::arg("k_hidden_dims") = 0,
        py::arg("v_hidden_dims") = 0, py::arg("dsa_hidden_dims") = 0,
        py::arg("dsa_c8_scale_plane_bytes") = 0,
        py::arg("paged_kv_block_size") = 0);
  m.def("fused_multi_layer_kv_transfer", &fused_multi_layer_kv_transfer,
        py::arg("key_value"), py::arg("staging_cache"),
        py::arg("key_value_ptrs"), py::arg("slot_mapping"),
        py::arg("paged_memory_device"), py::arg("page_buffer_size"),
        py::arg("direction"), py::arg("use_mla"), py::arg("kvcache_format_raw"),
        py::arg("k_hidden_dims") = 0, py::arg("v_hidden_dims") = 0,
        py::arg("dsa_hidden_dims") = 0, py::arg("dsa_c8_scale_plane_bytes") = 0,
        py::arg("paged_kv_block_size") = 0);
  m.def("multi_layer_kv_transfer_310p", &multi_layer_kv_transfer_310p);
  m.def("single_layer_kv_transfer", &single_layer_kv_transfer_wrapper);
  m.def("batched_fused_single_layer_kv_transfer",
        &batched_fused_single_layer_kv_transfer_wrapper);
  m.def("multi_layer_kv_transfer_unilateral",
        &multi_layer_kv_transfer_unilateral);
  m.def("load_and_reshape_flash", &load_and_reshape_flash);
  m.def("reshape_and_cache_back_flash", &reshape_and_cache_back_flash);
  m.def("encode_fast_new", &encode_ascend_new);
  m.def("decode_fast_new", &decode_ascend_new);
  m.def("decode_fast_prefsum", &decode_ascend_prefsum);
  m.def("calculate_cdf", &calculate_cdf);
  m.def("rotary_embedding_k_fused", &rotary_embedding_k_fused);
  m.def("alloc_pinned_ptr", &alloc_pinned_ptr);
  m.def("free_pinned_ptr", &free_pinned_ptr);
  m.def("alloc_pinned_numa_ptr", &alloc_pinned_numa_ptr);
  m.def("free_pinned_numa_ptr", &free_pinned_numa_ptr);
  m.def("get_gpu_pci_bus_id", &get_npu_pci_bus_id);

  m.def("pac_prepare_enc_metadata", &pac_prepare_enc_metadata);
  m.def("pac_encode", &pac_encode);
  m.def("pac_decode", &pac_decode);

  py::enum_<TransferDirection>(m, "TransferDirection", py::module_local())
      .value("H2D", TransferDirection::H2D)
      .value("D2H", TransferDirection::D2H)
      .export_values();
  py::enum_<EngineKVFormat>(m, "EngineKVFormat", py::module_local())
      .value("NB_NL_TWO_BS_NH_HS", EngineKVFormat::NB_NL_TWO_BS_NH_HS)
      .value("NL_X_TWO_NB_BS_NH_HS", EngineKVFormat::NL_X_TWO_NB_BS_NH_HS)
      .value("NL_X_NB_TWO_BS_NH_HS", EngineKVFormat::NL_X_NB_TWO_BS_NH_HS)
      .value("NL_X_NB_BS_HS", EngineKVFormat::NL_X_NB_BS_HS)
      .value("TWO_X_NL_X_NBBS_NH_HS", EngineKVFormat::TWO_X_NL_X_NBBS_NH_HS)
      .value("NL_X_NBBS_ONE_HS", EngineKVFormat::NL_X_NBBS_ONE_HS)
      .value("NL_X_TWO_NB_NH_BS_HS", EngineKVFormat::NL_X_TWO_NB_NH_BS_HS)
      .value("NL_X_NB_TWO_NH_BS_HS", EngineKVFormat::NL_X_NB_TWO_NH_BS_HS)
      .value("NB_NL_TWO_NH_BS_HS", EngineKVFormat::NB_NL_TWO_NH_BS_HS)
      .value("TWO_X_NL_X_NB_BS_NH_HS", EngineKVFormat::TWO_X_NL_X_NB_BS_NH_HS)
      .value("NL_X_NB_NH_BS_TWO_HS", EngineKVFormat::NL_X_NB_NH_BS_TWO_HS)
      .value("NL_X_NB_BS_NH_TWO_HS", EngineKVFormat::NL_X_NB_BS_NH_TWO_HS)
      .value("NL_X_NB_NH_BS_CS", EngineKVFormat::NL_X_NB_NH_BS_CS)
      .value("NL_X_NB_BS_NH_CS", EngineKVFormat::NL_X_NB_BS_NH_CS)
      .value("NL_X_NB_BSV_BSS", EngineKVFormat::NL_X_NB_BSV_BSS)
      .value("NL_X_TWO_NB_NH_ONE_BS_HS",
             EngineKVFormat::NL_X_TWO_NB_NH_ONE_BS_HS)
      .value("NL_X_TWO_X_NB_BS_NH_HS", EngineKVFormat::NL_X_TWO_X_NB_BS_NH_HS)
      .value("NL_X_TWO_X_NB_BS_HS", EngineKVFormat::NL_X_TWO_X_NB_BS_HS)
      .export_values();
  m.def("is_cross_layer", [](EngineKVFormat f) { return is_cross_layer(f); },
        py::arg("engine_kv_format"));
  m.def("is_kv_list", [](EngineKVFormat f) { return is_kv_list(f); },
        py::arg("engine_kv_format"));
  m.def("is_layer_list", [](EngineKVFormat f) { return is_layer_list(f); },
        py::arg("engine_kv_format"));
  m.def("is_mla", [](EngineKVFormat f) { return is_mla(f); },
        py::arg("engine_kv_format"));
  m.def("is_kv_second_tuple",
        [](EngineKVFormat f) { return is_kv_second_tuple(f); },
        py::arg("engine_kv_format"));
  // dynamic_attr: MP stores torch dtype as a Python-only side channel
  // (make_page_buffer_shape_desc). Same flag as lmcache_native.
  py::class_<PageBufferShapeDesc>(m, "PageBufferShapeDesc",
                                  py::module_local(), py::dynamic_attr())
      .def(py::init<>())
      .def_readwrite("kv_size", &PageBufferShapeDesc::kv_size)
      .def_readwrite("nl", &PageBufferShapeDesc::nl)
      .def_readwrite("nb", &PageBufferShapeDesc::nb)
      .def_readwrite("bs", &PageBufferShapeDesc::bs)
      .def_readwrite("nh", &PageBufferShapeDesc::nh)
      .def_readwrite("hs", &PageBufferShapeDesc::hs)
      .def_readwrite("element_size", &PageBufferShapeDesc::element_size)
      .def_readwrite("block_stride_elems",
                     &PageBufferShapeDesc::block_stride_elems);
  m.def(
      "multi_layer_block_kv_transfer",
      [](const torch::Tensor& paged_buffer_ptrs_tensor,
         std::vector<int64_t> lmcache_objects_ptrs,
         const torch::Tensor& block_ids, const torch::Device& device,
         const py::object& direction, const py::object& shape_desc,
         int lmcache_chunk_size, const py::object& engine_kv_format,
         int skip_prefix_n_blocks) {
        const auto dir = static_cast<TransferDirection>(
            py::int_(direction).cast<int>());
        const auto fmt = static_cast<EngineKVFormat>(
            py::int_(engine_kv_format).cast<int>());
        // Duck-type attrs. Do not isinstance-check the local C++
        // PageBufferShapeDesc: MP passes lmcache_native's class (different
        // module, not module_local). Cross-module isinstance SIGSEGVs on
        // AffinityThreadPool workers.
        PageBufferShapeDesc sd;
        sd.kv_size = shape_desc.attr("kv_size").cast<int>();
        sd.nl = shape_desc.attr("nl").cast<int>();
        sd.nb = shape_desc.attr("nb").cast<int>();
        sd.bs = shape_desc.attr("bs").cast<int>();
        sd.nh = shape_desc.attr("nh").cast<int>();
        sd.hs = shape_desc.attr("hs").cast<int>();
        sd.element_size = shape_desc.attr("element_size").cast<int>();
        sd.block_stride_elems =
            shape_desc.attr("block_stride_elems").cast<int>();
        // Keep the GIL through TORCH_CHECK so failures surface as Python
        // exceptions instead of aborting the lmcache server.
        multi_layer_block_kv_transfer(
            paged_buffer_ptrs_tensor, std::move(lmcache_objects_ptrs),
            block_ids, device, dir, sd, lmcache_chunk_size, fmt,
            skip_prefix_n_blocks);
      },
      py::arg("paged_buffer_ptrs_tensor"), py::arg("lmcache_objects_ptrs"),
      py::arg("block_ids"), py::arg("device"), py::arg("direction"),
      py::arg("shape_desc"), py::arg("lmcache_chunk_size"),
      py::arg("engine_kv_format"), py::arg("skip_prefix_n_blocks"));
  py::class_<StagingCopy>(m, "StagingCopy", py::module_local())
      .def(py::init([](uintptr_t dest, uintptr_t src, size_t nbytes,
                       size_t host_offset) {
             return StagingCopy{dest, src, nbytes, host_offset};
           }),
           py::arg("dest"), py::arg("src"), py::arg("nbytes"),
           py::arg("host_offset"));
  py::class_<LaunchVar>(m, "LaunchVar", py::module_local())
      .def(py::init([](int group_idx, int64_t block_ids_offset,
                       int total_blocks, int num_objects,
                       int skip_prefix_n_blocks) {
             return LaunchVar{group_idx, block_ids_offset, total_blocks,
                              num_objects, skip_prefix_n_blocks};
           }),
           py::arg("group_idx"), py::arg("block_ids_offset"),
           py::arg("total_blocks"), py::arg("num_objects"),
           py::arg("skip_prefix_n_blocks"));
  py::class_<BatchStep>(m, "BatchStep", py::module_local())
      .def(py::init([](std::vector<StagingCopy> staging,
                       std::vector<LaunchVar> launches) {
             return BatchStep{std::move(staging), std::move(launches)};
           }),
           py::arg("staging"), py::arg("launches"));
  py::class_<KernelGroupSpec>(m, "KernelGroupSpec", py::module_local())
      .def(py::init([](uintptr_t paged_buffer_ptrs,
                       std::vector<int64_t> lmcache_objects_ptrs,
                       const py::object& shape_desc, int lmcache_chunk_size,
                       int engine_kv_format, uintptr_t block_ids_base,
                       int64_t block_ids_capacity) {
             // Duck-type lmcache_native.PageBufferShapeDesc. Do not
             // py::isinstance the local C++ struct (cross-module SIGSEGV).
             PageBufferShapeDesc sd;
             sd.kv_size = shape_desc.attr("kv_size").cast<int>();
             sd.nl = shape_desc.attr("nl").cast<int>();
             sd.nb = shape_desc.attr("nb").cast<int>();
             sd.bs = shape_desc.attr("bs").cast<int>();
             sd.nh = shape_desc.attr("nh").cast<int>();
             sd.hs = shape_desc.attr("hs").cast<int>();
             sd.element_size = shape_desc.attr("element_size").cast<int>();
             sd.block_stride_elems =
                 shape_desc.attr("block_stride_elems").cast<int>();
             return KernelGroupSpec{
                 paged_buffer_ptrs,
                 std::move(lmcache_objects_ptrs),
                 sd,
                 lmcache_chunk_size,
                 static_cast<EngineKVFormat>(engine_kv_format),
                 block_ids_base,
                 block_ids_capacity};
           }),
           py::arg("paged_buffer_ptrs"), py::arg("lmcache_objects_ptrs"),
           py::arg("shape_desc"), py::arg("lmcache_chunk_size"),
           py::arg("engine_kv_format"), py::arg("block_ids_base"),
           py::arg("block_ids_capacity"));
  m.def(
      "execute_object_group_transfer",
      [](int direction, const torch::Device& device,
         size_t host_buffer_alignment,
         const std::vector<KernelGroupSpec>& kernel_group_specs,
         const std::vector<BatchStep>& batch_steps) {
        return execute_object_group_transfer(
            static_cast<TransferDirection>(direction), device,
            host_buffer_alignment, kernel_group_specs, batch_steps);
      },
      py::arg("direction"), py::arg("device"), py::arg("host_buffer_alignment"),
      py::arg("kernel_group_specs"), py::arg("batch_steps"));
      // No gil_scoped_release: NPU OpCommand.Run() posts to torch_npu's
      // Python task queue. Releasing the GIL here deadlocks (CUDA launches
      // without that queue). One Run() still implies one stream wait.
  m.def("lmcache_memcpy_async", &lmcache_memcpy_async,
        py::call_guard<py::gil_scoped_release>());
}
