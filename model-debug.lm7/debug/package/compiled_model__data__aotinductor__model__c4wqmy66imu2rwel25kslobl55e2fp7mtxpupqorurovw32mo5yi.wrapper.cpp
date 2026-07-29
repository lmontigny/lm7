
#include <torch/csrc/inductor/aoti_include/cpu.h>
// Definition of AOTI runtime interface functions

#include <torch/csrc/inductor/aoti_runtime/interface.h>
#include <torch/csrc/inductor/aoti_runtime/model_container.h>

#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

// Stores the last error message from a failed AOTI runtime call so that
// callers on the other side of the C ABI boundary can retrieve it via
// AOTInductorGetLastError(). Without this, exception messages (e.g.
// "CUDA error: an illegal memory access was encountered") are lost when
// CONVERT_EXCEPTION_TO_ERROR_CODE catches them and returns an error code.
static thread_local std::string g_aoti_last_error;

#define CONVERT_EXCEPTION_TO_ERROR_CODE(...)      \
  try {                                           \
    g_aoti_last_error.clear();                    \
    __VA_ARGS__                                   \
  } catch (const std::exception& e) {             \
    g_aoti_last_error = e.what();                 \
    std::cerr << "Error: " << e.what() << '\n';   \
    return AOTI_RUNTIME_FAILURE;                  \
  } catch (...) {                                 \
    g_aoti_last_error = "Unknown exception";      \
    std::cerr << "Unknown exception occurred.\n"; \
    return AOTI_RUNTIME_FAILURE;                  \
  }                                               \
  return AOTI_RUNTIME_SUCCESS;

#define AOTI_VECTOR_SIZE_CHECK(actual_size, expected_size, name)  \
  do {                                                            \
    AOTI_RUNTIME_CHECK(                                           \
        actual_size == expected_size,                             \
        "expected " + std::string(name) + " vector size to be " + \
            std::to_string(expected_size) + ", but got " +        \
            std::to_string(actual_size));                         \
  } while (0)

// AOTInductor uses at::addmm_out, which doesn't supports
// arguments that requires gradient. For this reason, we
// enforce no_grad context for run APIs.
//
// A RAII, thread local (!) guard that enables or disables grad mode upon
// construction, and sets it back to the original value upon destruction.
struct AOTINoGradGuard {
  AOTINoGradGuard() {
    aoti_torch_grad_mode_set_enabled(false);
  }
  AOTINoGradGuard(const AOTINoGradGuard&) = delete;
  AOTINoGradGuard(AOTINoGradGuard&&) noexcept = delete;
  ~AOTINoGradGuard() {
    aoti_torch_grad_mode_set_enabled(prev_mode);
  }
  AOTINoGradGuard& operator=(const AOTINoGradGuard&) = delete;
  AOTINoGradGuard& operator=(AOTINoGradGuard&&) noexcept = delete;
  bool prev_mode{aoti_torch_grad_mode_is_enabled()};
};

namespace {

std::unordered_map<std::string, AtenTensorHandle> constant_map_from_pairs(
    const AOTInductorConstantMapEntry* pairs,
    size_t num_pairs) {
  std::unordered_map<std::string, AtenTensorHandle> input_map;
  input_map.reserve(num_pairs);
  for (size_t i = 0; i < num_pairs; ++i) {
    input_map.emplace(pairs[i].name, pairs[i].handle);
  }
  return input_map;
}

// Shared constructor for AOTInductorModelCreate / AOTInductorModelCreateV2.
// `populate(constant_map)` is called between model construction and
// optional embedded-blob loading.
template <typename Populate>
AOTIRuntimeError createModelImpl(
    AOTInductorModelHandle* model_handle,
    bool load_constants_from_blob,
    Populate&& populate) {
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto constant_map = std::make_shared<torch::aot_inductor::ConstantMap>();
    auto constant_array = std::make_shared<
        std::vector<torch::aot_inductor::ConstantHandle>>();
    auto* model = new torch::aot_inductor::AOTInductorModel(
        constant_map,
        constant_array,
        // device_str is hardcoded, as AOTInductorModelCreate is only used
        // for CPU models.
        "cpu",
        "");
    populate(*constant_map);
    if (load_constants_from_blob) {
      model->load_constants();
    }
    *model_handle = reinterpret_cast<AOTInductorModelHandle>(model);
  })
}

} // namespace

extern "C" {

AOTIRuntimeError AOTInductorModelContainerCreate(
    AOTInductorModelContainerHandle* container_handle,
    size_t num_models,
    bool is_cpu,
    const char* cubin_dir) {
      return AOTInductorModelContainerCreateWithDevice(
        container_handle,
        num_models,
        is_cpu ? "cpu" : "cuda",
        cubin_dir);
}

AOTIRuntimeError AOTInductorModelContainerCreateWithDevice(
    AOTInductorModelContainerHandle* container_handle,
    size_t num_models,
    const char* device_str,
    const char* cubin_dir) {

  if (num_models == 0) {
    std::cerr << "Error: num_models must be positive, but got 0\n";
    return AOTI_RUNTIME_FAILURE;
  }
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    std::optional<std::string> cubin_dir_opt;
    if (cubin_dir != nullptr) {
      cubin_dir_opt.emplace(cubin_dir);
    }
    auto* container = new torch::aot_inductor::AOTInductorModelContainer(
        num_models, std::string(device_str), cubin_dir_opt);
    *container_handle =
        reinterpret_cast<AOTInductorModelContainerHandle>(container);
  })
}


AOTIRuntimeError AOTInductorModelContainerDelete(
    AOTInductorModelContainerHandle container_handle) {
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto* container =
        reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
            container_handle);
    delete container;
  });
}

AOTIRuntimeError AOTInductorModelContainerRun(
    AOTInductorModelContainerHandle container_handle,
    AtenTensorHandle* input_handles, // array of input AtenTensorHandle; handles
                                     // are stolen; the array itself is borrowed
    size_t num_inputs,
    AtenTensorHandle*
        output_handles, // array for writing output AtenTensorHandle; handles
                        // will be stolen by the caller; the array itself is
                        // borrowed
    size_t num_outputs,
    AOTInductorStreamHandle stream_handle,
    AOTIProxyExecutorHandle proxy_executor_handle) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  AOTI_VECTOR_SIZE_CHECK(num_inputs, container->num_inputs(), "inputs");
  AOTI_VECTOR_SIZE_CHECK(num_outputs, container->num_outputs(), "outputs");

  auto stream =
      reinterpret_cast<torch::aot_inductor::DeviceStreamType>(stream_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    AOTINoGradGuard guard;
    container->run(
        input_handles, output_handles, stream, proxy_executor_handle);
  })
}

AOTIRuntimeError AOTInductorModelContainerRunSingleThreaded(
    AOTInductorModelContainerHandle container_handle,
    AtenTensorHandle* input_handles, // array of input AtenTensorHandle; handles
                                     // are stolen; the array itself is borrowed
    size_t num_inputs,
    AtenTensorHandle*
        output_handles, // array for writing output AtenTensorHandle; handles
                        // will be stolen by the caller; the array itself is
                        // borrowed
    size_t num_outputs,
    AOTInductorStreamHandle stream_handle,
    AOTIProxyExecutorHandle proxy_executor_handle) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  AOTI_VECTOR_SIZE_CHECK(num_inputs, container->num_inputs(), "inputs");
  AOTI_VECTOR_SIZE_CHECK(num_outputs, container->num_outputs(), "outputs");

  auto stream =
      reinterpret_cast<torch::aot_inductor::DeviceStreamType>(stream_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    AOTINoGradGuard guard;
    container->run_single_threaded(
        input_handles, output_handles, stream, proxy_executor_handle);
  })
}

AOTIRuntimeError AOTInductorModelContainerGetNumConstants(
    AOTInductorModelContainerHandle container_handle,
    size_t* num_constants) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
    { *num_constants = container->num_constants(); })
}

AOTIRuntimeError AOTInductorModelContainerGetConstantName(
    AOTInductorModelContainerHandle container_handle,
    size_t idx,
    const char** name) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
    { *name = container->constant_name(idx); })
}

AOTIRuntimeError AOTInductorModelContainerGetConstantOriginalFQN(
    AOTInductorModelContainerHandle container_handle,
    size_t idx,
    const char** original_fqn) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
    { *original_fqn = container->constant_original_fqn(idx); })
}

AOTIRuntimeError AOTInductorModelContainerGetConstantFromFolded(
    AOTInductorModelContainerHandle container_handle,
    size_t idx,
    bool* from_folded) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({ *from_folded = container->constant_from_folded(idx); })
}

AOTIRuntimeError AOTInductorModelContainerGetConstantType(
    AOTInductorModelContainerHandle container_handle,
    size_t idx,
    int32_t* type) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({ *type = container->constant_type(idx); })
}

AOTIRuntimeError AOTInductorModelContainerGetConstantDtype(
    AOTInductorModelContainerHandle container_handle,
    size_t idx,
    int32_t* dtype) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
    { *dtype = container->constant_dtype(idx); })
}

AOTIRuntimeError AOTInductorModelContainerGetConstantDataSize(
  AOTInductorModelContainerHandle container_handle,
  size_t idx,
  size_t* data_size) {
  auto* container =
    reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
        container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
    { *data_size = container->constant_data_size(idx); })
}

AOTIRuntimeError AOTInductorModelContainerExtractConstantsMap(
    AOTInductorModelContainerHandle container_handle,
    AOTInductorConstantMapHandle constant_map_handle,
    bool use_inactive) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  auto constants_map = reinterpret_cast<std::unordered_map<std::string, AtenTensorHandle>*>(constant_map_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
    { const auto ret = container->extract_constants_map(use_inactive);
      for (const auto& pair: ret) {
        constants_map->emplace(pair.first, pair.second);
      }
    })
}

AOTIRuntimeError AOTInductorModelContainerExtractConstantsMapEntries(
    AOTInductorModelContainerHandle container_handle,
    const AOTInductorConstantMapEntry** entries,
    size_t* num_entries,
    bool use_inactive) {
  if (!entries || !num_entries) {
    return AOTI_RUNTIME_FAILURE;
  }
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    const auto& extracted =
        container->extract_constants_map_entries(use_inactive);
    *entries = extracted.empty() ? nullptr : extracted.data();
    *num_entries = extracted.size();
  })
}

AOTIRuntimeError AOTInductorModelContainerUpdateUserManagedConstantBuffer(
    AOTInductorModelContainerHandle container_handle,
    AOTInductorConstantMapHandle constant_map_handle,
    bool use_inactive,
    bool validate_full_update) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  auto input_map = reinterpret_cast<std::unordered_map<std::string, AtenTensorHandle>*>(constant_map_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    container->update_constant_buffer(
        *input_map, use_inactive, validate_full_update, /* user_managed = */ true);
  })
}

AOTIRuntimeError AOTInductorModelContainerUpdateUserManagedConstantBufferPairs(
    AOTInductorModelContainerHandle container_handle,
    const AOTInductorConstantMapEntry* pairs,
    size_t num_pairs,
    bool use_inactive,
    bool validate_full_update) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(container_handle);
  // Build a local unordered_map inside
  std::unordered_map<std::string, AtenTensorHandle> input_map;
  input_map.reserve(num_pairs);
  for (size_t i = 0; i < num_pairs; ++i) {
      input_map.emplace(pairs[i].name, pairs[i].handle);
  }
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    container->update_constant_buffer(
        input_map, use_inactive, validate_full_update, /*user_managed=*/true);
  })
}

AOTIRuntimeError AOTInductorModelContainerUpdateConstantBuffer(
    AOTInductorModelContainerHandle container_handle,
    AOTInductorConstantMapHandle constant_map_handle,
    bool use_inactive,
    bool validate_full_update) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  auto input_map = reinterpret_cast<std::unordered_map<std::string, AtenTensorHandle>*>(constant_map_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    container->update_constant_buffer(
        *input_map, use_inactive, validate_full_update);
  })
}

AOTIRuntimeError AOTInductorModelContainerUpdateConstantBufferPairs(
    AOTInductorModelContainerHandle container_handle,
    const AOTInductorConstantMapEntry* pairs,
    size_t num_pairs,
    bool use_inactive,
    bool validate_full_update) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  auto input_map = constant_map_from_pairs(pairs, num_pairs);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    container->update_constant_buffer(
        input_map, use_inactive, validate_full_update);
  })
}

AOTIRuntimeError AOTInductorModelContainerUpdateConstantBufferFromCpu(
    AOTInductorModelContainerHandle container_handle,
    AOTInductorConstantMapHandle constant_map_handle,
    bool use_inactive,
    bool validate_full_update) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  auto input_map = reinterpret_cast<std::unordered_map<std::string, AtenTensorHandle>*>(constant_map_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    container->update_constant_buffer(
        *input_map,
        use_inactive,
        validate_full_update,
        /*user_managed=*/false,
        /*allow_h2d_copy=*/true);
  })
}

AOTIRuntimeError AOTInductorModelContainerUpdateConstantBufferFromCpuPairs(
    AOTInductorModelContainerHandle container_handle,
    const AOTInductorConstantMapEntry* pairs,
    size_t num_pairs,
    bool use_inactive,
    bool validate_full_update) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  auto input_map = constant_map_from_pairs(pairs, num_pairs);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    container->update_constant_buffer(
        input_map,
        use_inactive,
        validate_full_update,
        /*user_managed=*/false,
        /*allow_h2d_copy=*/true);
  })
}

AOTIRuntimeError AOTInductorModelContainerUpdateInactiveConstantBuffer(
    AOTInductorModelContainerHandle container_handle,
    AOTInductorConstantMapHandle constant_map_handle) {
  return AOTInductorModelContainerUpdateConstantBuffer(
      container_handle,
      constant_map_handle,
      /*use_inactive=*/true,
      /*validate_full_update=*/true);
}

AOTIRuntimeError AOTInductorModelContainerUpdateInactiveConstantBufferPairs(
    AOTInductorModelContainerHandle container_handle,
    const AOTInductorConstantMapEntry* pairs,
    size_t num_pairs) {
  return AOTInductorModelContainerUpdateConstantBufferPairs(
      container_handle,
      pairs,
      num_pairs,
      /*use_inactive=*/true,
      /*validate_full_update=*/true);
}

AOTIRuntimeError AOTInductorModelContainerFreeInactiveConstantBuffer(
    AOTInductorModelContainerHandle container_handle) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    container->free_inactive_constant_buffer();
  })
}

AOTIRuntimeError AOTInductorModelContainerRunConstantFolding(
    AOTInductorModelContainerHandle container_handle,
    bool use_inactive,
    AOTInductorStreamHandle stream_handle,
    AOTIProxyExecutorHandle proxy_executor_handle) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  auto stream =
      reinterpret_cast<torch::aot_inductor::DeviceStreamType>(stream_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    AOTINoGradGuard guard;
    container->run_const_fold(use_inactive, stream, proxy_executor_handle);
  })
}

AOTIRuntimeError AOTInductorModelContainerSwapConstantBuffer(
    AOTInductorModelContainerHandle container_handle) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    container->swap_constant_buffer();
  })
}

AOTIRuntimeError AOTInductorModelContainerGetNumInputs(
    AOTInductorModelContainerHandle container_handle,
    size_t* ret_num_inputs) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
      { *ret_num_inputs = container->num_inputs(); })
}

AOTIRuntimeError AOTInductorModelContainerGetInputName(
    AOTInductorModelContainerHandle container_handle,
    size_t input_idx,
    const char** ret_input_names) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
      { *ret_input_names = container->input_name(input_idx); })
}

AOTIRuntimeError AOTInductorModelContainerGetNumOutputs(
    AOTInductorModelContainerHandle container_handle,
    size_t* ret_num_outputs) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
      { *ret_num_outputs = container->num_outputs(); })
}

AOTIRuntimeError AOTInductorModelContainerGetOutputName(
    AOTInductorModelContainerHandle container_handle,
    size_t output_idx,
    const char** ret_output_names) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
      { *ret_output_names = container->output_name(output_idx); })
}

AOTIRuntimeError AOTInductorModelContainerGetCallSpec(
    AOTInductorModelContainerHandle container_handle,
    const char** in_spec,
    const char** out_spec) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    *in_spec = container->get_in_spec();
    *out_spec = container->get_out_spec();
  })
}

AOTIRuntimeError AOTInductorModelCreate(
    AOTInductorModelHandle* model_handle,
    AOTInductorConstantMapHandle constant_map_handle) {
  return createModelImpl(
      model_handle, constant_map_handle == nullptr, [=](auto& constant_map) {
        auto* input_map = reinterpret_cast<
            std::unordered_map<std::string, AtenTensorHandle>*>(
            constant_map_handle);
        if (input_map) {
          for (const auto& kv : *input_map) {
            constant_map.emplace(kv.first, kv.second);
          }
        }
      });
}

AOTIRuntimeError AOTInductorModelCreateV2(
    AOTInductorModelHandle* model_handle,
    const AOTInductorConstantMapEntry* pairs,
    size_t num_pairs) {
  return createModelImpl(
      model_handle, pairs == nullptr || num_pairs == 0, [=](auto& constant_map) {
        if (pairs && num_pairs > 0) {
          constant_map.reserve(num_pairs);
          for (size_t i = 0; i < num_pairs; ++i) {
            constant_map.emplace(pairs[i].name, pairs[i].handle);
          }
        }
      });
}

AOTIRuntimeError AOTInductorModelRun(
    AOTInductorModelHandle model_handle,
    AtenTensorHandle* input_handles,
    AtenTensorHandle* output_handles) {
  auto model =
      reinterpret_cast<torch::aot_inductor::AOTInductorModel*>(model_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    AOTINoGradGuard guard;
    model->run_impl(
        input_handles,
        output_handles,
        (torch::aot_inductor::DeviceStreamType) nullptr,
        nullptr);
  })
}

AOTIRuntimeError AOTInductorModelDelete(AOTInductorModelHandle model_handle){
    CONVERT_EXCEPTION_TO_ERROR_CODE({
      auto model = reinterpret_cast<torch::aot_inductor::AOTInductorModel*>(
          model_handle);
      delete model;
    })}

AOTIRuntimeError AOTInductorModelGetNumOutputs(
    AOTInductorModelHandle model_handle,
    size_t* ret_num_outputs) {
  CONVERT_EXCEPTION_TO_ERROR_CODE({
      auto model = reinterpret_cast<torch::aot_inductor::AOTInductorModel*>(model_handle);
      *ret_num_outputs = model->num_outputs();
  })
}

AOTIRuntimeError AOTInductorModelUpdateConstantsMap(
    AOTInductorModelHandle model_handle,
    AOTInductorConstantMapHandle constant_map_handle) {
  auto model =
      reinterpret_cast<torch::aot_inductor::AOTInductorModel*>(model_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto constant_map = std::make_shared<torch::aot_inductor::ConstantMap>();
    auto input_map =
        reinterpret_cast<std::unordered_map<std::string, AtenTensorHandle>*>(
            constant_map_handle);

    for (auto const& kv : *input_map) {
      constant_map->emplace(kv.first, kv.second);
    }
    model->update_constants_map(std::move(constant_map));
  })
}

// C-ABI-safe variant: uses an array of (name, handle) pairs instead of an
// opaque pointer to std::unordered_map, so the host and DSO can use
// different C++ standard libraries without ABI conflicts.
AOTIRuntimeError AOTInductorModelUpdateConstantsMapV2(
    AOTInductorModelHandle model_handle,
    const AOTInductorConstantMapEntry* pairs,
    int32_t num_pairs) {
  auto model =
      reinterpret_cast<torch::aot_inductor::AOTInductorModel*>(model_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto constant_map = std::make_shared<torch::aot_inductor::ConstantMap>();
    constant_map->reserve(num_pairs);
    for (int32_t i = 0; i < num_pairs; ++i) {
      constant_map->emplace(pairs[i].name, pairs[i].handle);
    }
    model->update_constants_map(std::move(constant_map));
  })
}

AOTIRuntimeError AOTInductorModelContainerGetConstantsBlobSize(
    AOTInductorModelContainerHandle container_handle,
    uint64_t* ret_size) {
  auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
      { *ret_size = container->constant_blob_size(); })
}


// Load weights from a single blob in weight_blob_ptr
AOTIRuntimeError AOTInductorModelUpdateConstantsFromBlob(
    AOTInductorModelContainerHandle container_handle,
    const uint8_t* weight_blob_ptr){
    auto* container =
      reinterpret_cast<torch::aot_inductor::AOTInductorModelContainer*>(
          container_handle);
  CONVERT_EXCEPTION_TO_ERROR_CODE(
      {container->update_constants_from_blob(weight_blob_ptr); })
    }


AOTIRuntimeError AOTInductorGetLastError(
    const char** error_msg) {
  *error_msg = g_aoti_last_error.c_str();
  return AOTI_RUNTIME_SUCCESS;
}

} // extern "C"

extern "C"  void  cpp_fused_relu_0(float* in_out_ptr0,
                       float* out_ptr0);
CACHE_TORCH_DTYPE(float32);
CACHE_TORCH_DEVICE(cpu);
CACHE_TORCH_LAYOUT(strided);
namespace torch::aot_inductor {
namespace {
class AOTInductorModelKernels : public AOTInductorModelKernelsBase {
  public:
    void* cpp_fused_relu_0{nullptr};
};
}  // namespace



AOTInductorModel::AOTInductorModel(std::shared_ptr<ConstantMap> constants_map,
                                   std::shared_ptr<std::vector<ConstantHandle>> constants_array,
                                   const std::string& device_str,
                                   std::optional<std::string> cubin_dir)
    : AOTInductorModelBase(1,
                           1,
                           3,
                           device_str,
                           std::move(cubin_dir),
                           true) {
    inputs_info_[0].name = "arg4_1";
    constants_info_[0].name = "constant_0_weight";
    constants_info_[0].dtype = cached_torch_dtype_float32;
    constants_info_[0].device_type = cached_torch_device_type_cpu;
    constants_info_[0].offset = 0;
    constants_info_[0].data_size = 2048;
    constants_info_[0].from_folded = false;
    constants_info_[0].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Unknown);
    constants_info_[0].shape = {32, 16};
    constants_info_[0].stride = {16, 1};
    constants_info_[0].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[0].original_fqn = "0.weight";
    constants_info_[1].name = "constant_0_bias";
    constants_info_[1].dtype = cached_torch_dtype_float32;
    constants_info_[1].device_type = cached_torch_device_type_cpu;
    constants_info_[1].offset = 0;
    constants_info_[1].data_size = 128;
    constants_info_[1].from_folded = false;
    constants_info_[1].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Unknown);
    constants_info_[1].shape = {32};
    constants_info_[1].stride = {1};
    constants_info_[1].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[1].original_fqn = "0.bias";
    constants_info_[2].name = "constant_2_weight";
    constants_info_[2].dtype = cached_torch_dtype_float32;
    constants_info_[2].device_type = cached_torch_device_type_cpu;
    constants_info_[2].offset = 0;
    constants_info_[2].data_size = 512;
    constants_info_[2].from_folded = false;
    constants_info_[2].type = static_cast<int32_t>(torch::aot_inductor::ConstantType::Unknown);
    constants_info_[2].shape = {4, 32};
    constants_info_[2].stride = {32, 1};
    constants_info_[2].layout = static_cast<int32_t>(cached_torch_layout_strided);
    constants_info_[2].original_fqn = "2.weight";
    update_constants_map(std::move(constants_map));
    update_constants_array(std::move(constants_array));
    in_spec_ = R"([1, {"type": "builtins.tuple", "context": "null", "children_spec": [{"type": "builtins.tuple", "context": "null", "children_spec": [{"type": null, "context": null, "children_spec": []}]}, {"type": "builtins.dict", "context": "[]", "children_spec": []}]}])";
    out_spec_ = R"([1, {"type": null, "context": null, "children_spec": []}])";
    outputs_info_[0].name = "output0";
    this->kernels_ = std::make_unique<AOTInductorModelKernels>();
}

std::unordered_map<std::string, AtenTensorHandle> AOTInductorModel::const_run_impl(
    DeviceStreamType stream,
    AOTIProxyExecutorHandle proxy_executor,
    bool initialization
) {

    if (!initialization) {
        std::cerr << "[WARNING] Calling constant_folding in model, but compiled with config: "
                  << "aot_inductor.use_runtime_constant_folding=False\n";
    }
    return {};
}
} // namespace torch::aot_inductor
using namespace torch::aot_inductor;
namespace torch::aot_inductor {

void AOTInductorModel::_const_run_impl(
    std::vector<AtenTensorHandle>& output_handles,
    DeviceStreamType stream,
    AOTIProxyExecutorHandle proxy_executor
) {}

AOTI_NOINLINE static void check_input_0(
    AtenTensorHandle* input_handles
) {
    ConstantHandle arg4_1 = ConstantHandle(input_handles[0]);
    int32_t arg4_1_dtype;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_dtype(arg4_1, &arg4_1_dtype));

    int32_t arg4_1_expected_dtype = aoti_torch_dtype_float32();
    if (arg4_1_expected_dtype != arg4_1_dtype) {
        std::stringstream ss;
        ss << "input_handles[0]: unmatched dtype, "
           << "expected: " << arg4_1_expected_dtype << "(at::kFloat), "
           << "but got: " << arg4_1_dtype << "\n";
        throw std::runtime_error(std::move(ss).str());
    }
    auto arg4_1_size = arg4_1.sizes();

    if (2 != arg4_1_size[0]) {
        std::stringstream ss;
        ss << "input_handles[0]: unmatched dim value at 0, "
           << "expected: 2, " << "but got: " << arg4_1_size[0]
           << "\n";
        throw std::runtime_error(std::move(ss).str());
    }

    if (16 != arg4_1_size[1]) {
        std::stringstream ss;
        ss << "input_handles[0]: unmatched dim value at 1, "
           << "expected: 16, " << "but got: " << arg4_1_size[1]
           << "\n";
        throw std::runtime_error(std::move(ss).str());
    }
    auto arg4_1_stride = arg4_1.strides();

    if (16 != arg4_1_stride[0]) {
        std::stringstream ss;
        ss << "input_handles[0]: unmatched stride value at 0, "
           << "expected: 16, " << "but got: " << arg4_1_stride[0]
           << "\n";
        throw std::runtime_error(std::move(ss).str());
    }

    if (1 != arg4_1_stride[1]) {
        std::stringstream ss;
        ss << "input_handles[0]: unmatched stride value at 1, "
           << "expected: 1, " << "but got: " << arg4_1_stride[1]
           << "\n";
        throw std::runtime_error(std::move(ss).str());
    }
    int32_t arg4_1_device_type;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_get_device_type(arg4_1, &arg4_1_device_type));

    int32_t arg4_1_expected_device_type = 0;
    if (arg4_1_expected_device_type != arg4_1_device_type) {
        std::stringstream ss;
        ss << "input_handles[0]: unmatched device type, "
        << "expected: " << arg4_1_expected_device_type << "0(cpu), "
        << "but got: " << arg4_1_device_type << "\n";
        throw std::runtime_error(std::move(ss).str());
    }
}

static bool _check_aoti_runtime_check_inputs_env() {
    const static char* env_var_value = getenv("AOTI_RUNTIME_CHECK_INPUTS");
    const static bool result = env_var_value != nullptr && env_var_value[0] != '0';
    return result;
}

AOTI_NOINLINE static void __check_inputs_outputs(
    AtenTensorHandle* input_handles,
    AtenTensorHandle* output_handles) {
    if (!_check_aoti_runtime_check_inputs_env()){
        return;
    }
    check_input_0(input_handles);
}

void AOTInductorModel::run_impl(
    AtenTensorHandle*
        input_handles, // array of input AtenTensorHandle; handles
                        // are stolen; the array itself is borrowed
    AtenTensorHandle*
        output_handles, // array for writing output AtenTensorHandle; handles
                        // will be stolen by the caller; the array itself is
                        // borrowed
    DeviceStreamType stream,
    AOTIProxyExecutorHandle proxy_executor
) {
    __check_inputs_outputs(input_handles, output_handles);
    auto inputs = steal_from_raw_handles_to_raii_handles(input_handles, 1);
    auto arg4_1 = std::move(inputs[0]);
    [[maybe_unused]] auto& constant_0_weight = constants_->at(0);
    [[maybe_unused]] auto& constant_0_bias = constants_->at(1);
    [[maybe_unused]] auto& constant_2_weight = constants_->at(2);
    inputs.clear();
    [[maybe_unused]] auto& kernels = static_cast<AOTInductorModelKernels&>(*this->kernels_.get());
    if (_check_aoti_runtime_check_inputs_env()) { assert_size_stride(arg4_1, {2L, 16L}, {16L, 1L}, "input"); }
    static constexpr int64_t int_array_0[] = {2L, 32L};
    static constexpr int64_t int_array_1[] = {32L, 1L};
    AtenTensorHandle buf0_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(2, int_array_0, int_array_1, cached_torch_dtype_float32, cached_torch_device_type_cpu, this->device_idx_, &buf0_handle));
    RAIIAtenTensorHandle buf0(buf0_handle);
    // Topologically Sorted Source Nodes: [linear], Original ATen: [aten.t, aten.addmm]
    static constexpr int64_t int_array_2[] = {16L, 32L};
    static constexpr int64_t int_array_3[] = {1L, 16L};
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_cpu_addmm_out(buf0, constant_0_bias, arg4_1, wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(constant_0_weight, 2, int_array_2, int_array_3, 0L)), 1L, 1L));
    arg4_1.reset();
    auto buf1 = std::move(buf0);  // reuse
    static constexpr int64_t int_array_4[] = {4L, };
    static constexpr int64_t int_array_5[] = {1L, };
    AtenTensorHandle buf2_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(1, int_array_4, int_array_5, cached_torch_dtype_float32, cached_torch_device_type_cpu, this->device_idx_, &buf2_handle));
    RAIIAtenTensorHandle buf2(buf2_handle);
    cpp_fused_relu_0((float*)(buf1.data_ptr()), (float*)(buf2.data_ptr()));
    static constexpr int64_t int_array_6[] = {2L, 4L};
    static constexpr int64_t int_array_7[] = {4L, 1L};
    AtenTensorHandle buf3_handle;
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_empty_strided(2, int_array_6, int_array_7, cached_torch_dtype_float32, cached_torch_device_type_cpu, this->device_idx_, &buf3_handle));
    RAIIAtenTensorHandle buf3(buf3_handle);
    // Topologically Sorted Source Nodes: [relu, linear_1], Original ATen: [aten.relu, aten.t, aten.addmm]
    static constexpr int64_t int_array_8[] = {32L, 4L};
    static constexpr int64_t int_array_9[] = {1L, 32L};
    AOTI_TORCH_ERROR_CODE_CHECK(aoti_torch_cpu_addmm_out(buf3, buf2, buf1, wrap_with_raii_handle_if_needed(reinterpret_tensor_wrapper(constant_2_weight, 2, int_array_8, int_array_9, 0L)), 1L, 1L));
    output_handles[0] = buf3.release();
} // AOTInductorModel::run_impl
} // namespace torch::aot_inductor




// Compile cmd
// g++ /tmp/torchinductor_bernard/cddhcpcdybzvymlvmkeivzsnk5lmiq7ice542tuiegeomvou7bn6/c4wqmy66imu2rwel25kslobl55e2fp7mtxpupqorurovw32mo5yi.wrapper.cpp -D TORCH_INDUCTOR_CPP_WRAPPER -D STANDALONE_TORCH_HEADER -D TORCH_INDUCTOR_PRECOMPILE_HEADERS -D  C10_USING_CUSTOM_GENERATED_MACROS -D CPU_CAPABILITY_AVX2  -O1 -DNDEBUG -fno-omit-frame-pointer -g1 -fno-trapping-math -funsafe-math-optimizations -ffinite-math-only -fno-signed-zeros -fno-finite-math-only -fno-unsafe-math-optimizations -fmath-errno -ffp-contract=off -fexcess-precision=fast -fno-tree-loop-vectorize -march=native -fPIC -Wall -std=c++20 -Wno-unused-variable -Wno-unknown-pragmas -pedantic -fopenmp  -include /tmp/torchinductor_bernard/precompiled_headers/cqbwvgtkgf24t42sn4u7vdxnbbdwtat6neb7jc2d5o4hjtih4row.h -I/usr/include/python3.12 -I/home/bernard/dev/lm7/.venv/lib/python3.12/site-packages/torch/include -I/home/bernard/dev/lm7/.venv/lib/python3.12/site-packages/torch/include/torch/csrc/api/include   -mavx2 -mfma -mf16c  -c -o /tmp/torchinductor_bernard/cddhcpcdybzvymlvmkeivzsnk5lmiq7ice542tuiegeomvou7bn6/c4wqmy66imu2rwel25kslobl55e2fp7mtxpupqorurovw32mo5yi.wrapper.o
// Link cmd
// g++ /tmp/torchinductor_bernard/cddhcpcdybzvymlvmkeivzsnk5lmiq7ice542tuiegeomvou7bn6/c4wqmy66imu2rwel25kslobl55e2fp7mtxpupqorurovw32mo5yi.wrapper.o /tmp/torchinductor_bernard/cddhcpcdybzvymlvmkeivzsnk5lmiq7ice542tuiegeomvou7bn6/cclnsjb5zbaads6kegi4qy5i2b6ixi2fzz4a3pu4i6qgf6closlx.kernel.o /tmp/torchinductor_bernard/cddhcpcdybzvymlvmkeivzsnk5lmiq7ice542tuiegeomvou7bn6/c4wqmy66imu2rwel25kslobl55e2fp7mtxpupqorurovw32mo5yi/ctrsxmmxgotwt2gcantv6qe5g4hugda73ji2v3ijip227636shte.weights.o -D TORCH_INDUCTOR_CPP_WRAPPER -D STANDALONE_TORCH_HEADER -D TORCH_INDUCTOR_PRECOMPILE_HEADERS -D  C10_USING_CUSTOM_GENERATED_MACROS -D CPU_CAPABILITY_AVX2  -O3 -DNDEBUG -fno-omit-frame-pointer -g1 -fno-trapping-math -funsafe-math-optimizations -ffinite-math-only -fno-signed-zeros -fno-finite-math-only -fno-unsafe-math-optimizations -fmath-errno -ffp-contract=off -fexcess-precision=fast -fno-tree-loop-vectorize -march=native -shared -fPIC -Wall -std=c++20 -Wno-unused-variable -Wno-unknown-pragmas -pedantic -fopenmp  -I/usr/include/python3.12 -I/home/bernard/dev/lm7/.venv/lib/python3.12/site-packages/torch/include -I/home/bernard/dev/lm7/.venv/lib/python3.12/site-packages/torch/include/torch/csrc/api/include   -mavx2 -mfma -mf16c  -o /tmp/torchinductor_bernard/cddhcpcdybzvymlvmkeivzsnk5lmiq7ice542tuiegeomvou7bn6/c4wqmy66imu2rwel25kslobl55e2fp7mtxpupqorurovw32mo5yi.wrapper.so  -ltorch -ltorch_cpu -lgomp  -L/usr/lib/x86_64-linux-gnu -L/home/bernard/dev/lm7/.venv/lib/python3.12/site-packages/torch/lib 

// Compile cmd
// g++ /tmp/torchinductor_bernard/cddhcpcdybzvymlvmkeivzsnk5lmiq7ice542tuiegeomvou7bn6/c4wqmy66imu2rwel25kslobl55e2fp7mtxpupqorurovw32mo5yi.wrapper.cpp -D TORCH_INDUCTOR_CPP_WRAPPER -D STANDALONE_TORCH_HEADER -D TORCH_INDUCTOR_PRECOMPILE_HEADERS -D  C10_USING_CUSTOM_GENERATED_MACROS -D CPU_CAPABILITY_AVX2  -O1 -DNDEBUG -fno-omit-frame-pointer -g1 -fno-trapping-math -funsafe-math-optimizations -ffinite-math-only -fno-signed-zeros -fno-finite-math-only -fno-unsafe-math-optimizations -fmath-errno -ffp-contract=off -fexcess-precision=fast -fno-tree-loop-vectorize -march=native -fPIC -Wall -std=c++20 -Wno-unused-variable -Wno-unknown-pragmas -pedantic -fopenmp  -include /tmp/torchinductor_bernard/precompiled_headers/cqbwvgtkgf24t42sn4u7vdxnbbdwtat6neb7jc2d5o4hjtih4row.h -I/usr/include/python3.12 -I/home/bernard/dev/lm7/.venv/lib/python3.12/site-packages/torch/include -I/home/bernard/dev/lm7/.venv/lib/python3.12/site-packages/torch/include/torch/csrc/api/include   -mavx2 -mfma -mf16c  -c -o /tmp/torchinductor_bernard/cddhcpcdybzvymlvmkeivzsnk5lmiq7ice542tuiegeomvou7bn6/c4wqmy66imu2rwel25kslobl55e2fp7mtxpupqorurovw32mo5yi.wrapper.o
// Link cmd
// g++ /tmp/torchinductor_bernard/cddhcpcdybzvymlvmkeivzsnk5lmiq7ice542tuiegeomvou7bn6/c4wqmy66imu2rwel25kslobl55e2fp7mtxpupqorurovw32mo5yi.wrapper.o /tmp/torchinductor_bernard/cddhcpcdybzvymlvmkeivzsnk5lmiq7ice542tuiegeomvou7bn6/cclnsjb5zbaads6kegi4qy5i2b6ixi2fzz4a3pu4i6qgf6closlx.kernel.o /tmp/torchinductor_bernard/cddhcpcdybzvymlvmkeivzsnk5lmiq7ice542tuiegeomvou7bn6/c4wqmy66imu2rwel25kslobl55e2fp7mtxpupqorurovw32mo5yi/ctrsxmmxgotwt2gcantv6qe5g4hugda73ji2v3ijip227636shte.weights.o -D TORCH_INDUCTOR_CPP_WRAPPER -D STANDALONE_TORCH_HEADER -D TORCH_INDUCTOR_PRECOMPILE_HEADERS -D  C10_USING_CUSTOM_GENERATED_MACROS -D CPU_CAPABILITY_AVX2  -O3 -DNDEBUG -fno-omit-frame-pointer -g1 -fno-trapping-math -funsafe-math-optimizations -ffinite-math-only -fno-signed-zeros -fno-finite-math-only -fno-unsafe-math-optimizations -fmath-errno -ffp-contract=off -fexcess-precision=fast -fno-tree-loop-vectorize -march=native -shared -fPIC -Wall -std=c++20 -Wno-unused-variable -Wno-unknown-pragmas -pedantic -fopenmp  -I/usr/include/python3.12 -I/home/bernard/dev/lm7/.venv/lib/python3.12/site-packages/torch/include -I/home/bernard/dev/lm7/.venv/lib/python3.12/site-packages/torch/include/torch/csrc/api/include   -mavx2 -mfma -mf16c  -o /tmp/torchinductor_bernard/cddhcpcdybzvymlvmkeivzsnk5lmiq7ice542tuiegeomvou7bn6/c4wqmy66imu2rwel25kslobl55e2fp7mtxpupqorurovw32mo5yi.wrapper.so  -ltorch -ltorch_cpu -lgomp  -L/usr/lib/x86_64-linux-gnu -L/home/bernard/dev/lm7/.venv/lib/python3.12/site-packages/torch/lib 

// Compile cmd
// g++ /tmp/torchinductor_bernard/cddhcpcdybzvymlvmkeivzsnk5lmiq7ice542tuiegeomvou7bn6/c4wqmy66imu2rwel25kslobl55e2fp7mtxpupqorurovw32mo5yi.wrapper.cpp -D TORCH_INDUCTOR_CPP_WRAPPER -D STANDALONE_TORCH_HEADER -D TORCH_INDUCTOR_PRECOMPILE_HEADERS -D  C10_USING_CUSTOM_GENERATED_MACROS -D CPU_CAPABILITY_AVX2  -O1 -DNDEBUG -fno-omit-frame-pointer -g1 -fno-trapping-math -funsafe-math-optimizations -ffinite-math-only -fno-signed-zeros -fno-finite-math-only -fno-unsafe-math-optimizations -fmath-errno -ffp-contract=off -fexcess-precision=fast -fno-tree-loop-vectorize -march=native -fPIC -Wall -std=c++20 -Wno-unused-variable -Wno-unknown-pragmas -pedantic -fopenmp  -include /tmp/torchinductor_bernard/precompiled_headers/cqbwvgtkgf24t42sn4u7vdxnbbdwtat6neb7jc2d5o4hjtih4row.h -I/usr/include/python3.12 -I/home/bernard/dev/lm7/.venv/lib/python3.12/site-packages/torch/include -I/home/bernard/dev/lm7/.venv/lib/python3.12/site-packages/torch/include/torch/csrc/api/include   -mavx2 -mfma -mf16c  -c -o /tmp/torchinductor_bernard/cddhcpcdybzvymlvmkeivzsnk5lmiq7ice542tuiegeomvou7bn6/c4wqmy66imu2rwel25kslobl55e2fp7mtxpupqorurovw32mo5yi.wrapper.o
// Link cmd
// g++ /tmp/torchinductor_bernard/cddhcpcdybzvymlvmkeivzsnk5lmiq7ice542tuiegeomvou7bn6/c4wqmy66imu2rwel25kslobl55e2fp7mtxpupqorurovw32mo5yi.wrapper.o /tmp/torchinductor_bernard/cddhcpcdybzvymlvmkeivzsnk5lmiq7ice542tuiegeomvou7bn6/cclnsjb5zbaads6kegi4qy5i2b6ixi2fzz4a3pu4i6qgf6closlx.kernel.o /tmp/torchinductor_bernard/cddhcpcdybzvymlvmkeivzsnk5lmiq7ice542tuiegeomvou7bn6/c4wqmy66imu2rwel25kslobl55e2fp7mtxpupqorurovw32mo5yi/ctrsxmmxgotwt2gcantv6qe5g4hugda73ji2v3ijip227636shte.weights.o -D TORCH_INDUCTOR_CPP_WRAPPER -D STANDALONE_TORCH_HEADER -D TORCH_INDUCTOR_PRECOMPILE_HEADERS -D  C10_USING_CUSTOM_GENERATED_MACROS -D CPU_CAPABILITY_AVX2  -O3 -DNDEBUG -fno-omit-frame-pointer -g1 -fno-trapping-math -funsafe-math-optimizations -ffinite-math-only -fno-signed-zeros -fno-finite-math-only -fno-unsafe-math-optimizations -fmath-errno -ffp-contract=off -fexcess-precision=fast -fno-tree-loop-vectorize -march=native -shared -fPIC -Wall -std=c++20 -Wno-unused-variable -Wno-unknown-pragmas -pedantic -fopenmp  -I/usr/include/python3.12 -I/home/bernard/dev/lm7/.venv/lib/python3.12/site-packages/torch/include -I/home/bernard/dev/lm7/.venv/lib/python3.12/site-packages/torch/include/torch/csrc/api/include   -mavx2 -mfma -mf16c  -o /tmp/torchinductor_bernard/cddhcpcdybzvymlvmkeivzsnk5lmiq7ice542tuiegeomvou7bn6/c4wqmy66imu2rwel25kslobl55e2fp7mtxpupqorurovw32mo5yi.wrapper.so  -ltorch -ltorch_cpu -lgomp  -L/usr/lib/x86_64-linux-gnu -L/home/bernard/dev/lm7/.venv/lib/python3.12/site-packages/torch/lib 
