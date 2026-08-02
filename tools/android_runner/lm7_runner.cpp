/*
 * A minimal on-device runner for LM7 ExecuTorch artifacts.
 *
 * ExecuTorch's example_runner validates a BundledProgram, which is the right
 * tool for a small model and the wrong one for a real model. Two limits force
 * this runner to exist:
 *
 *   - BundledProgram serialization goes .pte -> JSON -> flatc. A 622 MB
 *     SmolLM2 .pte produces a JSON intermediate that aborts flatc, so a
 *     language model cannot be bundled at all.
 *   - example_runner reports outputs by logging one line per element. A
 *     transformer's logits are hundreds of thousands of values.
 *
 * This runner takes a plain .pte, reads its inputs from raw binary files, and
 * writes its outputs to a raw binary file. Shapes and dtypes come from the
 * method metadata inside the .pte, so the host does not have to describe them.
 * It also repeats the forward pass in-process, which is what makes a steady
 * state measurable at all -- example_runner has no iteration flag, so repeats
 * there measure process startup and, over a network-attached device, the link.
 *
 * Build it with the ExecuTorch cross-compile described in
 * docs/android-device-testing.md.
 */

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

#include <gflags/gflags.h>

#include <executorch/extension/module/module.h>
#include <executorch/extension/tensor/tensor.h>

using executorch::extension::Module;
using executorch::extension::make_tensor_ptr;
using executorch::extension::TensorPtr;
using executorch::runtime::EValue;
using executorch::runtime::Error;

DEFINE_string(model_path, "model.pte", "Path to the .pte to execute.");
DEFINE_string(
    inputs,
    "",
    "Comma-separated raw tensor files, one per method input, in order.");
DEFINE_string(
    output_path,
    "output.bin",
    "Where to write the raw bytes of output 0.");
DEFINE_int32(repeats, 1, "Timed forward passes to run.");
DEFINE_int32(warmup, 0, "Untimed forward passes to run first.");

namespace {

std::vector<std::string> split(const std::string& value, char separator) {
  std::vector<std::string> parts;
  std::string current;
  for (char c : value) {
    if (c == separator) {
      if (!current.empty()) {
        parts.push_back(current);
      }
      current.clear();
    } else {
      current += c;
    }
  }
  if (!current.empty()) {
    parts.push_back(current);
  }
  return parts;
}

bool read_file(const std::string& path, std::vector<uint8_t>& into) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    fprintf(stderr, "cannot open input file %s\n", path.c_str());
    return false;
  }
  const std::streamsize size = file.tellg();
  file.seekg(0, std::ios::beg);
  into.resize(static_cast<size_t>(size));
  return static_cast<bool>(file.read(reinterpret_cast<char*>(into.data()), size));
}

} // namespace

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  Module module(FLAGS_model_path);
  if (module.load_method("forward") != Error::Ok) {
    fprintf(stderr, "failed to load forward from %s\n", FLAGS_model_path.c_str());
    return 1;
  }

  const auto meta = module.method_meta("forward");
  if (!meta.ok()) {
    fprintf(stderr, "failed to read method metadata\n");
    return 1;
  }

  const auto paths = split(FLAGS_inputs, ',');
  const size_t expected = meta->num_inputs();
  if (paths.size() != expected) {
    fprintf(
        stderr,
        "method takes %zu inputs but %zu files were given\n",
        expected,
        paths.size());
    return 1;
  }

  // The buffers must outlive the tensors that point into them, and must not
  // reallocate while tensors hold their addresses.
  std::vector<std::vector<uint8_t>> buffers;
  buffers.reserve(expected);
  std::vector<EValue> values;
  std::vector<TensorPtr> tensors;
  tensors.reserve(expected);

  for (size_t i = 0; i < expected; ++i) {
    const auto info = meta->input_tensor_meta(i);
    if (!info.ok()) {
      fprintf(stderr, "input %zu is not a tensor\n", i);
      return 1;
    }
    buffers.emplace_back();
    if (!read_file(paths[i], buffers.back())) {
      return 1;
    }
    if (buffers.back().size() != info->nbytes()) {
      fprintf(
          stderr,
          "input %zu: file has %zu bytes, method wants %zu\n",
          i,
          buffers.back().size(),
          info->nbytes());
      return 1;
    }
    std::vector<executorch::aten::SizesType> sizes(
        info->sizes().begin(), info->sizes().end());
    tensors.push_back(make_tensor_ptr(
        std::move(sizes), buffers.back().data(), info->scalar_type()));
    values.emplace_back(tensors.back());
  }

  for (int i = 0; i < FLAGS_warmup; ++i) {
    if (!module.forward(values).ok()) {
      fprintf(stderr, "warmup forward failed\n");
      return 1;
    }
  }

  std::vector<EValue> outputs;
  for (int i = 0; i < FLAGS_repeats; ++i) {
    const auto started = std::chrono::steady_clock::now();
    auto result = module.forward(values);
    const auto elapsed = std::chrono::steady_clock::now() - started;
    if (!result.ok()) {
      fprintf(stderr, "forward failed on iteration %d\n", i);
      return 1;
    }
    printf(
        "iter_ms %.4f\n",
        std::chrono::duration<double, std::milli>(elapsed).count());
    outputs = std::move(*result);
  }

  if (outputs.empty() || !outputs[0].isTensor()) {
    fprintf(stderr, "method produced no tensor output\n");
    return 1;
  }

  const auto tensor = outputs[0].toTensor();
  printf("output_numel %zu\n", static_cast<size_t>(tensor.numel()));
  printf("output_nbytes %zu\n", static_cast<size_t>(tensor.nbytes()));
  printf("output_dtype %d\n", static_cast<int>(tensor.scalar_type()));
  printf("output_outputs %zu\n", outputs.size());

  std::ofstream out(FLAGS_output_path, std::ios::binary);
  if (!out) {
    fprintf(stderr, "cannot open %s for writing\n", FLAGS_output_path.c_str());
    return 1;
  }
  out.write(
      reinterpret_cast<const char*>(tensor.const_data_ptr()), tensor.nbytes());
  if (!out) {
    fprintf(stderr, "failed writing %s\n", FLAGS_output_path.c_str());
    return 1;
  }
  printf("ok\n");
  return 0;
}
