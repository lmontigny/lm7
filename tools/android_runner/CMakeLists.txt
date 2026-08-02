# Minimal on-device runner for LM7 ExecuTorch artifacts.
#
# Built against an already-installed ExecuTorch. See
# docs/android-device-testing.md for the cross-compile invocation; the same
# CMakeLists builds a host binary if you point it at a host ExecuTorch install.

cmake_minimum_required(VERSION 3.19)
project(lm7_android_runner CXX)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

find_package(executorch CONFIG REQUIRED)
find_package(gflags REQUIRED)

add_executable(lm7_runner lm7_runner.cpp)

# Link the aggregate targets, not the individual libraries.
#
# Delegates and kernels register themselves from static initializers that
# nothing references by name, so they must be whole-archive linked or the
# runtime fails with "Backend XnnpackBackend is not registered". The installed
# executorch-config.cmake already sets that link option on every target that
# needs it -- adding it again here links each archive twice and produces
# hundreds of duplicate-symbol errors instead.
#
#   executorch_backends   -> xnnpack_backend
#   executorch_kernels    -> portable_ops_lib, portable_kernels
#   executorch_extensions -> extension_module_static, extension_tensor, loaders
target_link_libraries(
  lm7_runner
  executorch
  executorch_backends
  executorch_kernels
  executorch_extensions
  gflags
)
