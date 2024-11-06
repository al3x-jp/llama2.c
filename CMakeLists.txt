cmake_minimum_required(VERSION 3.22.1)

set(CMAKE_CXX_COMPILER "${DPCPP_HOME}/llvm/build/bin/clang++")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")

if(CUDA_BACKEND)
	# Build for CUDA BACK-END (requires an NVIDIA CARD)
	message(STATUS "Building for CUDA back-end.")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O3 -Wno-writable-strings -fsycl -fsycl-targets=nvptx64-nvidia-cuda --cuda-path=${CUDA_TOOLKIT}")
else()
	# Build Native Platform
    message(STATUS "Building for native OpencCL back-end.")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O3 -fsycl -fno-sycl-instrument-device-code")
endif()

project(run)
add_executable(run run.c run_sycl.cpp)
target_compile_definitions(run PRIVATE USE_SYCL=1)
target_link_libraries(run m)