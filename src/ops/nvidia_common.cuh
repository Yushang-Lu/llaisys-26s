#pragma once

#include "../device/nvidia/nvidia_common.cuh"
#include "../utils.hpp"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cstddef>

namespace llaisys::ops::nvidia {

template <typename T>
__device__ inline float toFloat(T value);

template <>
__device__ inline float toFloat<float>(float value) {
    return value;
}

template <>
__device__ inline float toFloat<__half>(__half value) {
    return __half2float(value);
}

template <>
__device__ inline float toFloat<__nv_bfloat16>(__nv_bfloat16 value) {
    return __bfloat162float(value);
}

template <typename T>
__device__ inline T fromFloat(float value);

template <>
__device__ inline float fromFloat<float>(float value) {
    return value;
}

template <>
__device__ inline __half fromFloat<__half>(float value) {
    return __float2half_rn(value);
}

template <>
__device__ inline __nv_bfloat16 fromFloat<__nv_bfloat16>(float value) {
    return __float2bfloat16_rn(value);
}

inline cudaStream_t cudaStream(llaisysStream_t stream) {
    return reinterpret_cast<cudaStream_t>(stream);
}

inline unsigned int elementwiseBlocks(size_t numel, unsigned int threads = 256) {
    const size_t needed = (numel + threads - 1) / threads;
    return static_cast<unsigned int>(std::min<size_t>(needed, 65535));
}

} // namespace llaisys::ops::nvidia
