#include "argmax_nvidia.cuh"

#include "../../nvidia_common.cuh"

#include <cmath>
#include <cstdint>

namespace llaisys::ops::nvidia {
namespace {

__device__ bool betterValue(
    float candidate_value,
    size_t candidate_index,
    float current_value,
    size_t current_index) {
    return candidate_value > current_value ||
           (candidate_value == current_value && candidate_index < current_index);
}

template <typename T>
__global__ void argmaxKernel(
    int64_t *max_idx,
    T *max_val,
    const T *vals,
    size_t numel) {
    __shared__ float values[256];
    __shared__ size_t indices[256];

    float local_value = -INFINITY;
    size_t local_index = numel;
    for (size_t index = threadIdx.x;
         index < numel;
         index += blockDim.x) {
        const float value = toFloat(vals[index]);
        if (betterValue(value, index, local_value, local_index)) {
            local_value = value;
            local_index = index;
        }
    }

    values[threadIdx.x] = local_value;
    indices[threadIdx.x] = local_index;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride != 0; stride /= 2) {
        if (threadIdx.x < stride &&
            betterValue(
                values[threadIdx.x + stride],
                indices[threadIdx.x + stride],
                values[threadIdx.x],
                indices[threadIdx.x])) {
            values[threadIdx.x] = values[threadIdx.x + stride];
            indices[threadIdx.x] = indices[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        *max_idx = static_cast<int64_t>(indices[0]);
        *max_val = vals[indices[0]];
    }
}

template <typename T>
void launchArgmax(
    std::byte *max_idx,
    std::byte *max_val,
    const std::byte *vals,
    size_t numel,
    cudaStream_t stream) {
    argmaxKernel<<<1, 256, 0, stream>>>(
        reinterpret_cast<int64_t *>(max_idx),
        reinterpret_cast<T *>(max_val),
        reinterpret_cast<const T *>(vals),
        numel);
    device::nvidia::checkKernelLaunch("argmax kernel launch");
}

} // namespace

void argmax(
    std::byte *max_idx,
    std::byte *max_val,
    const std::byte *vals,
    llaisysDataType_t dtype,
    size_t numel,
    llaisysStream_t stream) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return launchArgmax<float>(max_idx, max_val, vals, numel, cudaStream(stream));
    case LLAISYS_DTYPE_F16:
        return launchArgmax<__half>(max_idx, max_val, vals, numel, cudaStream(stream));
    case LLAISYS_DTYPE_BF16:
        return launchArgmax<__nv_bfloat16>(max_idx, max_val, vals, numel, cudaStream(stream));
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::nvidia
