#include "argmax_nvidia.cuh"

#include "../../../device/nvidia/nvidia_resource.cuh"
#include "../../nvidia_common.cuh"

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace llaisys::ops::nvidia {
namespace {

constexpr unsigned int kThreads = 256;
constexpr size_t kSingleBlockMaxElements = 4096;
constexpr unsigned int kMaxStageOneBlocks = 256;

__device__ bool betterValue(
    float candidate_value,
    size_t candidate_index,
    float current_value,
    size_t current_index) {
    const bool candidate_nan = isnan(candidate_value);
    const bool current_nan = isnan(current_value);
    if (candidate_nan != current_nan) {
        return candidate_nan;
    }
    if (candidate_nan) {
        return candidate_index < current_index;
    }
    return candidate_value > current_value ||
           (candidate_value == current_value && candidate_index < current_index);
}

__device__ __forceinline__ void reduceBlock(
    float *values,
    size_t *indices) {
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
}

template <typename T>
__global__ void argmaxSingleBlockKernel(
    int64_t *max_idx,
    T *max_val,
    const T *vals,
    size_t numel) {
    __shared__ float values[kThreads];
    __shared__ size_t indices[kThreads];

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
    reduceBlock(values, indices);

    if (threadIdx.x == 0) {
        *max_idx = static_cast<int64_t>(indices[0]);
        *max_val = vals[indices[0]];
    }
}

template <typename T>
__global__ void argmaxStageOneKernel(
    float *partial_values,
    size_t *partial_indices,
    const T *vals,
    size_t numel) {
    __shared__ float values[kThreads];
    __shared__ size_t indices[kThreads];

    float local_value = -INFINITY;
    size_t local_index = numel;
    const size_t start =
        static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride =
        static_cast<size_t>(blockDim.x) * gridDim.x;
    for (size_t index = start; index < numel; index += stride) {
        const float value = toFloat(vals[index]);
        if (betterValue(value, index, local_value, local_index)) {
            local_value = value;
            local_index = index;
        }
    }

    values[threadIdx.x] = local_value;
    indices[threadIdx.x] = local_index;
    reduceBlock(values, indices);

    if (threadIdx.x == 0) {
        partial_values[blockIdx.x] = values[0];
        partial_indices[blockIdx.x] = indices[0];
    }
}

template <typename T>
__global__ void argmaxStageTwoKernel(
    int64_t *max_idx,
    T *max_val,
    const T *vals,
    const float *partial_values,
    const size_t *partial_indices,
    size_t partial_count) {
    __shared__ float values[kThreads];
    __shared__ size_t indices[kThreads];

    float local_value = -INFINITY;
    size_t local_index = static_cast<size_t>(-1);
    if (threadIdx.x < partial_count) {
        local_value = partial_values[threadIdx.x];
        local_index = partial_indices[threadIdx.x];
    }
    values[threadIdx.x] = local_value;
    indices[threadIdx.x] = local_index;
    reduceBlock(values, indices);

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
    int device_id,
    cudaStream_t stream) {
    if (numel <= kSingleBlockMaxElements) {
        argmaxSingleBlockKernel<<<1, kThreads, 0, stream>>>(
            reinterpret_cast<int64_t *>(max_idx),
            reinterpret_cast<T *>(max_val),
            reinterpret_cast<const T *>(vals),
            numel);
        device::nvidia::checkKernelLaunch("argmax kernel launch");
        return;
    }

    const size_t needed_blocks = (numel - 1) / kThreads + 1;
    const unsigned int blocks = static_cast<unsigned int>(
        std::min<size_t>(needed_blocks, kMaxStageOneBlocks));
    const auto workspace =
        device::nvidia::resource(device_id).argmaxWorkspace(blocks);
    argmaxStageOneKernel<<<blocks, kThreads, 0, stream>>>(
        workspace.values,
        workspace.indices,
        reinterpret_cast<const T *>(vals),
        numel);
    device::nvidia::checkKernelLaunch("argmax stage-one kernel launch");

    argmaxStageTwoKernel<<<1, kThreads, 0, stream>>>(
        reinterpret_cast<int64_t *>(max_idx),
        reinterpret_cast<T *>(max_val),
        reinterpret_cast<const T *>(vals),
        workspace.values,
        workspace.indices,
        blocks);
    device::nvidia::checkKernelLaunch("argmax stage-two kernel launch");
}

} // namespace

void argmax(
    std::byte *max_idx,
    std::byte *max_val,
    const std::byte *vals,
    llaisysDataType_t dtype,
    size_t numel,
    int device_id,
    llaisysStream_t stream) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return launchArgmax<float>(
            max_idx, max_val, vals, numel, device_id, cudaStream(stream));
    case LLAISYS_DTYPE_F16:
        return launchArgmax<__half>(
            max_idx, max_val, vals, numel, device_id, cudaStream(stream));
    case LLAISYS_DTYPE_BF16:
        return launchArgmax<__nv_bfloat16>(
            max_idx, max_val, vals, numel, device_id, cudaStream(stream));
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::nvidia
