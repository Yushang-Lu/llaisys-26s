#include "swiglu_nvidia.cuh"

#include "../../nvidia_common.cuh"

#include <cmath>

namespace llaisys::ops::nvidia {
namespace {

template <typename T>
__global__ void swigluKernel(
    T *out,
    const T *gate,
    const T *up,
    size_t numel) {
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < numel;
         index += static_cast<size_t>(blockDim.x) * gridDim.x) {
        const float gate_value = toFloat(gate[index]);
        const float silu = gate_value / (1.0f + expf(-gate_value));
        out[index] = fromFloat<T>(toFloat(up[index]) * silu);
    }
}

template <typename T>
void launchSwiGLU(
    std::byte *out,
    const std::byte *gate,
    const std::byte *up,
    size_t numel,
    cudaStream_t stream) {
    constexpr unsigned int threads = 256;
    swigluKernel<<<elementwiseBlocks(numel, threads), threads, 0, stream>>>(
        reinterpret_cast<T *>(out),
        reinterpret_cast<const T *>(gate),
        reinterpret_cast<const T *>(up),
        numel);
    device::nvidia::checkKernelLaunch("SwiGLU kernel launch");
}

} // namespace

void swiglu(
    std::byte *out,
    const std::byte *gate,
    const std::byte *up,
    llaisysDataType_t dtype,
    size_t numel,
    llaisysStream_t stream) {
    if (numel == 0) {
        return;
    }
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return launchSwiGLU<float>(out, gate, up, numel, cudaStream(stream));
    case LLAISYS_DTYPE_F16:
        return launchSwiGLU<__half>(out, gate, up, numel, cudaStream(stream));
    case LLAISYS_DTYPE_BF16:
        return launchSwiGLU<__nv_bfloat16>(out, gate, up, numel, cudaStream(stream));
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::nvidia
