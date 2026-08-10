#include "add_nvidia.cuh"

#include "../../nvidia_common.cuh"

namespace llaisys::ops::nvidia {
namespace {

template <typename T>
__global__ void addKernel(T *c, const T *a, const T *b, size_t numel) {
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < numel;
         index += static_cast<size_t>(blockDim.x) * gridDim.x) {
        c[index] = fromFloat<T>(toFloat(a[index]) + toFloat(b[index]));
    }
}

template <typename T>
void launchAdd(
    std::byte *c,
    const std::byte *a,
    const std::byte *b,
    size_t numel,
    cudaStream_t stream) {
    constexpr unsigned int threads = 256;
    addKernel<<<elementwiseBlocks(numel, threads), threads, 0, stream>>>(
        reinterpret_cast<T *>(c),
        reinterpret_cast<const T *>(a),
        reinterpret_cast<const T *>(b),
        numel);
    device::nvidia::checkKernelLaunch("add kernel launch");
}

} // namespace

void add(
    std::byte *c,
    const std::byte *a,
    const std::byte *b,
    llaisysDataType_t dtype,
    size_t numel,
    llaisysStream_t stream) {
    if (numel == 0) {
        return;
    }
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return launchAdd<float>(c, a, b, numel, cudaStream(stream));
    case LLAISYS_DTYPE_F16:
        return launchAdd<__half>(c, a, b, numel, cudaStream(stream));
    case LLAISYS_DTYPE_BF16:
        return launchAdd<__nv_bfloat16>(c, a, b, numel, cudaStream(stream));
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::nvidia
