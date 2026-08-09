#include "linear_nvidia.cuh"

#include "../../../device/nvidia/nvidia_resource.cuh"
#include "../../nvidia_common.cuh"

#include <climits>
#include <stdexcept>

namespace llaisys::ops::nvidia {
namespace {

cudaDataType_t cudaDataType(llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return CUDA_R_32F;
    case LLAISYS_DTYPE_F16:
        return CUDA_R_16F;
    case LLAISYS_DTYPE_BF16:
        return CUDA_R_16BF;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

template <typename T>
__global__ void addBiasKernel(
    T *out,
    const T *bias,
    size_t numel,
    size_t out_features) {
    for (size_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < numel;
         index += static_cast<size_t>(blockDim.x) * gridDim.x) {
        out[index] = fromFloat<T>(
            toFloat(out[index]) + toFloat(bias[index % out_features]));
    }
}

template <typename T>
void launchBias(
    std::byte *out,
    const std::byte *bias,
    size_t numel,
    size_t out_features,
    cudaStream_t stream) {
    constexpr unsigned int threads = 256;
    addBiasKernel<<<elementwiseBlocks(numel, threads), threads, 0, stream>>>(
        reinterpret_cast<T *>(out),
        reinterpret_cast<const T *>(bias),
        numel,
        out_features);
    device::nvidia::checkKernelLaunch("linear bias kernel launch");
}

void addBias(
    std::byte *out,
    const std::byte *bias,
    llaisysDataType_t dtype,
    size_t numel,
    size_t out_features,
    cudaStream_t stream) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return launchBias<float>(out, bias, numel, out_features, stream);
    case LLAISYS_DTYPE_F16:
        return launchBias<__half>(out, bias, numel, out_features, stream);
    case LLAISYS_DTYPE_BF16:
        return launchBias<__nv_bfloat16>(out, bias, numel, out_features, stream);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace

void linear(
    const std::byte *in,
    const std::byte *weight,
    const std::byte *bias,
    std::byte *out,
    llaisysDataType_t dtype,
    size_t batch,
    size_t in_features,
    size_t out_features,
    int device_id,
    llaisysStream_t stream) {
    if (batch == 0 || out_features == 0) {
        return;
    }
    if (batch > INT_MAX || in_features > INT_MAX || out_features > INT_MAX) {
        throw std::overflow_error("linear dimensions exceed cuBLAS integer limits");
    }

    const cudaDataType_t data_type = cudaDataType(dtype);
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const int m = static_cast<int>(out_features);
    const int n = static_cast<int>(batch);
    const int k = static_cast<int>(in_features);
    const cudaStream_t cuda_stream = cudaStream(stream);
    cublasHandle_t handle = device::nvidia::resource(device_id).cublas(cuda_stream);
    const cublasGemmAlgo_t algorithm =
        dtype == LLAISYS_DTYPE_F32
            ? CUBLAS_GEMM_DEFAULT
            : CUBLAS_GEMM_DEFAULT_TENSOR_OP;

    device::nvidia::checkCublas(
        cublasGemmEx(
            handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            m,
            n,
            k,
            &alpha,
            weight,
            data_type,
            k,
            in,
            data_type,
            k,
            &beta,
            out,
            data_type,
            m,
            CUBLAS_COMPUTE_32F,
            algorithm),
        "linear GEMM");

    if (bias != nullptr) {
        addBias(out, bias, dtype, batch * out_features, out_features, cuda_stream);
    }
}

} // namespace llaisys::ops::nvidia
