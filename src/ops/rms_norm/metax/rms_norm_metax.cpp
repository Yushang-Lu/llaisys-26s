#include "rms_norm_metax.hpp"

#include "../../metax_common.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace llaisys::ops::metax {
namespace {

template <typename T>
__global__ void rmsNormKernel(
    T *out,
    const T *in,
    const T *weight,
    size_t hidden_size,
    float eps) {
    __shared__ float reduction[256];
    const size_t row = blockIdx.x;
    const size_t row_offset = row * hidden_size;

    float square_sum = 0.0f;
    for (size_t column = threadIdx.x;
         column < hidden_size;
         column += blockDim.x) {
        const float value = toFloat(in[row_offset + column]);
        square_sum += value * value;
    }
    reduction[threadIdx.x] = square_sum;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride != 0; stride /= 2) {
        if (threadIdx.x < stride) {
            reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        }
        __syncthreads();
    }

    const float inv_rms = rsqrtf(
        reduction[0] / static_cast<float>(hidden_size) + eps);
    for (size_t column = threadIdx.x;
         column < hidden_size;
         column += blockDim.x) {
        out[row_offset + column] = fromFloat<T>(
            toFloat(in[row_offset + column]) * inv_rms * toFloat(weight[column]));
    }
}

template <typename T>
void launchRmsNorm(
    std::byte *out,
    const std::byte *in,
    const std::byte *weight,
    size_t rows,
    size_t hidden_size,
    float eps,
    cudaStream_t stream) {
    if (rows > std::numeric_limits<unsigned int>::max()) {
        throw std::overflow_error("RMSNorm row count exceeds MetaX grid limit");
    }
    rmsNormKernel<<<static_cast<unsigned int>(rows), 256, 0, stream>>>(
        reinterpret_cast<T *>(out),
        reinterpret_cast<const T *>(in),
        reinterpret_cast<const T *>(weight),
        hidden_size,
        eps);
    device::metax::checkKernelLaunch("RMSNorm kernel launch");
}

} // namespace

void rms_norm(
    std::byte *out,
    const std::byte *in,
    const std::byte *weight,
    llaisysDataType_t dtype,
    size_t rows,
    size_t hidden_size,
    float eps,
    llaisysStream_t stream) {
    if (rows == 0) {
        return;
    }
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return launchRmsNorm<float>(out, in, weight, rows, hidden_size, eps, macaStream(stream));
    case LLAISYS_DTYPE_F16:
        return launchRmsNorm<__half>(out, in, weight, rows, hidden_size, eps, macaStream(stream));
    case LLAISYS_DTYPE_BF16:
        return launchRmsNorm<__nv_bfloat16>(out, in, weight, rows, hidden_size, eps, macaStream(stream));
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::metax
