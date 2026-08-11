#include "rope_metax.hpp"

#include "../../metax_common.hpp"

#include <cmath>

namespace llaisys::ops::metax {
namespace {

template <typename T>
__global__ void ropeKernel(
    T *out,
    const T *in,
    const int64_t *pos_ids,
    size_t total_pairs,
    size_t num_heads,
    size_t head_dim,
    float theta) {
    const size_t half_dim = head_dim / 2;
    for (size_t linear = blockIdx.x * blockDim.x + threadIdx.x;
         linear < total_pairs;
         linear += static_cast<size_t>(blockDim.x) * gridDim.x) {
        const size_t feature = linear % half_dim;
        const size_t vector_index = linear / half_dim;
        const size_t token = vector_index / num_heads;
        const size_t first = vector_index * head_dim + feature;
        const size_t second = first + half_dim;

        const float exponent =
            2.0f * static_cast<float>(feature) / static_cast<float>(head_dim);
        const float angle = static_cast<float>(pos_ids[token]) /
                            powf(theta, exponent);
        float sin_value = 0.0f;
        float cos_value = 0.0f;
        sincosf(angle, &sin_value, &cos_value);

        const float a = toFloat(in[first]);
        const float b = toFloat(in[second]);
        out[first] = fromFloat<T>(a * cos_value - b * sin_value);
        out[second] = fromFloat<T>(b * cos_value + a * sin_value);
    }
}

template <typename T>
void launchRope(
    std::byte *out,
    const std::byte *in,
    const int64_t *pos_ids,
    size_t seq_len,
    size_t num_heads,
    size_t head_dim,
    float theta,
    cudaStream_t stream) {
    const size_t total_pairs = seq_len * num_heads * (head_dim / 2);
    constexpr unsigned int threads = 256;
    ropeKernel<<<elementwiseBlocks(total_pairs, threads), threads, 0, stream>>>(
        reinterpret_cast<T *>(out),
        reinterpret_cast<const T *>(in),
        pos_ids,
        total_pairs,
        num_heads,
        head_dim,
        theta);
    device::metax::checkKernelLaunch("RoPE kernel launch");
}

} // namespace

void rope(
    std::byte *out,
    const std::byte *in,
    const int64_t *pos_ids,
    llaisysDataType_t dtype,
    size_t seq_len,
    size_t num_heads,
    size_t head_dim,
    float theta,
    llaisysStream_t stream) {
    if (seq_len == 0 || num_heads == 0) {
        return;
    }
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return launchRope<float>(out, in, pos_ids, seq_len, num_heads, head_dim, theta, macaStream(stream));
    case LLAISYS_DTYPE_F16:
        return launchRope<__half>(out, in, pos_ids, seq_len, num_heads, head_dim, theta, macaStream(stream));
    case LLAISYS_DTYPE_BF16:
        return launchRope<__nv_bfloat16>(out, in, pos_ids, seq_len, num_heads, head_dim, theta, macaStream(stream));
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::metax
