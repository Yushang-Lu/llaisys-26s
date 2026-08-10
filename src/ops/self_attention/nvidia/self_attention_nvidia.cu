#include "self_attention_nvidia.cuh"

#include "../../../device/nvidia/nvidia_resource.cuh"
#include "../../nvidia_common.cuh"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace llaisys::ops::nvidia {
namespace {

constexpr unsigned int kMaxThreads = 256;

unsigned int threadsFor(size_t size) {
    unsigned int threads = 32;
    while (threads < size && threads < kMaxThreads) {
        threads *= 2;
    }
    return threads;
}

template <typename T>
__global__ void attentionScoreKernel(
    float *scores,
    const T *q,
    const T *k,
    size_t q_len,
    size_t kv_len,
    size_t num_heads,
    size_t num_kv_heads,
    size_t head_dim,
    float scale) {
    __shared__ float reduction[kMaxThreads];
    const size_t score_index = blockIdx.x;
    const size_t key_index = score_index % kv_len;
    const size_t row = score_index / kv_len;
    const size_t head = row % num_heads;
    const size_t query_index = row / num_heads;
    const size_t visible_kv_len = kv_len - q_len + query_index + 1;

    if (key_index >= visible_kv_len) {
        if (threadIdx.x == 0) {
            scores[score_index] = -INFINITY;
        }
        return;
    }

    const size_t heads_per_kv = num_heads / num_kv_heads;
    const size_t kv_head = head / heads_per_kv;
    const size_t q_offset = (query_index * num_heads + head) * head_dim;
    const size_t k_offset = (key_index * num_kv_heads + kv_head) * head_dim;

    float dot = 0.0f;
    for (size_t dim = threadIdx.x; dim < head_dim; dim += blockDim.x) {
        dot += toFloat(q[q_offset + dim]) * toFloat(k[k_offset + dim]);
    }
    reduction[threadIdx.x] = dot;
    __syncthreads();

    for (unsigned int stride = blockDim.x / 2; stride != 0; stride /= 2) {
        if (threadIdx.x < stride) {
            reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        scores[score_index] = reduction[0] * scale;
    }
}

__global__ void attentionSoftmaxKernel(
    float *scores,
    size_t q_len,
    size_t kv_len,
    size_t num_heads) {
    __shared__ float reduction[kMaxThreads];
    const size_t row = blockIdx.x;
    const size_t query_index = row / num_heads;
    const size_t visible_kv_len = kv_len - q_len + query_index + 1;
    float *row_scores = scores + row * kv_len;

    float local_max = -INFINITY;
    for (size_t key = threadIdx.x; key < visible_kv_len; key += blockDim.x) {
        local_max = fmaxf(local_max, row_scores[key]);
    }
    reduction[threadIdx.x] = local_max;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride != 0; stride /= 2) {
        if (threadIdx.x < stride) {
            reduction[threadIdx.x] = fmaxf(
                reduction[threadIdx.x], reduction[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    const float max_score = reduction[0];

    float local_sum = 0.0f;
    for (size_t key = threadIdx.x; key < visible_kv_len; key += blockDim.x) {
        const float weight = expf(row_scores[key] - max_score);
        row_scores[key] = weight;
        local_sum += weight;
    }
    reduction[threadIdx.x] = local_sum;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride != 0; stride /= 2) {
        if (threadIdx.x < stride) {
            reduction[threadIdx.x] += reduction[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float inv_sum = 1.0f / reduction[0];
    for (size_t key = threadIdx.x; key < visible_kv_len; key += blockDim.x) {
        row_scores[key] *= inv_sum;
    }
    for (size_t key = visible_kv_len + threadIdx.x;
         key < kv_len;
         key += blockDim.x) {
        row_scores[key] = 0.0f;
    }
}

template <typename T>
__global__ void attentionValueKernel(
    T *attn_val,
    const float *scores,
    const T *v,
    size_t q_len,
    size_t kv_len,
    size_t num_heads,
    size_t num_kv_heads,
    size_t value_dim) {
    const size_t row = blockIdx.x;
    const size_t head = row % num_heads;
    const size_t query_index = row / num_heads;
    const size_t visible_kv_len = kv_len - q_len + query_index + 1;
    const size_t heads_per_kv = num_heads / num_kv_heads;
    const size_t kv_head = head / heads_per_kv;
    const float *row_scores = scores + row * kv_len;
    const size_t output_offset = row * value_dim;

    for (size_t dim = threadIdx.x; dim < value_dim; dim += blockDim.x) {
        float value = 0.0f;
        for (size_t key = 0; key < visible_kv_len; ++key) {
            const size_t value_offset =
                (key * num_kv_heads + kv_head) * value_dim + dim;
            value += row_scores[key] * toFloat(v[value_offset]);
        }
        attn_val[output_offset + dim] = fromFloat<T>(value);
    }
}

template <typename T>
void launchAttention(
    std::byte *attn_val,
    const std::byte *q,
    const std::byte *k,
    const std::byte *v,
    float *scores,
    size_t q_len,
    size_t kv_len,
    size_t num_heads,
    size_t num_kv_heads,
    size_t head_dim,
    size_t value_dim,
    float scale,
    cudaStream_t stream) {
    const size_t rows = q_len * num_heads;
    const size_t score_count = rows * kv_len;
    if (rows > std::numeric_limits<unsigned int>::max() ||
        score_count > std::numeric_limits<unsigned int>::max()) {
        throw std::overflow_error("self-attention dimensions exceed CUDA grid limits");
    }

    const unsigned int score_threads = threadsFor(head_dim);
    const unsigned int softmax_threads = threadsFor(kv_len);
    const unsigned int value_threads = threadsFor(value_dim);

    attentionScoreKernel<<<static_cast<unsigned int>(score_count), score_threads, 0, stream>>>(
        scores,
        reinterpret_cast<const T *>(q),
        reinterpret_cast<const T *>(k),
        q_len,
        kv_len,
        num_heads,
        num_kv_heads,
        head_dim,
        scale);
    device::nvidia::checkKernelLaunch("self-attention score kernel launch");

    attentionSoftmaxKernel<<<static_cast<unsigned int>(rows), softmax_threads, 0, stream>>>(
        scores, q_len, kv_len, num_heads);
    device::nvidia::checkKernelLaunch("self-attention softmax kernel launch");

    attentionValueKernel<<<static_cast<unsigned int>(rows), value_threads, 0, stream>>>(
        reinterpret_cast<T *>(attn_val),
        scores,
        reinterpret_cast<const T *>(v),
        q_len,
        kv_len,
        num_heads,
        num_kv_heads,
        value_dim);
    device::nvidia::checkKernelLaunch("self-attention value kernel launch");
}

} // namespace

void self_attention(
    std::byte *attn_val,
    const std::byte *q,
    const std::byte *k,
    const std::byte *v,
    llaisysDataType_t dtype,
    size_t q_len,
    size_t kv_len,
    size_t num_heads,
    size_t num_kv_heads,
    size_t head_dim,
    size_t value_dim,
    float scale,
    int device_id,
    llaisysStream_t stream) {
    if (q_len > std::numeric_limits<size_t>::max() / num_heads ||
        q_len * num_heads > std::numeric_limits<size_t>::max() / kv_len) {
        throw std::overflow_error("self-attention workspace size overflow");
    }
    const size_t score_count = q_len * num_heads * kv_len;
    float *scores = device::nvidia::resource(device_id).attentionScores(score_count);
    const cudaStream_t cuda_stream = cudaStream(stream);

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return launchAttention<float>(
            attn_val, q, k, v, scores,
            q_len, kv_len, num_heads, num_kv_heads,
            head_dim, value_dim, scale, cuda_stream);
    case LLAISYS_DTYPE_F16:
        return launchAttention<__half>(
            attn_val, q, k, v, scores,
            q_len, kv_len, num_heads, num_kv_heads,
            head_dim, value_dim, scale, cuda_stream);
    case LLAISYS_DTYPE_BF16:
        return launchAttention<__nv_bfloat16>(
            attn_val, q, k, v, scores,
            q_len, kv_len, num_heads, num_kv_heads,
            head_dim, value_dim, scale, cuda_stream);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::nvidia
