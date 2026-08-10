#include "embedding_nvidia.cuh"

#include "../../nvidia_common.cuh"

namespace llaisys::ops::nvidia {
namespace {

template <typename T>
__global__ void embeddingKernel(
    T *out,
    const int64_t *index,
    const T *weight,
    size_t num_indices,
    size_t num_embeddings,
    size_t hidden_size) {
    const size_t numel = num_indices * hidden_size;
    for (size_t linear = blockIdx.x * blockDim.x + threadIdx.x;
         linear < numel;
         linear += static_cast<size_t>(blockDim.x) * gridDim.x) {
        const size_t row = linear / hidden_size;
        const size_t column = linear % hidden_size;
        const int64_t embedding_index = index[row];
        if (embedding_index >= 0 &&
            static_cast<size_t>(embedding_index) < num_embeddings) {
            out[linear] = weight[static_cast<size_t>(embedding_index) * hidden_size + column];
        }
    }
}

template <typename T>
void launchEmbedding(
    std::byte *out,
    const int64_t *index,
    const std::byte *weight,
    size_t num_indices,
    size_t num_embeddings,
    size_t hidden_size,
    cudaStream_t stream) {
    const size_t numel = num_indices * hidden_size;
    constexpr unsigned int threads = 256;
    embeddingKernel<<<elementwiseBlocks(numel, threads), threads, 0, stream>>>(
        reinterpret_cast<T *>(out),
        index,
        reinterpret_cast<const T *>(weight),
        num_indices,
        num_embeddings,
        hidden_size);
    device::nvidia::checkKernelLaunch("embedding kernel launch");
}

} // namespace

void embedding(
    std::byte *out,
    const int64_t *index,
    const std::byte *weight,
    llaisysDataType_t dtype,
    size_t num_indices,
    size_t num_embeddings,
    size_t hidden_size,
    llaisysStream_t stream) {
    if (num_indices == 0 || hidden_size == 0) {
        return;
    }
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return launchEmbedding<float>(out, index, weight, num_indices, num_embeddings, hidden_size, cudaStream(stream));
    case LLAISYS_DTYPE_F16:
        return launchEmbedding<__half>(out, index, weight, num_indices, num_embeddings, hidden_size, cudaStream(stream));
    case LLAISYS_DTYPE_BF16:
        return launchEmbedding<__nv_bfloat16>(out, index, weight, num_indices, num_embeddings, hidden_size, cudaStream(stream));
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::nvidia
