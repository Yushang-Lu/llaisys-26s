#include "embedding_cpu.hpp"

#include "../../../utils.hpp"

#include <cstring>

template <typename T>
void embedding_(
    T *out,
    const int64_t *index,
    const T *weight,
    size_t num_indices,
    size_t num_embeddings,
    size_t hidden_size) {
    const size_t row_bytes = hidden_size * sizeof(T);
    for (size_t i = 0; i < num_indices; ++i) {
        const int64_t idx = index[i];
        ASSERT(idx >= 0 && static_cast<size_t>(idx) < num_embeddings,
               "embedding: index out of range");
        const T *src = weight + static_cast<size_t>(idx) * hidden_size;
        T *dst = out + i * hidden_size;
        std::memcpy(dst, src, row_bytes);
    }
}

namespace llaisys::ops::cpu {
void embedding(
    std::byte *out,
    const int64_t *index,
    const std::byte *weight,
    llaisysDataType_t type,
    size_t num_indices,
    size_t num_embeddings,
    size_t hidden_size) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return embedding_(
            reinterpret_cast<float *>(out),
            index,
            reinterpret_cast<const float *>(weight),
            num_indices,
            num_embeddings,
            hidden_size);
    case LLAISYS_DTYPE_BF16:
        return embedding_(
            reinterpret_cast<llaisys::bf16_t *>(out),
            index,
            reinterpret_cast<const llaisys::bf16_t *>(weight),
            num_indices,
            num_embeddings,
            hidden_size);
    case LLAISYS_DTYPE_F16:
        return embedding_(
            reinterpret_cast<llaisys::fp16_t *>(out),
            index,
            reinterpret_cast<const llaisys::fp16_t *>(weight),
            num_indices,
            num_embeddings,
            hidden_size);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
