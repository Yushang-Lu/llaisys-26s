#include "rope_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void rope_(
    T *out,
    const T *in,
    const int64_t *pos_ids,
    size_t seq_len,
    size_t num_heads,
    size_t head_dim,
    float theta) {
    const size_t half_dim = head_dim / 2;
    for (size_t i = 0; i < seq_len; ++i) {
        const float position = static_cast<float>(pos_ids[i]);
        for (size_t head = 0; head < num_heads; ++head) {
            const size_t vector_offset = (i * num_heads + head) * head_dim;
            for (size_t j = 0; j < half_dim; ++j) {
                const float exponent = 2.0f * static_cast<float>(j) / static_cast<float>(head_dim);
                const float angle = position / std::pow(theta, exponent);
                const float sin_value = std::sin(angle);
                const float cos_value = std::cos(angle);

                const size_t a_index = vector_offset + j;
                const size_t b_index = a_index + half_dim;
                const float a = llaisys::utils::cast<float>(in[a_index]);
                const float b = llaisys::utils::cast<float>(in[b_index]);
                out[a_index] = llaisys::utils::cast<T>(a * cos_value - b * sin_value);
                out[b_index] = llaisys::utils::cast<T>(b * cos_value + a * sin_value);
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rope(
    std::byte *out,
    const std::byte *in,
    const int64_t *pos_ids,
    llaisysDataType_t dtype,
    size_t seq_len,
    size_t num_heads,
    size_t head_dim,
    float theta) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return rope_(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(in),
            pos_ids, seq_len, num_heads, head_dim, theta);
    case LLAISYS_DTYPE_BF16:
        return rope_(
            reinterpret_cast<llaisys::bf16_t *>(out),
            reinterpret_cast<const llaisys::bf16_t *>(in),
            pos_ids, seq_len, num_heads, head_dim, theta);
    case LLAISYS_DTYPE_F16:
        return rope_(
            reinterpret_cast<llaisys::fp16_t *>(out),
            reinterpret_cast<const llaisys::fp16_t *>(in),
            pos_ids, seq_len, num_heads, head_dim, theta);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu
