#include "rms_norm_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void rms_norm_(
    T *out,
    const T *in,
    const T *weight,
    size_t rows,
    size_t hidden_size,
    float eps) {
    for (size_t i = 0; i < rows; ++i) {
        const size_t row_offset = i * hidden_size;
        float square_sum = 0.0f;
        for (size_t j = 0; j < hidden_size; ++j) {
            const float value = llaisys::utils::cast<float>(in[row_offset + j]);
            square_sum += value * value;
        }

        const float mean_square = square_sum / static_cast<float>(hidden_size);
        const float inv_rms = 1.0f / std::sqrt(mean_square + eps);
        for (size_t j = 0; j < hidden_size; ++j) {
            const float value = llaisys::utils::cast<float>(in[row_offset + j]);
            const float scale = llaisys::utils::cast<float>(weight[j]);
            out[row_offset + j] = llaisys::utils::cast<T>(value * inv_rms * scale);
        }
    }
}

namespace llaisys::ops::cpu {
void rms_norm(
    std::byte *out,
    const std::byte *in,
    const std::byte *weight,
    llaisysDataType_t dtype,
    size_t rows,
    size_t hidden_size,
    float eps) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_(
            reinterpret_cast<float *>(out),
            reinterpret_cast<const float *>(in),
            reinterpret_cast<const float *>(weight),
            rows, hidden_size, eps);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_(
            reinterpret_cast<llaisys::bf16_t *>(out),
            reinterpret_cast<const llaisys::bf16_t *>(in),
            reinterpret_cast<const llaisys::bf16_t *>(weight),
            rows, hidden_size, eps);
    case LLAISYS_DTYPE_F16:
        return rms_norm_(
            reinterpret_cast<llaisys::fp16_t *>(out),
            reinterpret_cast<const llaisys::fp16_t *>(in),
            reinterpret_cast<const llaisys::fp16_t *>(weight),
            rows, hidden_size, eps);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu
