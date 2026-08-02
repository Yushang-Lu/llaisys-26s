#include "linear_cpu.hpp"

#include "../../../utils.hpp"

template <typename T>
void linear_(
    const T *in,
    const T *weight,
    const T *bias,
    T *out,
    size_t batch,
    size_t in_features,
    size_t out_features) {
    for (size_t i = 0; i < batch; ++i) {
        for (size_t j = 0; j < out_features; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < in_features; ++k) {
                sum += llaisys::utils::cast<float>(in[i * in_features + k])
                     * llaisys::utils::cast<float>(weight[j * in_features + k]);
            }
            if (bias != nullptr) {
                sum += llaisys::utils::cast<float>(bias[j]);
            }
            out[i * out_features + j] = llaisys::utils::cast<T>(sum);
        }
    }
}

namespace llaisys::ops::cpu {
void linear(
    const std::byte *in,
    const std::byte *weight,
    const std::byte *bias,
    std::byte *out,
    llaisysDataType_t dtype,
    size_t batch,
    size_t in_features,
    size_t out_features) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return linear_(
            reinterpret_cast<const float *>(in),
            reinterpret_cast<const float *>(weight),
            reinterpret_cast<const float *>(bias),
            reinterpret_cast<float *>(out),
            batch, in_features, out_features);
    case LLAISYS_DTYPE_BF16:
        return linear_(
            reinterpret_cast<const llaisys::bf16_t *>(in),
            reinterpret_cast<const llaisys::bf16_t *>(weight),
            reinterpret_cast<const llaisys::bf16_t *>(bias),
            reinterpret_cast<llaisys::bf16_t *>(out),
            batch, in_features, out_features);
    case LLAISYS_DTYPE_F16:
        return linear_(
            reinterpret_cast<const llaisys::fp16_t *>(in),
            reinterpret_cast<const llaisys::fp16_t *>(weight),
            reinterpret_cast<const llaisys::fp16_t *>(bias),
            reinterpret_cast<llaisys::fp16_t *>(out),
            batch, in_features, out_features);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu
