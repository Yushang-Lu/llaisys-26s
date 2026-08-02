#include "self_attention_cpu.hpp"

#include "../../../utils.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

template <typename T>
void self_attention_(
    T *attn_val,
    const T *q,
    const T *k,
    const T *v,
    size_t q_len,
    size_t kv_len,
    size_t num_heads,
    size_t num_kv_heads,
    size_t head_dim,
    size_t value_dim,
    float scale) {
    const size_t query_heads_per_kv_head = num_heads / num_kv_heads;
    std::vector<float> weights(kv_len);

    for (size_t query_index = 0; query_index < q_len; ++query_index) {
        const size_t visible_kv_len = kv_len - q_len + query_index + 1;
        for (size_t head = 0; head < num_heads; ++head) {
            const size_t kv_head = head / query_heads_per_kv_head;
            const size_t q_offset = (query_index * num_heads + head) * head_dim;

            float max_score = -std::numeric_limits<float>::infinity();
            for (size_t key_index = 0; key_index < visible_kv_len; ++key_index) {
                const size_t k_offset = (key_index * num_kv_heads + kv_head) * head_dim;
                float score = 0.0f;
                for (size_t dim = 0; dim < head_dim; ++dim) {
                    score += llaisys::utils::cast<float>(q[q_offset + dim]) * llaisys::utils::cast<float>(k[k_offset + dim]);
                }
                score *= scale;
                weights[key_index] = score;
                max_score = std::max(max_score, score);
            }

            float weight_sum = 0.0f;
            for (size_t key_index = 0; key_index < visible_kv_len; ++key_index) {
                const float weight = std::exp(weights[key_index] - max_score);
                weights[key_index] = weight;
                weight_sum += weight;
            }

            const float inv_weight_sum = 1.0f / weight_sum;
            for (size_t key_index = 0; key_index < visible_kv_len; ++key_index) {
                weights[key_index] *= inv_weight_sum;
            }

            const size_t out_offset = (query_index * num_heads + head) * value_dim;
            for (size_t dim = 0; dim < value_dim; ++dim) {
                float value = 0.0f;
                for (size_t key_index = 0; key_index < visible_kv_len; ++key_index) {
                    const size_t v_offset = (key_index * num_kv_heads + kv_head) * value_dim;
                    value += weights[key_index] * llaisys::utils::cast<float>(v[v_offset + dim]);
                }
                attn_val[out_offset + dim] = llaisys::utils::cast<T>(value);
            }
        }
    }
}

namespace llaisys::ops::cpu {
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
    float scale) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return self_attention_(
            reinterpret_cast<float *>(attn_val),
            reinterpret_cast<const float *>(q),
            reinterpret_cast<const float *>(k),
            reinterpret_cast<const float *>(v),
            q_len,
            kv_len,
            num_heads,
            num_kv_heads,
            head_dim,
            value_dim,
            scale);
    case LLAISYS_DTYPE_BF16:
        return self_attention_(
            reinterpret_cast<llaisys::bf16_t *>(attn_val),
            reinterpret_cast<const llaisys::bf16_t *>(q),
            reinterpret_cast<const llaisys::bf16_t *>(k),
            reinterpret_cast<const llaisys::bf16_t *>(v),
            q_len,
            kv_len,
            num_heads,
            num_kv_heads,
            head_dim,
            value_dim,
            scale);
    case LLAISYS_DTYPE_F16:
        return self_attention_(
            reinterpret_cast<llaisys::fp16_t *>(attn_val),
            reinterpret_cast<const llaisys::fp16_t *>(q),
            reinterpret_cast<const llaisys::fp16_t *>(k),
            reinterpret_cast<const llaisys::fp16_t *>(v),
            q_len,
            kv_len,
            num_heads,
            num_kv_heads,
            head_dim,
            value_dim,
            scale);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu
