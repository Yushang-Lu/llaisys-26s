#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/self_attention_cpu.hpp"

#include <cmath>

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    ASSERT(q->ndim() == 3,
           "self_attention: q must be 3D");
    ASSERT(k->ndim() == 3,
           "self_attention: k must be 3D");
    ASSERT(v->ndim() == 3,
           "self_attention: v must be 3D");
    ASSERT(attn_val->ndim() == 3,
           "self_attention: attn_val must be 3D");
    const size_t q_len = q->shape()[0];
    const size_t num_heads = q->shape()[1];
    const size_t head_dim = q->shape()[2];
    const size_t kv_len = k->shape()[0];
    const size_t num_kv_heads = k->shape()[1];
    const size_t value_dim = v->shape()[2];
    ASSERT(q_len > 0,
           "self_attention: query length must be positive");
    ASSERT(kv_len >= q_len,
           "self_attention: kv length must be at least query length");
    ASSERT(num_heads > 0,
           "self_attention: query head count must be positive");
    ASSERT(num_kv_heads > 0,
           "self_attention: kv head count must be positive");
    ASSERT(head_dim > 0,
           "self_attention: head dimension must be positive");
    ASSERT(value_dim > 0,
           "self_attention: value dimension must be positive");
    ASSERT(k->shape()[0] == v->shape()[0],
           "self_attention: k and v sequence lengths must match");
    ASSERT(k->shape()[1] == v->shape()[1],
           "self_attention: k and v head counts must match");
    ASSERT(k->shape()[2] == head_dim,
           "self_attention: q and k head dimensions must match");
    ASSERT(attn_val->shape()[0] == q_len,
           "self_attention: output sequence length must match q");
    ASSERT(attn_val->shape()[1] == num_heads,
           "self_attention: output head count must match q");
    ASSERT(attn_val->shape()[2] == value_dim,
           "self_attention: output dimension must match v");
    ASSERT(num_heads % num_kv_heads == 0,
           "self_attention: query head count must be divisible by kv head count");
    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype(), k->dtype(), v->dtype());
    ASSERT(attn_val->isContiguous() && q->isContiguous() && k->isContiguous() && v->isContiguous(),
           "self_attention: all tensors must be contiguous");
    ASSERT(std::isfinite(scale),
           "self_attention: scale must be finite");
    const llaisysDataType_t dtype = attn_val->dtype();
    ASSERT(dtype == LLAISYS_DTYPE_F32 || dtype == LLAISYS_DTYPE_F16 || dtype == LLAISYS_DTYPE_BF16,
           "self_attention: dtype must be F32, F16, or BF16");

    // always support cpu calculation
    if (attn_val->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::self_attention(
            attn_val->data(),
            q->data(),
            k->data(),
            v->data(),
            dtype,
            q_len,
            kv_len,
            num_heads,
            num_kv_heads,
            head_dim,
            value_dim,
            scale);
    }

    llaisys::core::context().setDevice(attn_val->deviceType(), attn_val->deviceId());

    switch (attn_val->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::self_attention(
            attn_val->data(),
            q->data(),
            k->data(),
            v->data(),
            dtype,
            q_len,
            kv_len,
            num_heads,
            num_kv_heads,
            head_dim,
            value_dim,
            scale);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
