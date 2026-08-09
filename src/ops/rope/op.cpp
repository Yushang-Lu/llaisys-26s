#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rope_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/rope_nvidia.cuh"
#endif

#include <cmath>
#include <cstdint>

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in, pos_ids);
    ASSERT(in->ndim() == 3,
           "rope: in must be 3D");
    ASSERT(out->ndim() == 3,
           "rope: out must be 3D");
    ASSERT(pos_ids->ndim() == 1,
           "rope: pos_ids must be 1D");
    CHECK_SAME_SHAPE(out->shape(), in->shape());
    const size_t seq_len = in->shape()[0];
    const size_t num_heads = in->shape()[1];
    const size_t head_dim = in->shape()[2];
    ASSERT(pos_ids->numel() == seq_len,
           "rope: pos_ids size must equal sequence length");
    ASSERT(head_dim > 0 && head_dim % 2 == 0,
           "rope: head dimension must be positive and even");
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    ASSERT(pos_ids->dtype() == LLAISYS_DTYPE_I64,
           "rope: pos_ids dtype must be int64");
    ASSERT(out->isContiguous() && in->isContiguous() && pos_ids->isContiguous(),
           "rope: all tensors must be contiguous");
    ASSERT(std::isfinite(theta) && theta > 0.0f,
           "rope: theta must be finite and positive");
    const llaisysDataType_t dtype = out->dtype();
    ASSERT(dtype == LLAISYS_DTYPE_F32 || dtype == LLAISYS_DTYPE_F16 || dtype == LLAISYS_DTYPE_BF16,
           "rope: dtype must be F32, F16, or BF16");

    // always support cpu calculation
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rope(
            out->data(),
            in->data(),
            reinterpret_cast<const int64_t *>(pos_ids->data()),
            dtype,
            seq_len,
            num_heads,
            head_dim,
            theta);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rope(
            out->data(),
            in->data(),
            reinterpret_cast<const int64_t *>(pos_ids->data()),
            dtype,
            seq_len,
            num_heads,
            head_dim,
            theta);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::rope(
            out->data(),
            in->data(),
            reinterpret_cast<const int64_t *>(pos_ids->data()),
            dtype,
            seq_len,
            num_heads,
            head_dim,
            theta,
            llaisys::core::context().runtime().stream());
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
