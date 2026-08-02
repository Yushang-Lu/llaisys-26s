#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rms_norm_cpu.hpp"

#include <cmath>

namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    CHECK_SAME_DEVICE(out, in, weight);
    ASSERT(in->ndim() == 2,
           "rms_norm: in must be 2D");
    ASSERT(out->ndim() == 2,
           "rms_norm: out must be 2D");
    ASSERT(weight->ndim() == 1,
           "rms_norm: weight must be 1D");
    CHECK_SAME_SHAPE(out->shape(), in->shape());
    const size_t rows = in->shape()[0];
    const size_t hidden_size = in->shape()[1];
    ASSERT(hidden_size > 0,
           "rms_norm: hidden size must be positive");
    ASSERT(weight->shape()[0] == hidden_size,
           "rms_norm: weight size must equal hidden size");
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(),
           "rms_norm: all tensors must be contiguous");
    ASSERT(std::isfinite(eps) && eps > 0.0f,
           "rms_norm: eps must be finite and positive");
    const llaisysDataType_t dtype = out->dtype();
    ASSERT(dtype == LLAISYS_DTYPE_F32 || dtype == LLAISYS_DTYPE_F16 || dtype == LLAISYS_DTYPE_BF16,
           "rms_norm: dtype must be F32, F16, or BF16");

    // always support cpu calculation
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rms_norm(
            out->data(),
            in->data(),
            weight->data(),
            dtype,
            rows,
            hidden_size,
            eps);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rms_norm(
            out->data(),
            in->data(),
            weight->data(),
            dtype,
            rows,
            hidden_size,
            eps);
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
