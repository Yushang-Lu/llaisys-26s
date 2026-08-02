#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/linear_cpu.hpp"

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_SAME_DEVICE(out, in, weight);
    if (bias != nullptr) {
        CHECK_SAME_DEVICE(out, bias);
    }
    ASSERT(in->ndim() == 2,
           "linear: in must be 2D");
    ASSERT(weight->ndim() == 2,
           "linear: weight must be 2D");
    ASSERT(out->ndim() == 2,
           "linear: out must be 2D");
    size_t batch = in->shape()[0];
    size_t in_features = in->shape()[1];
    size_t out_features = weight->shape()[0];
    ASSERT(weight->shape()[1] == in_features,
           "linear: weight shape[1] must equal in_features");
    ASSERT(out->shape()[0] == batch,
           "linear: out.shape[0] must equal batch");
    ASSERT(out->shape()[1] == out_features,
           "linear: out.shape[1] must equal out_features");
    if (bias != nullptr) {
        ASSERT(bias->ndim() == 1,
               "linear: bias must be 1D");
        ASSERT(bias->shape()[0] == out_features,
               "linear: bias size must equal out_features");
        ASSERT(out->dtype() == bias->dtype(),
               "linear: out and bias must have same dtype");
    }
    ASSERT(out->dtype() == in->dtype(),
           "linear: out and in must have same dtype");
    ASSERT(out->dtype() == weight->dtype(),
           "linear: out and weight must have same dtype");
    ASSERT(out->isContiguous() &&
           in->isContiguous() &&
           weight->isContiguous(),
           "linear: all tensors must be contiguous");
    if (bias != nullptr) {
        ASSERT(bias->isContiguous(),
               "linear: bias must be contiguous");
    }
    llaisysDataType_t dtype = out->dtype();
    ASSERT(dtype == LLAISYS_DTYPE_F32 ||
           dtype == LLAISYS_DTYPE_F16 ||
           dtype == LLAISYS_DTYPE_BF16,
           "linear: dtype must be F32, F16, or BF16");

    // always support cpu calculation
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::linear(
            in->data(),
            weight->data(),
            bias != nullptr ? bias->data() : nullptr,
            out->data(),
            dtype,
            batch,
            in_features,
            out_features);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::linear(
            in->data(),
            weight->data(),
            bias != nullptr ? bias->data() : nullptr,
            out->data(),
            dtype,
            batch,
            in_features,
            out_features);
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
