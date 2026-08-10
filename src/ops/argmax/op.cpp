#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/argmax_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/argmax_nvidia.cuh"
#endif

namespace llaisys::ops {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    CHECK_SAME_DEVICE(max_idx, max_val, vals);
    ASSERT(vals->numel() > 0,
        "argmax: vals must not be empty");
    ASSERT(max_idx->ndim() == 1 && max_idx->numel() == 1,
        "argmax: max_idx must be shape[1]");
    ASSERT(max_val->ndim() == 1 && max_val->numel() == 1,
        "argmax: max_val must have shape[1]");
    ASSERT(vals->ndim() == 1,
        "argmax: vals must be a 1D tensor");
    ASSERT(max_idx->dtype() == LLAISYS_DTYPE_I64,
        "argmax: max_idx dtype must be int64");
    ASSERT(max_idx->isContiguous() &&
           max_val->isContiguous() &&
           vals->isContiguous(),
        "argmax: all tensors must be contiguous");

    llaisysDataType_t dtype = vals->dtype();
    ASSERT(dtype == LLAISYS_DTYPE_F32 ||
           dtype == LLAISYS_DTYPE_F16 ||
           dtype == LLAISYS_DTYPE_BF16,
        "argmax: vals dtype must be F32, F16 or BF16");
    ASSERT(max_val->dtype() == dtype,
        "argmax: max_val dtype must match vals dtype");

    // always support cpu calculation
    if (vals->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::argmax(
            max_idx->data(),
            max_val->data(),
            vals->data(),
            dtype,
            vals->numel());
    }

    llaisys::core::context().setDevice(
        vals->deviceType(), 
        vals->deviceId());

    switch (vals->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::argmax(
            max_idx->data(),
            max_val->data(),
            vals->data(),
            dtype,
            vals->numel());
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::argmax(
            max_idx->data(),
            max_val->data(),
            vals->data(),
            dtype,
            vals->numel(),
            vals->deviceId(),
            llaisys::core::context().runtime().stream());
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
