#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/swiglu_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/swiglu_nvidia.cuh"
#endif

namespace llaisys::ops {
void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    CHECK_SAME_DEVICE(out, gate, up);
    ASSERT(out->ndim() == 2,
           "swiglu: out must be 2D");
    ASSERT(gate->ndim() == 2,
           "swiglu: gate must be 2D");
    ASSERT(up->ndim() == 2,
           "swiglu: up must be 2D");
    CHECK_SAME_SHAPE(out->shape(), gate->shape(), up->shape());
    CHECK_SAME_DTYPE(out->dtype(), gate->dtype(), up->dtype());
    ASSERT(out->isContiguous() && gate->isContiguous() && up->isContiguous(),
           "swiglu: all tensors must be contiguous");
    const llaisysDataType_t dtype = out->dtype();
    ASSERT(dtype == LLAISYS_DTYPE_F32 || dtype == LLAISYS_DTYPE_F16 || dtype == LLAISYS_DTYPE_BF16,
           "swiglu: dtype must be F32, F16, or BF16");

    // always support cpu calculation
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::swiglu(
            out->data(),
            gate->data(),
            up->data(),
            dtype,
            out->numel());
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::swiglu(
            out->data(),
            gate->data(),
            up->data(),
            dtype,
            out->numel());
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::swiglu(
            out->data(),
            gate->data(),
            up->data(),
            dtype,
            out->numel(),
            llaisys::core::context().runtime().stream());
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
