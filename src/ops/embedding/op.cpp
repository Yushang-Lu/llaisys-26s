#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/embedding_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/embedding_nvidia.cuh"
#endif

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_SAME_DEVICE(out, index, weight);
    ASSERT(index->ndim() == 1,
           "embedding: index must be 1D");
    ASSERT(index->dtype() == LLAISYS_DTYPE_I64,
           "embedding: index must be int64");
    ASSERT(weight->ndim() == 2,
           "embedding: weight must be 2D");
    ASSERT(out->ndim() == 2,
           "embedding: out must be 2D");
    ASSERT(out->shape()[0] == index->numel(),
           "embedding: out.shape[0] must equal index.numel()");
    ASSERT(out->shape()[1] == weight->shape()[1],
           "embedding: hidden size must match weight");
    ASSERT(out->dtype() == weight->dtype(),
           "embedding: out and weight must have same dtype");
    ASSERT(out->isContiguous() &&
           index->isContiguous() &&
           weight->isContiguous(),
           "embedding: all tensors must be contiguous");
    llaisysDataType_t dtype = out->dtype();
    ASSERT(dtype == LLAISYS_DTYPE_F32 ||
           dtype == LLAISYS_DTYPE_F16 ||
           dtype == LLAISYS_DTYPE_BF16,
           "embedding: dtype must be F32, F16, or BF16");

    // always support cpu calculation
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::embedding(
            out->data(),
            reinterpret_cast<const int64_t *>(index->data()),
            weight->data(),
            dtype,
            index->numel(),
            weight->shape()[0],
            weight->shape()[1]);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
		return cpu::embedding(
            out->data(),
            reinterpret_cast<const int64_t *>(index->data()),
            weight->data(),
            dtype,
            index->numel(),
            weight->shape()[0],
            weight->shape()[1]);
#ifdef ENABLE_NVIDIA_API
	case LLAISYS_DEVICE_NVIDIA:
        return nvidia::embedding(
            out->data(),
            reinterpret_cast<const int64_t *>(index->data()),
            weight->data(),
            dtype,
            index->numel(),
            weight->shape()[0],
            weight->shape()[1],
            llaisys::core::context().runtime().stream());
#endif
    default:
       EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
