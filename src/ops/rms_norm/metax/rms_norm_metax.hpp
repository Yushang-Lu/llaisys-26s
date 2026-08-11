#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::metax {
void rms_norm(
    std::byte *out,
    const std::byte *in,
    const std::byte *weight,
    llaisysDataType_t dtype,
    size_t rows,
    size_t hidden_size,
    float eps,
    llaisysStream_t stream);
}
