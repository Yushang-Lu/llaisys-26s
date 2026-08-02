#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void linear(
    const std::byte *in,
    const std::byte *weight,
    const std::byte *bias,
    std::byte *out,
    llaisysDataType_t dtype,
    size_t batch,
    size_t in_features,
    size_t out_features);

} // namespace llaisys::ops::cpu
