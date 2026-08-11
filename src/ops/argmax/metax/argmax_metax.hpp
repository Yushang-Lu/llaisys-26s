#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::metax {
void argmax(
    std::byte *max_idx,
    std::byte *max_val,
    const std::byte *vals,
    llaisysDataType_t dtype,
    size_t numel,
    int device_id,
    llaisysStream_t stream);
}
