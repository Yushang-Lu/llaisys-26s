#pragma once

#include "llaisys.h"

#include <cstddef>
#include <cstdint>

namespace llaisys::ops::metax {
void embedding(
    std::byte *out,
    const int64_t *index,
    const std::byte *weight,
    llaisysDataType_t dtype,
    size_t num_indices,
    size_t num_embeddings,
    size_t hidden_size,
    llaisysStream_t stream);
}
