#pragma once

#include "../device_resource.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstddef>

namespace llaisys::device::nvidia {
class Resource : public llaisys::device::DeviceResource {
private:
    cublasHandle_t _cublas = nullptr;
    float *_attention_scores = nullptr;
    size_t _attention_score_capacity = 0;

public:
    Resource(int device_id);
    ~Resource();

    cublasHandle_t cublas(cudaStream_t stream);
    float *attentionScores(size_t elements);
};

Resource &resource(int device_id);
} // namespace llaisys::device::nvidia
