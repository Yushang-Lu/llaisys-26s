#pragma once

#include "../device_resource.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstddef>

namespace llaisys::device::metax {
struct ArgmaxWorkspace {
    float *values;
    size_t *indices;
};

class Resource : public llaisys::device::DeviceResource {
private:
    cublasHandle_t _mcblas = nullptr;
    cudaStream_t _mcblas_stream = nullptr;
    bool _mcblas_stream_bound = false;
    float *_attention_scores = nullptr;
    size_t _attention_score_capacity = 0;
    std::byte *_argmax_workspace = nullptr;

public:
    Resource(int device_id);
    ~Resource();

    cublasHandle_t mcblas(cudaStream_t stream);
    float *attentionScores(size_t elements);
    ArgmaxWorkspace argmaxWorkspace(size_t elements);
};

Resource &resource(int device_id);
} // namespace llaisys::device::metax
