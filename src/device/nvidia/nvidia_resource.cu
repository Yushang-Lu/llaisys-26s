#include "nvidia_resource.cuh"
#include "nvidia_common.cuh"

#include <limits>
#include <memory>
#include <stdexcept>
#include <unordered_map>

namespace llaisys::device::nvidia {

Resource::Resource(int device_id)
    : llaisys::device::DeviceResource(LLAISYS_DEVICE_NVIDIA, device_id) {
    checkCuda(cudaSetDevice(device_id), "set device for resource creation");
    checkCublas(cublasCreate(&_cublas), "handle creation");
}

Resource::~Resource() {
    cudaSetDevice(getDeviceId());
    if (_attention_scores != nullptr) {
        cudaFree(_attention_scores);
        _attention_scores = nullptr;
    }
    if (_cublas != nullptr) {
        cublasDestroy(_cublas);
        _cublas = nullptr;
    }
}

cublasHandle_t Resource::cublas(cudaStream_t stream) {
    checkCublas(cublasSetStream(_cublas, stream), "stream binding");
    return _cublas;
}

float *Resource::attentionScores(size_t elements) {
    if (elements > std::numeric_limits<size_t>::max() / sizeof(float)) {
        throw std::overflow_error("attention workspace size overflow");
    }
    if (elements <= _attention_score_capacity) {
        return _attention_scores;
    }

    if (_attention_scores != nullptr) {
        checkCuda(cudaFree(_attention_scores), "attention workspace release");
        _attention_scores = nullptr;
        _attention_score_capacity = 0;
    }

    if (elements != 0) {
        checkCuda(
            cudaMalloc(reinterpret_cast<void **>(&_attention_scores),
                       elements * sizeof(float)),
            "attention workspace allocation");
        _attention_score_capacity = elements;
    }
    return _attention_scores;
}

Resource &resource(int device_id) {
    thread_local std::unordered_map<int, std::unique_ptr<Resource>> resources;
    auto &entry = resources[device_id];
    if (!entry) {
        entry = std::make_unique<Resource>(device_id);
    }
    return *entry;
}

} // namespace llaisys::device::nvidia
