#include "nvidia_resource.cuh"
#include "nvidia_common.cuh"

#include <limits>
#include <memory>
#include <stdexcept>
#include <unordered_map>

namespace llaisys::device::nvidia {
namespace {

constexpr size_t kInitialAttentionScoreCapacity = 4096;
constexpr size_t kArgmaxWorkspaceCapacity = 256;

} // namespace

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
    if (_argmax_workspace != nullptr) {
        cudaFree(_argmax_workspace);
        _argmax_workspace = nullptr;
    }
    if (_cublas != nullptr) {
        cublasDestroy(_cublas);
        _cublas = nullptr;
    }
}

cublasHandle_t Resource::cublas(cudaStream_t stream) {
    if (!_cublas_stream_bound || _cublas_stream != stream) {
        checkCublas(cublasSetStream(_cublas, stream), "stream binding");
        _cublas_stream = stream;
        _cublas_stream_bound = true;
    }
    return _cublas;
}

float *Resource::attentionScores(size_t elements) {
    if (elements > std::numeric_limits<size_t>::max() / sizeof(float)) {
        throw std::overflow_error("attention workspace size overflow");
    }
    if (elements <= _attention_score_capacity) {
        return _attention_scores;
    }

    const size_t max_capacity =
        std::numeric_limits<size_t>::max() / sizeof(float);
    size_t new_capacity = _attention_score_capacity == 0
                            ? kInitialAttentionScoreCapacity
                            : _attention_score_capacity;
    while (new_capacity < elements) {
        if (new_capacity > max_capacity / 2) {
            new_capacity = elements;
            break;
        }
        new_capacity *= 2;
    }

    float *new_scores = nullptr;
    checkCuda(
        cudaMalloc(reinterpret_cast<void **>(&new_scores),
                   new_capacity * sizeof(float)),
        "attention workspace allocation");
    if (_attention_scores != nullptr) {
        checkCuda(cudaFree(_attention_scores), "attention workspace release");
    }
    _attention_scores = new_scores;
    _attention_score_capacity = new_capacity;
    return _attention_scores;
}

ArgmaxWorkspace Resource::argmaxWorkspace(size_t elements) {
    if (elements == 0 || elements > kArgmaxWorkspaceCapacity) {
        throw std::invalid_argument("argmax workspace element count is invalid");
    }
    if (_argmax_workspace == nullptr) {
        constexpr size_t workspace_bytes =
            kArgmaxWorkspaceCapacity * (sizeof(float) + sizeof(size_t));
        void *allocation = nullptr;
        checkCuda(
            cudaMalloc(&allocation, workspace_bytes),
            "argmax workspace allocation");
        _argmax_workspace = static_cast<std::byte *>(allocation);
    }

    return ArgmaxWorkspace{
        reinterpret_cast<float *>(_argmax_workspace),
        reinterpret_cast<size_t *>(
            _argmax_workspace +
            kArgmaxWorkspaceCapacity * sizeof(float))};
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
