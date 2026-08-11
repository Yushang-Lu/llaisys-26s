#include "../runtime_api.hpp"
#include "metax_common.hpp"

#include <cuda_runtime.h>

#include <stdexcept>

namespace llaisys::device::metax {
namespace runtime_api {

int getDeviceCount() {
    int count = 0;
    checkMaca(cudaGetDeviceCount(&count), "device count query");
    return count;
}

void setDevice(int device_id) {
    checkMaca(cudaSetDevice(device_id), "device selection");
}

void deviceSynchronize() {
    checkMaca(cudaDeviceSynchronize(), "device synchronization");
}

llaisysStream_t createStream() {
    cudaStream_t stream = nullptr;
    checkMaca(cudaStreamCreate(&stream), "stream creation");
    return reinterpret_cast<llaisysStream_t>(stream);
}

void destroyStream(llaisysStream_t stream) {
    if (stream != nullptr) {
        checkMaca(
            cudaStreamDestroy(reinterpret_cast<cudaStream_t>(stream)),
            "stream destruction");
    }
}

void streamSynchronize(llaisysStream_t stream) {
    checkMaca(
        cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)),
        "stream synchronization");
}

void *mallocDevice(size_t size) {
    if (size == 0) {
        return nullptr;
    }
    void *ptr = nullptr;
    checkMaca(cudaMalloc(&ptr, size), "device allocation");
    return ptr;
}

void freeDevice(void *ptr) {
    if (ptr != nullptr) {
        checkMaca(cudaFree(ptr), "device release");
    }
}

void *mallocHost(size_t size) {
    if (size == 0) {
        return nullptr;
    }
    void *ptr = nullptr;
    checkMaca(cudaMallocHost(&ptr, size), "pinned host allocation");
    return ptr;
}

void freeHost(void *ptr) {
    if (ptr != nullptr) {
        checkMaca(cudaFreeHost(ptr), "pinned host release");
    }
}

cudaMemcpyKind macaMemcpyKindFor(llaisysMemcpyKind_t kind) {
    switch (kind) {
    case LLAISYS_MEMCPY_H2H:
        return cudaMemcpyHostToHost;
    case LLAISYS_MEMCPY_H2D:
        return cudaMemcpyHostToDevice;
    case LLAISYS_MEMCPY_D2H:
        return cudaMemcpyDeviceToHost;
    case LLAISYS_MEMCPY_D2D:
        return cudaMemcpyDeviceToDevice;
    default:
        throw std::invalid_argument("invalid LLAISYS memcpy kind");
    }
}

void memcpySync(
    void *dst,
    const void *src,
    size_t size,
    llaisysMemcpyKind_t kind) {
    if (size == 0) {
        return;
    }
    checkMaca(
        cudaMemcpy(dst, src, size, macaMemcpyKindFor(kind)),
        "synchronous memory copy");
}

void memcpyAsync(
    void *dst,
    const void *src,
    size_t size,
    llaisysMemcpyKind_t kind,
    llaisysStream_t stream) {
    if (size == 0) {
        return;
    }
    checkMaca(
        cudaMemcpyAsync(
            dst,
            src,
            size,
            macaMemcpyKindFor(kind),
            reinterpret_cast<cudaStream_t>(stream)),
        "asynchronous memory copy");
}

static const LlaisysRuntimeAPI RUNTIME_API = {
    &getDeviceCount,
    &setDevice,
    &deviceSynchronize,
    &createStream,
    &destroyStream,
    &streamSynchronize,
    &mallocDevice,
    &freeDevice,
    &mallocHost,
    &freeHost,
    &memcpySync,
    &memcpyAsync};

} // namespace runtime_api

const LlaisysRuntimeAPI *getRuntimeAPI() {
    return &runtime_api::RUNTIME_API;
}

} // namespace llaisys::device::metax
