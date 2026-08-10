#pragma once

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace llaisys::device::nvidia {

inline const char *cublasStatusString(cublasStatus_t status) {
    switch (status) {
    case CUBLAS_STATUS_SUCCESS:
        return "success";
    case CUBLAS_STATUS_NOT_INITIALIZED:
        return "not initialized";
    case CUBLAS_STATUS_ALLOC_FAILED:
        return "allocation failed";
    case CUBLAS_STATUS_INVALID_VALUE:
        return "invalid value";
    case CUBLAS_STATUS_ARCH_MISMATCH:
        return "architecture mismatch";
    case CUBLAS_STATUS_MAPPING_ERROR:
        return "mapping error";
    case CUBLAS_STATUS_EXECUTION_FAILED:
        return "execution failed";
    case CUBLAS_STATUS_INTERNAL_ERROR:
        return "internal error";
    case CUBLAS_STATUS_NOT_SUPPORTED:
        return "not supported";
    case CUBLAS_STATUS_LICENSE_ERROR:
        return "license error";
    default:
        return "unknown error";
    }
}

inline void checkCuda(cudaError_t status, const char *operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string("CUDA ") + operation + " failed: " +
            cudaGetErrorString(status));
    }
}

inline void checkCublas(cublasStatus_t status, const char *operation) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(
            std::string("cuBLAS ") + operation + " failed: " +
            cublasStatusString(status));
    }
}

inline void checkKernelLaunch(const char *operation) {
    checkCuda(cudaPeekAtLastError(), operation);
}

} // namespace llaisys::device::nvidia
