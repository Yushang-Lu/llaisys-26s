#include "llaisys/models/qwen2.h"

#include "llaisys_tensor.hpp"

#include "../models/qwen2/qwen2.hpp"
#include "../utils.hpp"

#include <exception>
#include <memory>
#include <string>
#include <vector>

namespace {

thread_local std::string qwen2_last_error;

void clear_qwen2_error() {
    qwen2_last_error.clear();
}

void set_qwen2_error(const std::exception &error) {
    qwen2_last_error = error.what();
}

void set_qwen2_error() {
    qwen2_last_error = "unknown native exception";
}

} // namespace

struct LlaisysQwen2Model {
    LlaisysQwen2Model(const LlaisysQwen2Meta &meta,
                      llaisysDeviceType_t device_type,
                      int device_id)
        : model(std::make_unique<llaisys::models::Qwen2Model>(
            meta, device_type, device_id)) {
        bindWeights();
    }

    llaisysTensor_t addHandle(const llaisys::tensor_t &tensor) {
        auto handle = std::make_unique<LlaisysTensor>();
        handle->tensor = tensor;
        tensor_handles.push_back(std::move(handle));
        return tensor_handles.back().get();
    }

    void bindWeights() {
        const auto &native_weights = model->weights();
        const size_t nlayer = native_weights.layers.size();

        tensor_handles.reserve(3 + nlayer * 12);
        attn_norm_w.resize(nlayer);
        attn_q_w.resize(nlayer);
        attn_q_b.resize(nlayer);
        attn_k_w.resize(nlayer);
        attn_k_b.resize(nlayer);
        attn_v_w.resize(nlayer);
        attn_v_b.resize(nlayer);
        attn_o_w.resize(nlayer);
        mlp_norm_w.resize(nlayer);
        mlp_gate_w.resize(nlayer);
        mlp_up_w.resize(nlayer);
        mlp_down_w.resize(nlayer);

        weights.in_embed = addHandle(native_weights.in_embed);
        weights.out_embed = addHandle(native_weights.out_embed);
        weights.out_norm_w = addHandle(native_weights.out_norm_w);

        for (size_t layer_index = 0; layer_index < nlayer; ++layer_index) {
            const auto &layer = native_weights.layers[layer_index];
            attn_norm_w[layer_index] = addHandle(layer.attn_norm_w);
            attn_q_w[layer_index] = addHandle(layer.attn_q_w);
            attn_q_b[layer_index] = addHandle(layer.attn_q_b);
            attn_k_w[layer_index] = addHandle(layer.attn_k_w);
            attn_k_b[layer_index] = addHandle(layer.attn_k_b);
            attn_v_w[layer_index] = addHandle(layer.attn_v_w);
            attn_v_b[layer_index] = addHandle(layer.attn_v_b);
            attn_o_w[layer_index] = addHandle(layer.attn_o_w);
            mlp_norm_w[layer_index] = addHandle(layer.mlp_norm_w);
            mlp_gate_w[layer_index] = addHandle(layer.mlp_gate_w);
            mlp_up_w[layer_index] = addHandle(layer.mlp_up_w);
            mlp_down_w[layer_index] = addHandle(layer.mlp_down_w);
        }

        weights.attn_norm_w = attn_norm_w.data();
        weights.attn_q_w = attn_q_w.data();
        weights.attn_q_b = attn_q_b.data();
        weights.attn_k_w = attn_k_w.data();
        weights.attn_k_b = attn_k_b.data();
        weights.attn_v_w = attn_v_w.data();
        weights.attn_v_b = attn_v_b.data();
        weights.attn_o_w = attn_o_w.data();
        weights.mlp_norm_w = mlp_norm_w.data();
        weights.mlp_gate_w = mlp_gate_w.data();
        weights.mlp_up_w = mlp_up_w.data();
        weights.mlp_down_w = mlp_down_w.data();
    }

    std::unique_ptr<llaisys::models::Qwen2Model> model;
    LlaisysQwen2Weights weights{};
    std::vector<std::unique_ptr<LlaisysTensor>> tensor_handles;
    std::vector<llaisysTensor_t> attn_norm_w;
    std::vector<llaisysTensor_t> attn_q_w;
    std::vector<llaisysTensor_t> attn_q_b;
    std::vector<llaisysTensor_t> attn_k_w;
    std::vector<llaisysTensor_t> attn_k_b;
    std::vector<llaisysTensor_t> attn_v_w;
    std::vector<llaisysTensor_t> attn_v_b;
    std::vector<llaisysTensor_t> attn_o_w;
    std::vector<llaisysTensor_t> mlp_norm_w;
    std::vector<llaisysTensor_t> mlp_gate_w;
    std::vector<llaisysTensor_t> mlp_up_w;
    std::vector<llaisysTensor_t> mlp_down_w;
};

__C {
    const char *llaisysQwen2GetLastError(void) {
        return qwen2_last_error.empty() ? nullptr : qwen2_last_error.c_str();
    }

    LlaisysQwen2Model *llaisysQwen2ModelCreate(
        const LlaisysQwen2Meta *meta,
        llaisysDeviceType_t device,
        int *device_ids,
        int ndevice) {
        clear_qwen2_error();
        try {
            CHECK_ARGUMENT(meta != nullptr,
                           "Qwen2 model metadata cannot be null");
            CHECK_ARGUMENT(
                device == LLAISYS_DEVICE_CPU ||
                    device == LLAISYS_DEVICE_NVIDIA ||
                    device == LLAISYS_DEVICE_METAX,
                "Qwen2 model supports CPU, NVIDIA, or MetaX devices");
            CHECK_ARGUMENT(ndevice == 1,
                           "Qwen2 model supports exactly one device");

            CHECK_ARGUMENT(device_ids != nullptr,
                           "Qwen2 device id list cannot be null");
            const int device_id = device_ids[0];

            return new LlaisysQwen2Model(*meta, device, device_id);
        } catch (const std::exception &error) {
            set_qwen2_error(error);
        } catch (...) {
            set_qwen2_error();
        }
        return nullptr;
    }

    void llaisysQwen2ModelDestroy(LlaisysQwen2Model * model) {
        clear_qwen2_error();
        try {
            delete model;
        } catch (const std::exception &error) {
            set_qwen2_error(error);
        } catch (...) {
            set_qwen2_error();
        }
    }

    LlaisysQwen2Weights *llaisysQwen2ModelWeights(
        LlaisysQwen2Model * model) {
        clear_qwen2_error();
        try {
            CHECK_ARGUMENT(model != nullptr,
                           "Qwen2 model cannot be null");
            return &model->weights;
        } catch (const std::exception &error) {
            set_qwen2_error(error);
        } catch (...) {
            set_qwen2_error();
        }
        return nullptr;
    }

    void llaisysQwen2ModelReset(LlaisysQwen2Model * model) {
        clear_qwen2_error();
        try {
            CHECK_ARGUMENT(model != nullptr,
                           "Qwen2 model cannot be null");
            model->model->reset();
        } catch (const std::exception &error) {
            set_qwen2_error(error);
        } catch (...) {
            set_qwen2_error();
        }
    }

    int64_t llaisysQwen2ModelInfer(
        LlaisysQwen2Model * model,
        int64_t * token_ids,
        size_t ntoken) {
        clear_qwen2_error();
        try {
            CHECK_ARGUMENT(model != nullptr,
                           "Qwen2 model cannot be null");
            return model->model->infer(token_ids, ntoken);
        } catch (const std::exception &error) {
            set_qwen2_error(error);
        } catch (...) {
            set_qwen2_error();
        }
        return -1;
    }
}
