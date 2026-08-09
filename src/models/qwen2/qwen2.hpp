#pragma once

#include "llaisys/models/qwen2.h"

#include "../../tensor/tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace llaisys::models {

class Qwen2WorkerPool;
class Qwen2Workspace;

struct Qwen2LayerWeights {
    tensor_t attn_norm_w;
    tensor_t attn_q_w;
    tensor_t attn_q_b;
    tensor_t attn_k_w;
    tensor_t attn_k_b;
    tensor_t attn_v_w;
    tensor_t attn_v_b;
    tensor_t attn_o_w;
    tensor_t mlp_norm_w;
    tensor_t mlp_gate_w;
    tensor_t mlp_up_w;
    tensor_t mlp_down_w;
};

struct Qwen2Weights {
    tensor_t in_embed;
    tensor_t out_embed;
    tensor_t out_norm_w;
    std::vector<Qwen2LayerWeights> layers;
};

class Qwen2Model {
public:
    Qwen2Model(const LlaisysQwen2Meta &meta,
               llaisysDeviceType_t device_type,
               int device_id);

    Qwen2Model(const Qwen2Model &) = delete;
    Qwen2Model &operator=(const Qwen2Model &) = delete;
    Qwen2Model(Qwen2Model &&) = delete;
    Qwen2Model &operator=(Qwen2Model &&) = delete;
    ~Qwen2Model();

    Qwen2Weights &weights();
    const Qwen2Weights &weights() const;

    // token_ids is a block of newly appended tokens.
    // The method returns the greedy next-token prediction for the final token in that block.
    int64_t infer(const int64_t *token_ids, size_t ntoken);
    void reset();

private:
    tensor_t createTensor(const std::vector<size_t> &shape,
                          llaisysDataType_t dtype) const;
    void createWeights();
    void ensureCacheCapacity(size_t required_capacity);
    void ensureWorkspaceCapacity(size_t required_capacity);
    void prepareRope(const std::vector<int64_t> &position_ids);
    void linearBf16(const tensor_t &out,
                    const tensor_t &in,
                    const tensor_t &weight,
                    const tensor_t &bias);
    void ropeBf16(const tensor_t &out,
                  const tensor_t &in,
                  const tensor_t &position_ids,
                  size_t num_heads);
    void copyTensorData(const tensor_t &out, const tensor_t &in) const;

    LlaisysQwen2Meta _meta;
    llaisysDeviceType_t _device_type;
    int _device_id;
    Qwen2Weights _weights;
    std::unique_ptr<Qwen2WorkerPool> _workers;
    std::unique_ptr<Qwen2Workspace> _workspace;
    size_t _workspace_capacity = 0;

    std::vector<int64_t> _position_values;
    std::vector<float> _rope_denominators;
    std::vector<float> _rope_sin;
    std::vector<float> _rope_cos;

    // Cache entries have shape [capacity, num_kv_heads, head_dim].
    // Keys are stored after RoPE; values are stored directly from the value projection.
    std::vector<tensor_t> _key_cache;
    std::vector<tensor_t> _value_cache;
    size_t _cache_length = 0;
    size_t _cache_capacity = 0;
};

} // namespace llaisys::models
