#include "qwen2.hpp"

#include "../../ops/add/op.hpp"
#include "../../ops/argmax/op.hpp"
#include "../../ops/embedding/op.hpp"
#include "../../ops/linear/op.hpp"
#include "../../ops/rms_norm/op.hpp"
#include "../../ops/rope/op.hpp"
#include "../../ops/self_attention/op.hpp"
#include "../../ops/swiglu/op.hpp"
#include "../../utils.hpp"

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace llaisys::models {

class Qwen2WorkerPool {
public:
    using WorkFunction = void (*)(void *, size_t, size_t);

    Qwen2WorkerPool() {
        const unsigned int hardware_threads = std::thread::hardware_concurrency();
        const unsigned int available_threads = hardware_threads == 0 ? 4U : hardware_threads;
        _thread_count = std::max<size_t>(1, std::min<unsigned int>(4U, available_threads));

        _threads.reserve(_thread_count - 1);
        for (size_t worker = 1; worker < _thread_count; ++worker) {
            _threads.emplace_back([this, worker]() {
                workerLoop(worker);
            });
        }
    }

    Qwen2WorkerPool(const Qwen2WorkerPool &) = delete;
    Qwen2WorkerPool &operator=(const Qwen2WorkerPool &) = delete;

    ~Qwen2WorkerPool() {
        {
            std::lock_guard<std::mutex> lock(_mutex);
            _stopping = true;
        }
        _work_available.notify_all();
        for (auto &thread : _threads) {
            thread.join();
        }
    }

    void run(size_t total, WorkFunction work, void *context) {
        if (total == 0) {
            return;
        }
        CHECK_ARGUMENT(work != nullptr,
                       "Qwen2 worker function cannot be null");

        {
            std::lock_guard<std::mutex> lock(_mutex);
            _work = work;
            _context = context;
            _total = total;
            _completed_workers = 0;
            ++_generation;
        }
        _work_available.notify_all();

        executeChunk(0, total, work, context);

        std::unique_lock<std::mutex> lock(_mutex);
        _work_complete.wait(lock, [this]() {
            return _completed_workers == _threads.size();
        });
        _work = nullptr;
        _context = nullptr;
        _total = 0;
    }

private:
    void executeChunk(size_t worker,
                      size_t total,
                      WorkFunction work,
                      void *context) const {
        const size_t begin = total * worker / _thread_count;
        const size_t end = total * (worker + 1) / _thread_count;
        if (begin != end) {
            work(context, begin, end);
        }
    }

    void workerLoop(size_t worker) {
        size_t observed_generation = 0;
        while (true) {
            WorkFunction work = nullptr;
            void *context = nullptr;
            size_t total = 0;

            {
                std::unique_lock<std::mutex> lock(_mutex);
                _work_available.wait(lock, [this, observed_generation]() {
                    return _stopping || _generation != observed_generation;
                });
                if (_stopping) {
                    return;
                }

                observed_generation = _generation;
                work = _work;
                context = _context;
                total = _total;
            }

            executeChunk(worker, total, work, context);

            {
                std::lock_guard<std::mutex> lock(_mutex);
                ++_completed_workers;
            }
            _work_complete.notify_one();
        }
    }

    size_t _thread_count = 1;
    std::vector<std::thread> _threads;
    std::mutex _mutex;
    std::condition_variable _work_available;
    std::condition_variable _work_complete;
    WorkFunction _work = nullptr;
    void *_context = nullptr;
    size_t _total = 0;
    size_t _generation = 0;
    size_t _completed_workers = 0;
    bool _stopping = false;
};

class Qwen2Workspace {
public:
    tensor_t input_ids;
    tensor_t position_ids;
    tensor_t hidden_a;
    tensor_t hidden_b;
    tensor_t attention_input;
    tensor_t query_projection;
    tensor_t key_projection;
    tensor_t rotated_query;
    tensor_t attention_values;
    tensor_t attention_output;
    tensor_t post_attention;
    tensor_t mlp_input;
    tensor_t gate;
    tensor_t up;
    tensor_t activated;
    tensor_t mlp_output;
    tensor_t normalized;
    tensor_t logits;
    tensor_t max_index;
    tensor_t max_value;
    tensor_t host_max_index;
};

namespace {

inline float bf16ToFloat(bf16_t value) noexcept {
    const uint32_t bits = static_cast<uint32_t>(value._v) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

inline bf16_t floatToBf16(float value) noexcept {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t rounding_bias = 0x00007FFFU + ((bits >> 16) & 1U);
    return bf16_t{
        static_cast<uint16_t>((bits + rounding_bias) >> 16)};
}

struct Bf16LinearContext {
    const bf16_t *input;
    const bf16_t *weight;
    const bf16_t *bias;
    bf16_t *output;
    size_t batch;
    size_t in_features;
    size_t out_features;
};

void runBf16Linear(void *opaque, size_t begin, size_t end) {
    const auto &context = *static_cast<const Bf16LinearContext *>(opaque);
    constexpr size_t kOutputGroup = 4;

    size_t column = begin;
    while (column < end) {
        const bool full_group = end - column >= kOutputGroup;
        if (full_group) {
            const bf16_t *weight_0 = context.weight + column * context.in_features;
            const bf16_t *weight_1 = weight_0 + context.in_features;
            const bf16_t *weight_2 = weight_1 + context.in_features;
            const bf16_t *weight_3 = weight_2 + context.in_features;
            const float bias_0 = context.bias == nullptr
                                   ? 0.0f
                                   : bf16ToFloat(context.bias[column]);
            const float bias_1 = context.bias == nullptr
                                   ? 0.0f
                                   : bf16ToFloat(context.bias[column + 1]);
            const float bias_2 = context.bias == nullptr
                                   ? 0.0f
                                   : bf16ToFloat(context.bias[column + 2]);
            const float bias_3 = context.bias == nullptr
                                   ? 0.0f
                                   : bf16ToFloat(context.bias[column + 3]);

            for (size_t row = 0; row < context.batch; ++row) {
                const bf16_t *input_row = context.input + row * context.in_features;
                float sum_0 = 0.0f;
                float sum_1 = 0.0f;
                float sum_2 = 0.0f;
                float sum_3 = 0.0f;

                for (size_t feature = 0;
                     feature < context.in_features;
                     ++feature) {
                    const float input_value = bf16ToFloat(input_row[feature]);
                    sum_0 += input_value * bf16ToFloat(weight_0[feature]);
                    sum_1 += input_value * bf16ToFloat(weight_1[feature]);
                    sum_2 += input_value * bf16ToFloat(weight_2[feature]);
                    sum_3 += input_value * bf16ToFloat(weight_3[feature]);
                }

                if (context.bias != nullptr) {
                    sum_0 += bias_0;
                    sum_1 += bias_1;
                    sum_2 += bias_2;
                    sum_3 += bias_3;
                }
                const size_t output_offset = row * context.out_features + column;
                context.output[output_offset] = floatToBf16(sum_0);
                context.output[output_offset + 1] = floatToBf16(sum_1);
                context.output[output_offset + 2] = floatToBf16(sum_2);
                context.output[output_offset + 3] = floatToBf16(sum_3);
            }
            column += kOutputGroup;
            continue;
        }

        const bf16_t *weight_row = context.weight + column * context.in_features;
        const float bias_value = context.bias == nullptr
                                   ? 0.0f
                                   : bf16ToFloat(context.bias[column]);
        for (size_t row = 0; row < context.batch; ++row) {
            const bf16_t *input_row = context.input + row * context.in_features;
            float sum = 0.0f;
            for (size_t feature = 0;
                 feature < context.in_features;
                 ++feature) {
                sum += bf16ToFloat(input_row[feature]) * bf16ToFloat(weight_row[feature]);
            }
            if (context.bias != nullptr) {
                sum += bias_value;
            }
            context.output[row * context.out_features + column] = floatToBf16(sum);
        }
        ++column;
    }
}
struct Bf16RopeContext {
    const bf16_t *input;
    bf16_t *output;
    const float *sin_values;
    const float *cos_values;
    size_t num_heads;
    size_t head_dim;
};

void runBf16Rope(void *opaque, size_t begin, size_t end) {
    const auto &context = *static_cast<const Bf16RopeContext *>(opaque);
    const size_t half_dim = context.head_dim / 2;

    for (size_t vector_index = begin;
         vector_index < end;
         ++vector_index) {
        const size_t token = vector_index / context.num_heads;
        const size_t vector_offset = vector_index * context.head_dim;
        const size_t table_offset = token * half_dim;

        for (size_t feature = 0; feature < half_dim; ++feature) {
            const size_t first = vector_offset + feature;
            const size_t second = first + half_dim;
            const float a = bf16ToFloat(context.input[first]);
            const float b = bf16ToFloat(context.input[second]);
            const float sin_value = context.sin_values[table_offset + feature];
            const float cos_value = context.cos_values[table_offset + feature];

            context.output[first] = floatToBf16(a * cos_value - b * sin_value);
            context.output[second] = floatToBf16(b * cos_value + a * sin_value);
        }
    }
}

constexpr size_t kInitialCacheCapacity = 256;

void validateMeta(const LlaisysQwen2Meta &meta) {
    CHECK_ARGUMENT(meta.dtype == LLAISYS_DTYPE_BF16,
                   "Qwen2 inference only supports BF16 weights");
    CHECK_ARGUMENT(meta.nlayer > 0, "Qwen2 must have at least one layer");
    CHECK_ARGUMENT(meta.hs > 0, "Qwen2 hidden size must be positive");
    CHECK_ARGUMENT(meta.nh > 0, "Qwen2 attention head count must be positive");
    CHECK_ARGUMENT(meta.nkvh > 0, "Qwen2 key/value head count must be positive");
    CHECK_ARGUMENT(meta.dh > 0, "Qwen2 head dimension must be positive");
    CHECK_ARGUMENT(meta.di > 0, "Qwen2 intermediate size must be positive");
    CHECK_ARGUMENT(meta.maxseq > 0, "Qwen2 maximum sequence length must be positive");
    CHECK_ARGUMENT(meta.voc > 0, "Qwen2 vocabulary size must be positive");
    CHECK_ARGUMENT(meta.nh % meta.nkvh == 0,
                   "Qwen2 attention heads must be divisible by key/value heads");
    CHECK_ARGUMENT(meta.nh <= std::numeric_limits<size_t>::max() / meta.dh,
                   "Qwen2 attention dimensions overflow");
    CHECK_ARGUMENT(meta.hs == meta.nh * meta.dh,
                   "Qwen2 hidden size must equal num_heads * head_dim");
    CHECK_ARGUMENT(std::isfinite(meta.epsilon) && meta.epsilon > 0.0f,
                   "Qwen2 RMS norm epsilon must be finite and positive");
    CHECK_ARGUMENT(std::isfinite(meta.theta) && meta.theta > 0.0f,
                   "Qwen2 RoPE theta must be finite and positive");
    CHECK_ARGUMENT(meta.end_token >= 0 && static_cast<size_t>(meta.end_token) < meta.voc,
                   "Qwen2 EOS token must be within the vocabulary");
}

} // namespace

Qwen2Model::Qwen2Model(const LlaisysQwen2Meta &meta,
                       llaisysDeviceType_t device_type,
                       int device_id)
    : _meta(meta), _device_type(device_type), _device_id(device_id) {
    validateMeta(_meta);
    CHECK_ARGUMENT(
        _device_type == LLAISYS_DEVICE_CPU ||
            _device_type == LLAISYS_DEVICE_NVIDIA,
        "Qwen2 model supports CPU or NVIDIA devices");
    llaisys::core::context().setDevice(_device_type, _device_id);

    if (_device_type == LLAISYS_DEVICE_CPU) {
        _workers = std::make_unique<Qwen2WorkerPool>();
        _rope_denominators.resize(_meta.dh / 2);
        for (size_t feature = 0; feature < _rope_denominators.size(); ++feature) {
            const float exponent = 2.0f * static_cast<float>(feature) / static_cast<float>(_meta.dh);
            _rope_denominators[feature] = std::pow(_meta.theta, exponent);
        }
    }

    createWeights();
}

Qwen2Model::~Qwen2Model() = default;

Qwen2Weights &Qwen2Model::weights() {
    return _weights;
}

const Qwen2Weights &Qwen2Model::weights() const {
    return _weights;
}

tensor_t Qwen2Model::createTensor(const std::vector<size_t> &shape,
                                  llaisysDataType_t dtype) const {
    return Tensor::create(shape, dtype, _device_type, _device_id);
}

void Qwen2Model::createWeights() {
    const size_t query_size = _meta.nh * _meta.dh;
    const size_t key_value_size = _meta.nkvh * _meta.dh;

    _weights.in_embed = createTensor({_meta.voc, _meta.hs}, _meta.dtype);
    _weights.out_embed = createTensor({_meta.voc, _meta.hs}, _meta.dtype);
    _weights.out_norm_w = createTensor({_meta.hs}, _meta.dtype);
    _weights.layers.resize(_meta.nlayer);

    for (auto &layer : _weights.layers) {
        layer.attn_norm_w = createTensor({_meta.hs}, _meta.dtype);
        layer.attn_q_w = createTensor({query_size, _meta.hs}, _meta.dtype);
        layer.attn_q_b = createTensor({query_size}, _meta.dtype);
        layer.attn_k_w = createTensor({key_value_size, _meta.hs}, _meta.dtype);
        layer.attn_k_b = createTensor({key_value_size}, _meta.dtype);
        layer.attn_v_w = createTensor({key_value_size, _meta.hs}, _meta.dtype);
        layer.attn_v_b = createTensor({key_value_size}, _meta.dtype);
        layer.attn_o_w = createTensor({_meta.hs, query_size}, _meta.dtype);
        layer.mlp_norm_w = createTensor({_meta.hs}, _meta.dtype);
        layer.mlp_gate_w = createTensor({_meta.di, _meta.hs}, _meta.dtype);
        layer.mlp_up_w = createTensor({_meta.di, _meta.hs}, _meta.dtype);
        layer.mlp_down_w = createTensor({_meta.hs, _meta.di}, _meta.dtype);
    }
}

void Qwen2Model::ensureCacheCapacity(size_t required_capacity) {
    if (required_capacity <= _cache_capacity) {
        return;
    }

    CHECK_ARGUMENT(required_capacity <= _meta.maxseq,
                   "Qwen2 input exceeds maximum sequence length");

    size_t new_capacity = _cache_capacity == 0
                            ? std::min(kInitialCacheCapacity, _meta.maxseq)
                            : _cache_capacity;
    while (new_capacity < required_capacity) {
        if (new_capacity > _meta.maxseq / 2) {
            new_capacity = _meta.maxseq;
        } else {
            new_capacity *= 2;
        }
    }

    std::vector<tensor_t> new_key_cache;
    std::vector<tensor_t> new_value_cache;
    new_key_cache.reserve(_meta.nlayer);
    new_value_cache.reserve(_meta.nlayer);

    for (size_t layer = 0; layer < _meta.nlayer; ++layer) {
        auto new_keys = createTensor(
            {new_capacity, _meta.nkvh, _meta.dh}, _meta.dtype);
        auto new_values = createTensor(
            {new_capacity, _meta.nkvh, _meta.dh}, _meta.dtype);

        if (_cache_length != 0) {
            auto key_prefix = new_keys->slice(0, 0, _cache_length);
            auto value_prefix = new_values->slice(0, 0, _cache_length);
            auto old_key_prefix = _key_cache[layer]->slice(0, 0, _cache_length);
            auto old_value_prefix = _value_cache[layer]->slice(0, 0, _cache_length);
            copyTensorData(key_prefix, old_key_prefix);
            copyTensorData(value_prefix, old_value_prefix);
        }

        new_key_cache.push_back(std::move(new_keys));
        new_value_cache.push_back(std::move(new_values));
    }

    _key_cache = std::move(new_key_cache);
    _value_cache = std::move(new_value_cache);
    _cache_capacity = new_capacity;
}

void Qwen2Model::ensureWorkspaceCapacity(size_t required_capacity) {
    CHECK_ARGUMENT(required_capacity > 0,
                   "Qwen2 workspace capacity must be positive");
    if (required_capacity <= _workspace_capacity) {
        return;
    }

    size_t new_capacity = std::max<size_t>(1, _workspace_capacity);
    while (new_capacity < required_capacity) {
        if (new_capacity > _meta.maxseq / 2) {
            new_capacity = _meta.maxseq;
        } else {
            new_capacity *= 2;
        }
    }

    const size_t query_size = _meta.nh * _meta.dh;
    const size_t key_value_size = _meta.nkvh * _meta.dh;
    auto workspace = std::make_unique<Qwen2Workspace>();
    workspace->input_ids = createTensor({new_capacity}, LLAISYS_DTYPE_I64);
    workspace->position_ids = createTensor({new_capacity}, LLAISYS_DTYPE_I64);
    workspace->hidden_a = createTensor({new_capacity, _meta.hs}, _meta.dtype);
    workspace->hidden_b = createTensor({new_capacity, _meta.hs}, _meta.dtype);
    workspace->attention_input = createTensor({new_capacity, _meta.hs}, _meta.dtype);
    workspace->query_projection = createTensor({new_capacity, query_size}, _meta.dtype);
    workspace->key_projection = createTensor({new_capacity, key_value_size}, _meta.dtype);
    workspace->rotated_query = createTensor({new_capacity, _meta.nh, _meta.dh}, _meta.dtype);
    workspace->attention_values = createTensor({new_capacity, _meta.nh, _meta.dh}, _meta.dtype);
    workspace->attention_output = createTensor({new_capacity, _meta.hs}, _meta.dtype);
    workspace->post_attention = createTensor({new_capacity, _meta.hs}, _meta.dtype);
    workspace->mlp_input = createTensor({new_capacity, _meta.hs}, _meta.dtype);
    workspace->gate = createTensor({new_capacity, _meta.di}, _meta.dtype);
    workspace->up = createTensor({new_capacity, _meta.di}, _meta.dtype);
    workspace->activated = createTensor({new_capacity, _meta.di}, _meta.dtype);
    workspace->mlp_output = createTensor({new_capacity, _meta.hs}, _meta.dtype);
    workspace->normalized = createTensor({1, _meta.hs}, _meta.dtype);
    workspace->logits = createTensor({1, _meta.voc}, _meta.dtype);
    workspace->max_index = createTensor({1}, LLAISYS_DTYPE_I64);
    workspace->max_value = createTensor({1}, _meta.dtype);
    workspace->host_max_index = Tensor::create(
        {1}, LLAISYS_DTYPE_I64, LLAISYS_DEVICE_CPU, 0);

    _workspace = std::move(workspace);
    _workspace_capacity = new_capacity;
}

void Qwen2Model::prepareRope(
    const std::vector<int64_t> &position_ids) {
    const size_t half_dim = _meta.dh / 2;
    CHECK_ARGUMENT(_rope_denominators.size() == half_dim,
                   "Qwen2 RoPE frequency table has an invalid size");

    const size_t table_size = position_ids.size() * half_dim;
    _rope_sin.resize(table_size);
    _rope_cos.resize(table_size);

    for (size_t token = 0; token < position_ids.size(); ++token) {
        const float position = static_cast<float>(position_ids[token]);
        const size_t offset = token * half_dim;
        for (size_t feature = 0; feature < half_dim; ++feature) {
            const float angle = position / _rope_denominators[feature];
            _rope_sin[offset + feature] = std::sin(angle);
            _rope_cos[offset + feature] = std::cos(angle);
        }
    }
}

void Qwen2Model::linearBf16(const tensor_t &out,
                            const tensor_t &in,
                            const tensor_t &weight,
                            const tensor_t &bias) {
    if (_device_type == LLAISYS_DEVICE_NVIDIA) {
        ops::linear(out, in, weight, bias);
        return;
    }

    CHECK_SAME_DEVICE(out, in, weight);
    ASSERT(out->ndim() == 2 && in->ndim() == 2 && weight->ndim() == 2,
           "Qwen2 linear tensors must be 2D");
    ASSERT(out->dtype() == LLAISYS_DTYPE_BF16 && in->dtype() == LLAISYS_DTYPE_BF16 && weight->dtype() == LLAISYS_DTYPE_BF16,
           "Qwen2 linear tensors must use BF16");
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(),
           "Qwen2 linear tensors must be contiguous");

    const size_t batch = in->shape()[0];
    const size_t in_features = in->shape()[1];
    const size_t out_features = weight->shape()[0];
    ASSERT(weight->shape()[1] == in_features,
           "Qwen2 linear input and weight shapes do not match");
    ASSERT(out->shape()[0] == batch && out->shape()[1] == out_features,
           "Qwen2 linear output shape does not match");

    const bf16_t *bias_data = nullptr;
    if (bias != nullptr) {
        CHECK_SAME_DEVICE(out, bias);
        ASSERT(bias->ndim() == 1 && bias->shape()[0] == out_features,
               "Qwen2 linear bias shape does not match");
        ASSERT(bias->dtype() == LLAISYS_DTYPE_BF16 && bias->isContiguous(),
               "Qwen2 linear bias must be contiguous BF16");
        bias_data = reinterpret_cast<const bf16_t *>(bias->data());
    }

    Bf16LinearContext context{
        reinterpret_cast<const bf16_t *>(in->data()),
        reinterpret_cast<const bf16_t *>(weight->data()),
        bias_data,
        reinterpret_cast<bf16_t *>(out->data()),
        batch,
        in_features,
        out_features};
    _workers->run(out_features, &runBf16Linear, &context);
}

void Qwen2Model::ropeBf16(const tensor_t &out,
                          const tensor_t &in,
                          const tensor_t &position_ids,
                          size_t num_heads) {
    if (_device_type == LLAISYS_DEVICE_NVIDIA) {
        ops::rope(out, in, position_ids, _meta.theta);
        return;
    }

    CHECK_SAME_DEVICE(out, in);
    ASSERT(out->ndim() == 3 && in->ndim() == 3,
           "Qwen2 RoPE tensors must be 3D");
    ASSERT(out->shape() == in->shape(),
           "Qwen2 RoPE input and output shapes must match");
    ASSERT(in->shape()[1] == num_heads && in->shape()[2] == _meta.dh,
           "Qwen2 RoPE tensor shape does not match the model");
    ASSERT(out->dtype() == LLAISYS_DTYPE_BF16 && in->dtype() == LLAISYS_DTYPE_BF16,
           "Qwen2 RoPE tensors must use BF16");
    ASSERT(out->isContiguous() && in->isContiguous(),
           "Qwen2 RoPE tensors must be contiguous");
    ASSERT(_rope_sin.size() == in->shape()[0] * (_meta.dh / 2) && _rope_cos.size() == _rope_sin.size(),
           "Qwen2 RoPE lookup table has an invalid size");

    Bf16RopeContext context{
        reinterpret_cast<const bf16_t *>(in->data()),
        reinterpret_cast<bf16_t *>(out->data()),
        _rope_sin.data(),
        _rope_cos.data(),
        num_heads,
        _meta.dh};
    _workers->run(in->shape()[0] * num_heads,
                  &runBf16Rope,
                  &context);
}

void Qwen2Model::copyTensorData(
    const tensor_t &out,
    const tensor_t &in) const {
    CHECK_SAME_DEVICE(out, in);
    ASSERT(out->shape() == in->shape(),
           "Qwen2 tensor copy shapes must match");
    ASSERT(out->dtype() == in->dtype(),
           "Qwen2 tensor copy dtypes must match");
    ASSERT(out->isContiguous() && in->isContiguous(),
           "Qwen2 tensor copies must be contiguous");

    llaisys::core::context().setDevice(_device_type, _device_id);
    auto &runtime = llaisys::core::context().runtime();
    runtime.api()->memcpy_async(
        out->data(),
        in->data(),
        out->numel() * out->elementSize(),
        _device_type == LLAISYS_DEVICE_CPU
            ? LLAISYS_MEMCPY_H2H
            : LLAISYS_MEMCPY_D2D,
        runtime.stream());
}

int64_t Qwen2Model::infer(const int64_t *token_ids, size_t ntoken) {
    CHECK_ARGUMENT(token_ids != nullptr,
                   "Qwen2 token id pointer cannot be null");
    CHECK_ARGUMENT(ntoken > 0,
                   "Qwen2 inference requires at least one token");
    CHECK_ARGUMENT(_cache_length <= _meta.maxseq,
                   "Qwen2 cache length exceeds maximum sequence length");
    CHECK_ARGUMENT(ntoken <= _meta.maxseq - _cache_length,
                   "Qwen2 input exceeds maximum sequence length");

    for (size_t token = 0; token < ntoken; ++token) {
        CHECK_ARGUMENT(token_ids[token] >= 0 && static_cast<size_t>(token_ids[token]) < _meta.voc,
                       "Qwen2 token id is outside the vocabulary");
    }

    const size_t total_length = _cache_length + ntoken;
    ensureCacheCapacity(total_length);
    ensureWorkspaceCapacity(ntoken);
    auto &workspace = *_workspace;

    auto input_ids = workspace.input_ids->slice(0, 0, ntoken);
    input_ids->load(token_ids);

    _position_values.resize(ntoken);
    for (size_t token = 0; token < ntoken; ++token) {
        CHECK_ARGUMENT(_cache_length + token <= static_cast<size_t>(
                           std::numeric_limits<int64_t>::max()),
                       "Qwen2 position id overflows int64");
        _position_values[token] = static_cast<int64_t>(_cache_length + token);
    }
    auto position_ids = workspace.position_ids->slice(0, 0, ntoken);
    position_ids->load(_position_values.data());
    if (_device_type == LLAISYS_DEVICE_CPU) {
        prepareRope(_position_values);
    }

    auto hidden = workspace.hidden_a->slice(0, 0, ntoken);
    auto next_hidden = workspace.hidden_b->slice(0, 0, ntoken);
    ops::embedding(hidden, input_ids, _weights.in_embed);

    auto attention_input = workspace.attention_input->slice(0, 0, ntoken);
    auto query_projection = workspace.query_projection->slice(0, 0, ntoken);
    auto key_projection = workspace.key_projection->slice(0, 0, ntoken);
    auto query = query_projection->view({ntoken, _meta.nh, _meta.dh});
    auto keys = key_projection->view({ntoken, _meta.nkvh, _meta.dh});
    auto rotated_query = workspace.rotated_query->slice(0, 0, ntoken);
    auto attention_values = workspace.attention_values->slice(0, 0, ntoken);
    auto attention_merged = attention_values->view({ntoken, _meta.hs});
    auto attention_output = workspace.attention_output->slice(0, 0, ntoken);
    auto post_attention = workspace.post_attention->slice(0, 0, ntoken);
    auto mlp_input = workspace.mlp_input->slice(0, 0, ntoken);
    auto gate = workspace.gate->slice(0, 0, ntoken);
    auto up = workspace.up->slice(0, 0, ntoken);
    auto activated = workspace.activated->slice(0, 0, ntoken);
    auto mlp_output = workspace.mlp_output->slice(0, 0, ntoken);

    const float attention_scale = 1.0f / std::sqrt(static_cast<float>(_meta.dh));
    const size_t key_value_size = _meta.nkvh * _meta.dh;

    for (size_t layer_index = 0;
         layer_index < _meta.nlayer;
         ++layer_index) {
        const auto &layer = _weights.layers[layer_index];
        auto key_destination = _key_cache[layer_index]->slice(
            0, _cache_length, total_length);
        auto value_destination = _value_cache[layer_index]->slice(
            0, _cache_length, total_length);
        auto value_projection = value_destination->view(
            {ntoken, key_value_size});

        ops::rms_norm(attention_input,
                      hidden,
                      layer.attn_norm_w,
                      _meta.epsilon);

        linearBf16(query_projection,
                   attention_input,
                   layer.attn_q_w,
                   layer.attn_q_b);
        linearBf16(key_projection,
                   attention_input,
                   layer.attn_k_w,
                   layer.attn_k_b);
        linearBf16(value_projection,
                   attention_input,
                   layer.attn_v_w,
                   layer.attn_v_b);

        ropeBf16(rotated_query, query, position_ids, _meta.nh);
        ropeBf16(key_destination, keys, position_ids, _meta.nkvh);

        auto cached_keys = _key_cache[layer_index]->slice(0, 0, total_length);
        auto cached_values = _value_cache[layer_index]->slice(0, 0, total_length);
        ops::self_attention(attention_values,
                            rotated_query,
                            cached_keys,
                            cached_values,
                            attention_scale);

        linearBf16(attention_output,
                   attention_merged,
                   layer.attn_o_w,
                   nullptr);
        ops::add(post_attention, hidden, attention_output);

        ops::rms_norm(mlp_input,
                      post_attention,
                      layer.mlp_norm_w,
                      _meta.epsilon);
        linearBf16(gate,
                   mlp_input,
                   layer.mlp_gate_w,
                   nullptr);
        linearBf16(up,
                   mlp_input,
                   layer.mlp_up_w,
                   nullptr);
        ops::swiglu(activated, gate, up);
        linearBf16(mlp_output,
                   activated,
                   layer.mlp_down_w,
                   nullptr);

        ops::add(next_hidden, post_attention, mlp_output);
        std::swap(hidden, next_hidden);
    }

    auto last_hidden = hidden->slice(0, ntoken - 1, ntoken);
    ops::rms_norm(workspace.normalized,
                  last_hidden,
                  _weights.out_norm_w,
                  _meta.epsilon);
    linearBf16(workspace.logits,
               workspace.normalized,
               _weights.out_embed,
               nullptr);

    auto logits = workspace.logits->view({_meta.voc});
    ops::argmax(workspace.max_index,
                workspace.max_value,
                logits);

    _cache_length = total_length;
    if (_device_type == LLAISYS_DEVICE_CPU) {
        return *reinterpret_cast<const int64_t *>(
            workspace.max_index->data());
    }

    llaisys::core::context().setDevice(_device_type, _device_id);
    auto &runtime = llaisys::core::context().runtime();
    runtime.api()->memcpy_async(
        workspace.host_max_index->data(),
        workspace.max_index->data(),
        sizeof(int64_t),
        LLAISYS_MEMCPY_D2H,
        runtime.stream());
    runtime.synchronize();
    return *reinterpret_cast<const int64_t *>(
        workspace.host_max_index->data());
}

void Qwen2Model::reset() {
    _cache_length = 0;
}

} // namespace llaisys::models
