# 作业 4：GPU 平台支持与推理优化报告

本报告记录 LLAISYS 作业 4 的持续集成情况、CPU/NVIDIA/MetaX 复现流程与结果，以及各平台当前的支持状态。测试模型为 [DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)。

## 1. CI 情况

仓库使用 [GitHub Actions](https://github.com/Yushang-Lu/llaisys-26s/actions) 自动完成构建和 CPU 回归测试。工作流在 Pull Request 上触发；push 事件会忽略仅包含 Markdown 或 LICENSE 的变更。

最近一次代码提交的 CI 为 [Build and test #7](https://github.com/Yushang-Lu/llaisys-26s/actions/runs/31356271456)，对应 commit `af267c7`，结论为 **Success**。

| Job | 构建模式 | 结果 | 用时 |
| --- | --- | --- | ---: |
| Ubuntu Latest | Release | 通过 | 5 分 24 秒 |
| Windows Latest | Release | 通过 | 7 分 07 秒 |

CI 在两个操作系统上依次执行以下检查：

1. 使用 Xmake 构建并安装 LLAISYS。
2. 安装 Python 包。
3. 测试 CPU Runtime 和 Tensor。
4. 测试 Add、Argmax、Embedding、Linear、RMSNorm、RoPE、Self-Attention、SwiGLU 八个算子。
5. 运行 Qwen2 Greedy 推理并与 Hugging Face token 序列对齐。

GitHub-hosted runner 未提供本作业所需的 NVIDIA 或 MetaX GPU，因此两类 GPU 后端分别在对应的本地硬件环境中验证。

> **CI 截图占位**
>
> 在此粘贴最终 Pull Request 的 Checks 全绿截图，或 GitHub Actions Run #7 的 Windows/Ubuntu 双平台通过截图。

## 2. 复现流程及结果

### 2.1 测试环境

#### 2.1.1 NVIDIA 环境

| 项目 | 配置 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5090，Compute Capability 12.0，32 GiB |
| NVIDIA Driver | 610.43.02 |
| CUDA Toolkit / NVCC | 12.8 / 12.8.61 |
| Xmake | 2.8.7 |
| Python / PyTorch | Python 3.12 / PyTorch 2.13.0+cu130 |
| 模型与精度 | DeepSeek-R1-Distill-Qwen-1.5B，BF16 |
| 推理设置 | batch=1，提示词 `Who are you?`，Greedy，最多生成 128 tokens |

#### 2.1.2 MetaX C500 环境

MetaX 环境于 2026-08-10 核对，当前实例为受限 sGPU，不能按物理卡标称显存分配：

| 项目 | 实测值 |
| --- | --- |
| OS | Ubuntu 24.04.1 LTS，x86_64 |
| GPU | MetaX 曦云 C500；25% compute、16 GiB VRAM 配额 |
| 物理卡显示 | 64 GiB，不代表当前进程可用配额 |
| MX-SMI / Kernel Driver | 2.2.9 / 3.8.30 |
| MACA Runtime / SDK | 3.2.1.10 / `/opt/maca -> /opt/maca-3.2.1` |
| cucc / mxcc / 默认目标 | `/opt/maca/tools/cu-bridge/bin/cucc` / 1.0.0 / `xcore1000` |
| Host C++ / Xmake | GCC 13.3.0 / 3.1.0+HEAD.96ad28e |
| Python / PyTorch | Python 3.10.10 / PyTorch 2.6.0+metax3.2.1.3 |
| Transformers / NumPy | 4.57.1 / 1.26.4 |
| 模型 | `/data/models/DeepSeek-R1-Distill-Qwen-1.5B`，BF16 |

`pip check` 返回 `No broken requirements found`。实现和测试期间没有安装、卸载或升级 Python 包，也没有联网下载模型。

以下 CPU/NVIDIA 命令均在仓库根目录运行，其中 `[dir_path/to/model]` 需要替换为模型所在目录；MetaX 使用 2.4 节的固定解释器和本地模型路径。

### 2.2 CPU 复现

按照 README 的默认流程构建并安装 CPU 版本：

```bash
xmake
xmake install
pip install ./python/
```

运行 Runtime、Tensor、八个算子和端到端推理：

```bash
python test/test_runtime.py --device cpu
python test/test_tensor.py

python test/ops/add.py
python test/ops/argmax.py
python test/ops/embedding.py
python test/ops/linear.py
python test/ops/rms_norm.py
python test/ops/rope.py
python test/ops/self_attention.py
python test/ops/swiglu.py

python test/test_infer.py --model [dir_path/to/model] --test
```

Runtime 关键输出：

```console
Found 1 cpu devices
Testing device 0...
     Passed
Test passed!
```

其余测试输出较长，关键结论节选如下：

```console
===Test load===
===Test view===
===Test permute===
===Test slice===
Test passed!

Testing Ops.add on cpu
...
Test passed!
...
Testing Ops.swiglu on cpu
...
Test passed!

=== Your Result ===
Tokens:
[...]
...
Test passed!
```

CPU Runtime、Tensor、八个算子及 Qwen2 推理均通过；CPU 作为功能与 CI 回归基线，本报告未进行同口径的 128-token 性能统计。

### 2.3 NVIDIA 复现

按照 README 启用 NVIDIA 后端并重新构建：

```bash
xmake f --nv-gpu=y -cv
xmake
xmake install
pip install ./python/
```

运行 NVIDIA Runtime、Tensor、八个算子和端到端推理：

```bash
python test/test_runtime.py --device nvidia
python test/test_tensor.py --device nvidia

python test/ops/add.py --device nvidia
python test/ops/argmax.py --device nvidia
python test/ops/embedding.py --device nvidia
python test/ops/linear.py --device nvidia
python test/ops/rms_norm.py --device nvidia
python test/ops/rope.py --device nvidia
python test/ops/self_attention.py --device nvidia
python test/ops/swiglu.py --device nvidia

python test/test_infer.py --model [dir_path/to/model] --test --device nvidia
```

Runtime 关键输出：

```console
Found 1 nvidia devices
Testing device 0...
     Passed
Test passed!
```

算子与推理关键输出节选：

```console
Testing Ops.argmax on nvidia
   large vocabulary shape (151936,) dtype <f32>
   cross-block tie shape (8192,) dtype <f32>
   all negative shape (10000,) dtype <f32>
   NaN precedence shape (151936,) dtype <f32>
...
Test passed!

Testing Ops.self_attention on nvidia
...
Test passed!

=== Answer ===
...
Time elapsed: 2.16s

=== Your Result ===
...
Time elapsed: 0.42s

Test passed!
```

上述推理时间不包含模型加载，具体数值会随硬件和软件环境变化。

### 2.4 MetaX C500 复现

MetaX 使用与 MACA 版本配套的现有环境。先显式初始化工具链，不执行 `pip install`：

```bash
cd /data/llaisys-26s
source /opt/conda/etc/profile.d/conda.sh
conda activate /root/envs/myproject

export MACA_PATH=/opt/maca
export CUCC_PATH=$MACA_PATH/tools/cu-bridge
export CUDA_PATH=$CUCC_PATH
export PATH=$CUCC_PATH/bin:$CUCC_PATH/tools:$MACA_PATH/mxgpu_llvm/bin:$MACA_PATH/bin:$PATH
export LD_LIBRARY_PATH=$MACA_PATH/lib:$MACA_PATH/ompi/lib:$MACA_PATH/ucx/lib:$MACA_PATH/mxgpu_llvm/lib:/opt/mxdriver/lib:$LD_LIBRARY_PATH
export XMAKE_ROOT=y
export PYTHONPATH=/data/llaisys-26s/python

PY=/root/envs/myproject/bin/python
XMAKE=/root/envs/myproject/bin/xmake
MODEL=/data/models/DeepSeek-R1-Distill-Qwen-1.5B
```

启用独立 MetaX 后端并运行 Runtime、Tensor、八算子与离线端到端测试：

```bash
"$XMAKE" f -c --nv-gpu=n --metax-gpu=y
"$XMAKE" -vD
"$XMAKE" install

"$PY" test/test_runtime.py --device metax
"$PY" test/test_tensor.py --device metax

for op in add argmax embedding linear rms_norm rope self_attention swiglu; do
    "$PY" "test/ops/$op.py" --device metax
done

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
"$PY" test/test_infer.py --model "$MODEL" --test --device metax
```

CPU-only 回归使用 `"$XMAKE" f -c --nv-gpu=n --metax-gpu=n` 重新配置；Runtime、Tensor、八算子和 `test/test_qwen2_wrapper.py` 均通过。

### 2.5 正确性与性能结果

| 测试 | 结果 |
| --- | --- |
| CPU/NVIDIA/MetaX Runtime、Tensor | 通过 |
| CPU/NVIDIA/MetaX 八个算子回归 | 全部通过 |
| Argmax 151936 元素、跨 block 相同最大值、全负数和 NaN | 通过 |
| GQA Self-Attention：`nh=12`、`nkvh=2`、`hd=128`、`kv_len=1/32/128` | 通过 |
| Qwen2 wrapper 与 KV Cache 扩容 | 通过 |
| NVIDIA Greedy 128 步 token 对齐 | 与 Hugging Face 完全一致 |
| MetaX Greedy（最多 128 步）token 对齐 | 与 Hugging Face 完全一致 |

固定 128-token 输出的 token 序列 SHA-256 为：

```text
b9bee4bf2cf1489a06bcae06bcc81bd775964af96eb6617c552f11fe54dadf83
```

本次 NVIDIA 优化包括：

- V projection 直接写入 Value Cache，K projection 经 RoPE 后直接写入 Key Cache，稳态 decode 每个 token 避免 56 次 D2D copy。
- Attention 按工作维度动态选择 32～256 threads，减少短上下文中的空闲线程和无效归约。
- Argmax 对大输入使用两阶段归约，同时保持 PyTorch 的 NaN 和相同最大值取最小索引语义。
- 缓存 cuBLAS Stream 绑定并复用 Attention、Argmax workspace，降低 CUDA API 和内存管理开销。

性能测试使用 Release 构建，模型加载不计时。每个版本预热一次后执行 10 次完整 128-token 生成：

| 版本 | Commit | 中位耗时 | P95 |
| --- | --- | ---: | ---: |
| 优化前 NVIDIA 实现 | `8515c01` | 445.191 ms | 445.921 ms |
| 优化后 NVIDIA 实现 | `af267c7` | 424.484 ms | 424.599 ms |
| 改善 | — | 4.65% | 4.78% |

Nsight 在预热后的生成区间内未捕获到 D2D copy、`cudaMalloc` 或 `cudaFree`；两阶段 Argmax 总耗时约 4.03 μs/token，Attention 的 score、softmax、value kernel 平均每层分别约为 1.93、1.51、3.23 μs。

此外还验证了 batch=1 BF16 GEMV。虽然 token 保持一致，但中位耗时仅改善约 0.03%，低于 1% 的保留门槛，因此最终继续使用 `cublasGemmEx`。

## 3. 平台支持及状态

| 平台 | Runtime / Tensor | 八个算子 | Qwen2 推理 | 性能对比 | 状态 |
| --- | --- | --- | --- | --- | --- |
| CPU | 支持 | 支持 | 单设备推理支持 | 未进行同口径统计 | 已完成，CI 与本地回归通过 |
| NVIDIA | CUDA Runtime 与 NVIDIA Tensor 支持 | F32/F16/BF16 支持 | 单卡 BF16、KV Cache 支持 | Hugging Face 约 2.16 s；LLAISYS 约 0.42 s | 已完成，正确性与性能验证通过 |
| MetaX C500 | MACA Runtime 与 MetaX Tensor 支持 | F32/F16/BF16 支持 | 单卡 BF16、KV Cache 支持 | Hugging Face 2.79–3.17 s；LLAISYS 0.56–0.58 s（两次） | 已完成，正确性与本机验证通过 |

### 3.1 CPU

- 默认随项目构建，无需额外设备开关。
- 支持 Runtime、Tensor、八个算子和 Qwen2 单设备推理。
- Windows/Ubuntu Release CI 以及本地回归均已通过。
- 当前定位为功能和兼容性基线，未记录与 Hugging Face 同口径的 128-token 性能数据。

### 3.2 NVIDIA

- 支持设备与 Stream 管理、Device/Pinned Host 内存、同步和异步拷贝。
- Tensor 支持创建、加载、视图和调试；八个算子支持 F32、F16 和 BF16。
- Linear 使用 cuBLAS，归约和 Attention 使用 FP32 中间值。
- 支持 DeepSeek-R1-Distill-Qwen-1.5B 单卡 BF16 推理和 KV Cache，token 与 Hugging Face 完全一致。

同一 RTX 5090 环境下的端到端结果如下：

| 实现 | 128-token 推理耗时 | 相对 Hugging Face |
| --- | ---: | ---: |
| Hugging Face | 约 2.16 s | 1.00× |
| LLAISYS NVIDIA | 约 0.42 s | 约 5.1× |

该比较不包含模型加载，仅代表本报告所列硬件、模型和 Greedy 工作负载。

### 3.3 MetaX C500

- 通过独立 `LLAISYS_DEVICE_METAX`、`--metax-gpu=y` 和 `ENABLE_METAX_API` 接入，不复用 NVIDIA 设备身份。
- 支持设备与 Stream 管理、Device/Pinned Host 内存、同步和异步拷贝。
- 八算子均提供独立 `llaisys::ops::metax` 实现，支持 F32、F16 和 BF16。
- Linear 使用 mcBLAS，Argmax 与 Attention 复用 workspace，并保持 NaN/tie、GQA、causal 与短 decode 语义。
- DeepSeek-R1-Distill-Qwen-1.5B 单卡 BF16 greedy token 与 Hugging Face 完全一致。

当前 25% compute、16 GiB C500 sGPU 上两次正确性运行的观察范围如下：

| 实现 | 最多 128-token 推理耗时 | Token 结果 |
| --- | ---: | --- |
| Hugging Face / MetaX PyTorch | 2.79–3.17 s | 基准 |
| LLAISYS MetaX | 0.56–0.58 s | 完全一致 |

该数据不包含模型加载，只是受限 sGPU 上两次正确性运行的观察范围，不代表整卡 C500 或稳定性能基准。

#### 3.3.1 公共 ABI 与独立构建链

C ABI 在保留 `CPU=0`、`NVIDIA=1` 的基础上追加 `LLAISYS_DEVICE_METAX=2`，Python `DeviceType` 同步为 `METAX=2`、`COUNT=3`。未启用 MetaX 时，该设备类型使用 unsupported Runtime、设备数为 0，不影响 CPU-only 构建；所有算子的公共分发均显式进入 `llaisys::ops::metax`，没有借用 NVIDIA 设备身份或 namespace。

MetaX 默认关闭，仅在 `--metax-gpu=y` 时载入 `xmake/metax.lua`。构建链的关键点如下：

- `llaisys-device-metax` 与 `llaisys-ops-metax` 是独立静态 target。
- 本机没有标准 NVIDIA CUDA SDK 或 nvcc；MetaX kernel 使用 `.cpp` 扩展，由 cucc 以 `-x maca` 编译，并显式设置兼容头文件和 `-fPIC`。
- 最终共享库由 g++ 链接，并显式保留 `mcblas`、`mcruntime`、`runtime_cu`、`mccompiler` 及其库目录和 rpath，避免 `--as-needed` 丢失间接依赖。
- `ldd -r python/llaisys/libllaisys/libllaisys.so` 已确认 MACA 依赖完整、无未解析符号；构建过程没有创建全局 CUDA/nvcc 软链。
- NVIDIA 的 `--nv-gpu`、`ENABLE_NVIDIA_API`、`xmake/nvidia.lua`、gencode 和源文件保持独立。

#### 3.3.2 Runtime 与资源管理

`src/device/metax/metax_runtime_api.cpp` 完整实现通用 Runtime 接口：

| LLAISYS 能力 | MACA CUDA-compatible API |
| --- | --- |
| 设备计数/选择 | `cudaGetDeviceCount` / `cudaSetDevice` |
| 设备同步 | `cudaDeviceSynchronize` |
| Stream 创建/销毁/同步 | `cudaStreamCreate` / `cudaStreamDestroy` / `cudaStreamSynchronize` |
| Device 内存 | `cudaMalloc` / `cudaFree` |
| Pinned Host 内存 | `cudaMallocHost` / `cudaFreeHost` |
| 同步/异步拷贝 | `cudaMemcpy` / `cudaMemcpyAsync` |

H2H、H2D、D2H、D2D 均有显式映射；零字节分配或拷贝直接返回，其他错误转换为带操作上下文的 C++ 异常。`llaisys::device::metax::Resource` 按线程和 device id 复用 mcBLAS handle，并缓存最近绑定的 Stream；Attention 的 FP32 score workspace 按 2 倍扩容，Argmax 使用可容纳 256 个 partial result 的固定 workspace。Qwen2 KV Cache 从 256 token 起按需扩容，以控制 16 GiB 配额下的峰值显存。

#### 3.3.3 八算子实现与关键语义

| 算子 | MetaX 实现要点 |
| --- | --- |
| Add | Grid-stride elementwise kernel，F32 中间计算 |
| Argmax | 小输入单 block，大输入两阶段归约；FP32 比较 |
| Embedding | 展平输出后按 token index 读取权重行 |
| Linear | mcBLAS `cublasGemmEx` 兼容接口，随后独立 bias kernel |
| RMSNorm | 每行一个 block，FP32 平方和归约与 `rsqrtf` |
| RoPE | 每个旋转 pair 一个工作项，以 `sincosf` 计算 |
| Self-Attention | score、causal softmax、value 三个 kernel，FP32 score workspace |
| SwiGLU | FP32 SiLU 中间计算后转换回输出类型 |

Linear 将 row-major 张量改写为 `Y^T = W X^T` 调用 column-major BLAS，F32/F16/BF16 均使用 FP32 累加；本机 smoke test 和大矩阵用例验证了 mcBLAS 的 dtype、转置、leading dimension、Stream 与算法枚举。

Argmax 保持与 NVIDIA 专项测试相同的规则：NaN 优先于非 NaN，多个 NaN 或相同最大值均取最小索引，全负数与 `-inf` 不依赖零初始化。Self-Attention 支持 GQA 映射 `kv_head = query_head / (num_heads / num_kv_heads)`，按 `kv_len - q_len + query_index + 1` 计算 causal 可见长度，并以 FP32 完成 score、softmax 和 value 累加；专项测试覆盖 `nh=12`、`nkvh=2`、`hd=128`、`kv_len=1/32/128`。

#### 3.3.4 Qwen2 与 Python 参考路径

Qwen2 C API、native model 和 Python wrapper 均接受 MetaX。测试前端将设备映射分为两层：PyTorch 参考路径使用 `torch.device("cuda:<id>")`，因为 MetaX wheel 暴露 CUDA-compatible API；LLAISYS 路径使用独立 `DeviceType.METAX`。

参考模型改为普通 `from_pretrained` 后显式 `model.to(cuda:0)`，避免 `device_map` 触发未安装的可选 `accelerate` 依赖。MetaX PyTorch 的 `torch.backends.cuda.matmul.allow_tf32` 默认为 true，而 LLAISYS F32 Linear 与 Attention 使用 full-F32；严格参考测试仅在 MetaX PyTorch 计算区间临时关闭 TF32，并在结束后恢复，没有放宽容差或修改 NVIDIA 分支：

| 用例 | PyTorch 默认 TF32 与 LLAISYS 最大绝对差 | 禁用 TF32 后 |
| --- | ---: | ---: |
| Linear `512×4096×4096` | `3.23057e-5` | `0` |
| Self-Attention `q_len=2, hd=4` | `3.78847e-4` | `0` |

#### 3.3.5 验证边界与当前限制

- MetaX Runtime、Tensor、八算子三种浮点 dtype 和 Qwen2 token 对齐均通过；测试进程退出后，`mx-smi` 显示 sGPU `0/16000 MiB`，没有残留进程或持久显存占用。
- CPU-only 干净构建、Runtime、Tensor、八算子和 Qwen2 wrapper（`5 passed`）通过，且 CPU 共享库不含 MACA/NVIDIA 依赖。
- C500 主机没有标准 NVIDIA CUDA SDK，因此未声明 NVIDIA GPU 运行回归在本机通过；NVIDIA 结果仍以对应硬件环境和 CI 为准。
- MetaX 构建当前仅支持 Linux，并依赖 MACA 3.2.1/cu-bridge；本次只验证单个 C500 sGPU，未覆盖多卡、多个 sGPU 或其他 MACA 版本。
- Qwen2 native model 按作业范围支持单设备 BF16 greedy 推理，未实现采样；当前性能数字仅是受限配额上的两次正确性观察，不是多轮中位数/P95 或整卡基准。
