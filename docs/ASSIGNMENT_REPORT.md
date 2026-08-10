# 作业 4：CUDA 平台支持与推理优化报告

本报告记录 LLAISYS 作业 4 的持续集成情况、CPU/NVIDIA 复现流程与结果，以及各平台当前的支持状态。测试模型为 [DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)。

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

GitHub-hosted runner 未提供本作业所需的 NVIDIA GPU，因此 NVIDIA 构建、算子与端到端推理在本地 GPU 环境中验证。

> **CI 截图占位**
>
> 在此粘贴最终 Pull Request 的 Checks 全绿截图，或 GitHub Actions Run #7 的 Windows/Ubuntu 双平台通过截图。

## 2. 复现流程及结果

### 2.1 测试环境

| 项目 | 配置 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5090，Compute Capability 12.0，32 GiB |
| NVIDIA Driver | 610.43.02 |
| CUDA Toolkit / NVCC | 12.8 / 12.8.61 |
| Xmake | 2.8.7 |
| Python / PyTorch | Python 3.12 / PyTorch 2.13.0+cu130 |
| 模型与精度 | DeepSeek-R1-Distill-Qwen-1.5B，BF16 |
| 推理设置 | batch=1，提示词 `Who are you?`，Greedy，最多生成 128 tokens |

以下命令均在仓库根目录运行。命令中的 `[dir_path/to/model]` 需要替换为模型所在目录。

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

### 2.4 正确性与性能结果

| 测试 | 结果 |
| --- | --- |
| CPU/NVIDIA Runtime、Tensor | 通过 |
| CPU/NVIDIA 八个算子回归 | 全部通过 |
| Argmax 151936 元素、跨 block 相同最大值、全负数和 NaN | 通过 |
| GQA Self-Attention：`nh=12`、`nkvh=2`、`hd=128`、`kv_len=1/32/128` | 通过 |
| Qwen2 wrapper 与 KV Cache 扩容 | 通过 |
| NVIDIA Greedy 128 步 token 对齐 | 与 Hugging Face 完全一致 |

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
| 沐曦 | 待实现 | 待实现 | 待实现 | 待测试 | 待开发 |

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

### 3.3 沐曦（待开发）

沐曦后端尚未开发，本节仅为后续作业交付预留位置，不声明任何已支持能力。

> **沐曦平台结果占位**
>
> - GPU 型号与显存：待补充
> - 驱动与 SDK 版本：待补充
> - 构建与测试命令：待补充
> - Runtime、Tensor、算子状态：待补充
> - Qwen2 token 对齐结果：待补充
> - Hugging Face / LLAISYS 推理耗时：待补充
