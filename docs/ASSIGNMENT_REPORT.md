# LLAISYS-26S：CPU / GPU 平台支持与复现流程报告

本报告记录 LLAISYS-26S 作业的 CI 测试情况、CPU/NVIDIA/MetaX 复现流程与结果，以及各平台当前的支持状态。测试模型为 [DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B)。

## 1. CI 情况

仓库使用 [GitHub Actions](https://github.com/Yushang-Lu/llaisys-26s/actions) 自动完成构建和 CPU 回归测试。

最近一次代码提交的 CI 为 [Build and test #12](https://github.com/Yushang-Lu/llaisys-26s/actions/runs/31461447435)，对应 commit `57967b2`，结论为 **Success**。

| Job | 构建模式 | 结果 | 用时 |
| --- | --- | --- | ---: |
| Ubuntu Latest | Release | 通过 | 5 分 10 秒 |
| Windows Latest | Release | 通过 | 6 分 38 秒 |

GitHub-hosted runner 未提供本作业所需的 NVIDIA 或 MetaX GPU，因此两类 GPU 后端分别在对应的服务器硬件环境中验证。

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

MetaX 实例为受限 sGPU，与物理卡标称显存不同：

| 项目 | 实测值 |
| --- | --- |
| GPU | MetaX 曦云 C500；25% compute、16 GiB VRAM 配额 |
| 物理卡显示 | 64 GiB，不代表当前进程可用配额 |
| MX-SMI / Kernel Driver | 2.2.9 / 3.8.30 |
| MACA Runtime / SDK | 3.2.1.10 / 3.2.1 |
| cucc / mxcc / 默认目标 | cucc（由`CUCC_PATH`指定） / 1.0.0 / `xcore1000` |
| Host C++ / Xmake | GCC 13.3.0 / 3.1.0+HEAD.96ad28e |
| Python / PyTorch | Python 3.10.10 / PyTorch 2.6.0+metax3.2.1.3 |
| Transformers / NumPy | 4.57.1 / 1.26.4 |
| 模型 | DeepSeek-R1-Distill-Qwen-1.5B，BF16 |

以下命令均在仓库根目录运行，其中 `[dir_path/to/model]` 需要替换为模型所在目录；MetaX 复现前需进入与 MACA 版本匹配的 Python 环境。

### 2.2 CPU 复现

构建并安装：

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

CPU Runtime、Tensor、八个算子及 Qwen2 推理均通过。

### 2.3 NVIDIA 复现

启用 NVIDIA 后端并重新构建：

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

Runtime、Tensor、八个算子及 Qwen2 推理均通过。推理时间不包含模型加载，具体数值会随硬件和软件环境变化。

### 2.4 MetaX C500 复现

MetaX 依赖与 MACA 版本配套的 Python/PyTorch 环境。请先进入已安装 MetaX PyTorch、Transformers、NumPy 和 Xmake 的兼容环境，然后在仓库根目录初始化工具链。以下路径均可通过同名环境变量覆盖；示例默认值对应常见的 MACA 安装布局。

```bash
export MACA_PATH="${MACA_PATH:-/opt/maca}"
export CUCC_PATH="${CUCC_PATH:-$MACA_PATH/tools/cu-bridge}"
export MXDRIVER_PATH="${MXDRIVER_PATH:-/opt/mxdriver}"
export CUDA_PATH="${CUDA_PATH:-$CUCC_PATH}"

export PATH="$CUCC_PATH/bin:$CUCC_PATH/tools:$MACA_PATH/mxgpu_llvm/bin:$MACA_PATH/bin:$PATH"
export LD_LIBRARY_PATH="$MACA_PATH/lib:$MACA_PATH/ompi/lib:$MACA_PATH/ucx/lib:$MACA_PATH/mxgpu_llvm/lib:$MXDRIVER_PATH/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"

PYTHON="${PYTHON:-python}"
XMAKE="${XMAKE:-xmake}"

# 必填：本地模型目录
# export MODEL_PATH=/path/to/DeepSeek-R1-Distill-Qwen-1.5B
: "${MODEL_PATH:?请先设置 MODEL_PATH 为本地模型目录}"
```

启用独立 MetaX 后端并运行 Runtime、Tensor、八算子与离线端到端测试：

```bash
"$XMAKE" f -c --nv-gpu=n --metax-gpu=y
"$XMAKE" -vD
"$XMAKE" install

"$PYTHON" test/test_runtime.py --device metax
"$PYTHON" test/test_tensor.py --device metax

for op in add argmax embedding linear rms_norm rope self_attention swiglu; do
    "$PYTHON" "test/ops/$op.py" --device metax
done

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
"$PYTHON" test/test_infer.py --model "$MODEL_PATH" --test --device metax
```

Runtime、Tensor、八个算子及 Qwen2 推理均通过。

## 3. 平台支持及状态

| 平台 | Runtime / Tensor | 八个算子 | Qwen2 推理 | 性能对比 | 状态 |
| --- | --- | --- | --- | --- | --- |
| CPU | 支持 | F32/F16/BF16 支持 | CPU 推理、KV Cache 支持 | Hugging Face 约 15.23 s；LLAISYS 约 39.36 s | CI 与本地测试通过 |
| NVIDIA 5090 | 支持 | F32/F16/BF16 支持 | 单卡 BF16 推理、KV Cache 支持 | Hugging Face 约 2.16 s；LLAISYS 约 0.42 s | 本地测试通过 |
| MetaX C500 | 支持 | F32/F16/BF16 支持 | 单卡 BF16 推理、KV Cache 支持 | Hugging Face 约 2.79 s；LLAISYS 约 0.56 s | 本地测试通过 |

### 3.1 CPU

- 默认随项目构建，无需额外设备开关。
- 支持 Runtime、Tensor、八个算子和 Qwen2 单设备推理。
- Windows/Ubuntu Release CI 以及本地测试均已通过。

### 3.2 NVIDIA

- 支持设备与 Stream 管理、Device/Pinned Host 内存、同步和异步拷贝。
- Tensor 支持创建、加载、视图和调试；八个算子支持 F32、F16 和 BF16。
- Linear 使用 cuBLAS，归约和 Attention 使用 FP32 中间值。
- 支持 DeepSeek-R1-Distill-Qwen-1.5B 单卡 BF16 推理和 KV Cache，token 与 Hugging Face 完全一致。

同一 RTX 5090 环境下的推理测试结果如下：

| 实现 | 最多 128-token 推理耗时 | 相对 Hugging Face |
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

当前 25% compute、16 GiB C500 sGPU 上推理测试结果如下：

| 实现 | 最多 128-token 推理耗时 | 相对 Hugging Face |
| --- | ---: | --- |
| Hugging Face | 约 2.79 s | 1.00× |
| LLAISYS MetaX | 约 0.56 s | 约 5.0× |

该数据不包含模型加载，只是受限 sGPU 上运行的结果，不代表整卡 C500 稳定性能。
