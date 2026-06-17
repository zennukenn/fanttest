# vLLM Model Bench

`vllm-model-bench` 是一个用于自动化执行 vLLM 模型性能测试和稳定性测试的
Python 编排工具。

它不会重新实现 vLLM 的 benchmark 计算逻辑。性能测试会直接调用
`vllm bench serve`，并保留 vLLM 生成的原始 JSON 结果。本项目只负责启动
vLLM server、分配空闲 GPU、展开测试配置、执行测试任务，以及把 TTFT、TPOT
等已有指标汇总到 CSV。

## 功能

- 支持 Dense、MoE、VL、VL-MoE 类型的模型配置。
- 支持为每个模型单独配置 TP、DP、PP、EP、dtype、MTP/speculative decoding
  以及任意 vLLM server 参数。
- 简化 GPU 调度：只检测当前空闲 GPU，按 `tp * dp * pp` 分配资源；资源不够
  时等待。
- 性能测试通过 `vllm bench serve` 执行。
- 稳定性测试通过比较当前模型输出和已保存 baseline 判断输出是否变化。
- 支持 dry-run plan，提前查看任务、GPU 需求和将要执行的 server 命令。

## 安装

在项目目录下执行：

```bash
pip install -e .
```

运行 benchmark 的环境需要已经安装好 `vllm`，并且能够使用 `nvidia-smi`。

## 配置

编辑 `bench_config.py`。

每个模型可以按下面的方式配置：

```python
{
    "id": "vl_qwen_7b",
    "type": "vl",                  # dense, moe, vl, vl_moe
    "model": "Qwen/Qwen2.5-VL-7B-Instruct",
    "served_model_name": "vl_qwen_7b",
    "tp": 2,
    "dp": 1,
    "pp": 1,
    "enable_ep": False,
    "dtype": "bfloat16",
    "server_args": {
        "max_model_len": 32768,
        "limit_mm_per_prompt": {"image": 4},
    },
    "mtp": {
        "enabled": True,
        "spec_method": "deepseek_mtp",
        "spec_tokens": 2,
    },
    "extra_server_args": [],
    "bench": {
        "backend": "openai-chat",
        "endpoint": "/v1/chat/completions",
        "dataset_name": "random-mm",
        "extra_args": [],
    },
}
```

`server_args` 中的 key 会自动转换成 vLLM CLI 参数。例如：

```python
{"gpu_memory_utilization": 0.9}
```

会变成：

```bash
--gpu-memory-utilization 0.9
```

布尔值 `True` 会转换成 flag。例如：

```python
{"enable_eplb": True}
```

会变成：

```bash
--enable-eplb
```

如果当前 vLLM 版本更适合使用单个 JSON speculative 配置，可以这样写：

```python
"mtp": {
    "enabled": True,
    "speculative_config": {
        "method": "deepseek_mtp",
        "num_speculative_tokens": 2,
    },
}
```

## GPU 调度

调度逻辑刻意保持简单：

```text
required_gpus = tp * dp * pp
```

这个规则适用于 Dense、MoE、VL、VL-MoE。对于开启 EP 的 MoE 模型，vLLM 会根据
TP 和 DP 计算 EP size，所以单机资源分配仍然使用 `tp * dp * pp`。

空闲 GPU 通过 `nvidia-smi` 检测。默认情况下，一个 GPU 同时满足以下条件才会
被认为是空闲：

- 没有 compute process。
- `memory.used <= idle_memory_threshold_mb`。

示例：

- 4 张空闲 GPU 可以同时运行两个 `tp=2, dp=1` 的任务。
- 一个 `tp=4, dp=1` 的任务会占用全部 4 张 GPU。
- 一个开启 EP 的 MoE 任务，如果配置为 `tp=1, dp=8`，则需要 8 张 GPU。

## 命令

查看任务计划：

```bash
vmb --config bench_config.py plan --kind all
```

运行所有性能测试：

```bash
vmb --config bench_config.py run perf
```

运行稳定性测试，并自动创建缺失的 baseline：

```bash
vmb --config bench_config.py run stability
```

刷新稳定性测试 baseline：

```bash
vmb --config bench_config.py run stability --update-baseline
```

只运行某一个模型：

```bash
vmb --config bench_config.py run perf --model vl_qwen_7b
```

## 结果目录

默认输出到 `results/`：

```text
results/
  raw/                         # vllm bench serve 生成的原始 JSON
  csv/perf_results.csv         # 追加写入的性能结果汇总
  runs/<task_id>/              # server 日志、bench 日志、manifest
  stability/
    baselines/                 # 稳定性测试 baseline
    current/                   # 本次输出
    reports/                   # 对比报告和 diff
```

`perf_results.csv` 会保留原始 JSON 路径，方便每一行汇总结果都能追溯到
`vllm bench serve` 的原始输出。

## 说明

- 本项目不会自动调优 TP、DP、EP、dtype 或 MTP；配置文件是唯一来源。
- 如果不同 vLLM 版本的参数名有变化，可以把精确参数放到
  `extra_server_args` 或 `bench.extra_args`。
- `PERF_CASES` 支持设置 `temperature`、`top_p`、`request_rate` 等采样和压测
  参数，这些参数会转发给 `vllm bench serve`。
- 稳定性测试会使用 case 中的采样参数。为了做严格回归检查，建议使用
  `temperature=0`，并在模型支持时固定 `seed`。
