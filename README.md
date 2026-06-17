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
- 稳定性测试通过比较当前模型输出和已保存 baseline 判断输出是否变化，baseline
  会保存为结构化 JSON，便于追踪请求、响应、hash 和 diff。
- 支持 dry-run plan，提前查看任务、GPU 需求和将要执行的 server 命令。
- 运行时会打印全流程进度，包括排队、等待 GPU、GPU 分配、server 启动、
  ready 检查、client 执行、结果写入、server 停止和 task 完成。

## 安装

在项目目录下执行：

```bash
pip install -e .
```

运行 benchmark 的环境需要已经安装好 `vllm`。GPU 分配使用静态 GPU 池，不依赖
任何 SMI 命令。

## 配置

编辑 `bench_config.py`。

当前示例配置直接在 `MODELS` 中写完整参数，不使用生成函数：

- 每个模型直接声明 `tp`、`dp`、`dtype`、`enable_ep`、`server_args`。
- 每个模型直接声明自己的 `perf_cases`，性能 case 可以用数组简写。
- 每个模型直接声明自己的 `stability_cases`。
- 每个性能 case 会同时设置客户端 `--max-concurrency` 和服务端
  `--max-num-seqs`。
- VL prompt 会转换成 OpenAI-compatible chat messages，并使用
  `file://` 本地图片路径。

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
    "server_args": [
        "--max-model-len", "32768",
        "--limit-mm-per-prompt", '{"image": 4}',
    ],
    "mtp": {
        "enabled": True,
        "spec_method": "deepseek_mtp",
        "spec_tokens": 2,
    },
    "perf_cases": [
        [1, 1024, 1024],
        [8, 3024, 1024, 200, 0],
    ],
    "bench": {
        "backend": "openai-chat",
        "endpoint": "/v1/chat/completions",
        "dataset_name": "random-mm",
        "extra_args": [],
    },
}
```

`server_args` 可以直接写成 CLI 数组：

```python
"server_args": [
    "--enforce-eager",
    "--max-model-len", "32768",
]
```

也可以写成 dict，key 会自动转换成 vLLM CLI 参数。例如：

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

`perf_cases` 数组格式：

```python
[max_concurrency, input_len, output_len]
[max_concurrency, input_len, output_len, num_prompts]
[max_concurrency, input_len, output_len, num_prompts, temperature]
```

例如：

```python
"perf_cases": [
    [1, 1024, 1024],
    [8, 3024, 1024, 200, 0],
]
```

运行器会自动生成 case id，例如 `seq8_i3024_o1024`，并自动为服务端加入：

```bash
--max-num-seqs 8
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

GPU 分配使用静态 GPU 池。当前配置是：

```python
"gpu_indices": [0, 1, 2, 3],
```

调度器只看 `gpu_indices` 和本次进程内部的 reservation：

- `gpu_indices` 中的 GPU 都被视为可分配。
- 调度器会按任务所需 `tp * dp * pp` 数量分配 GPU。
- 本工具启动的多个任务不会互相抢同一张卡。
- 如果剩余 GPU 不够，任务会等待其他任务释放。
- 不检查外部进程是否正在使用 GPU。

运行前需要确认 `gpu_indices` 中的卡没有被外部任务占用。如果机器不是 4 卡，
请按实际情况修改 `gpu_indices`。

查看当前静态 GPU 池：

```bash
vmb --config bench_config.py gpu
```

如果服务端需要使用非 `CUDA_VISIBLE_DEVICES` 的环境变量控制可见设备，可以在
`GLOBAL` 中配置：

```python
"gpu_visible_devices_env": "CUDA_VISIBLE_DEVICES"
```

示例：

- `gpu_indices=[0,1,2,3]` 可以同时运行两个 `tp=2, dp=1` 的任务。
- 一个 `tp=4, dp=1` 的任务会占用全部 4 张 GPU。
- 一个开启 EP 的 MoE 任务，如果配置为 `tp=1, dp=8`，则需要 8 张 GPU。

## 命令

查看任务计划：

```bash
vmb --config bench_config.py plan --kind all
```

查看 GPU 探测诊断：

```bash
vmb --config bench_config.py gpu
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

## 进度日志

运行 `vmb run ...` 时，终端会持续打印结构化进度日志。每行包含 UTC 时间、
事件名、task id 和关键字段，例如：

```text
2026-06-17T08:00:00+00:00 | task_queued | task=perf_qwen3_32b_seq1_i1024_o1024 | kind=perf | model=qwen3_32b | case=seq1_i1024_o1024 | required_gpus=2
2026-06-17T08:00:00+00:00 | waiting_gpu | task=perf_qwen3_32b_seq1_i1024_o1024 | required_gpus=2
2026-06-17T08:00:00+00:00 | gpu_snapshot | task=perf_qwen3_32b_seq1_i1024_o1024 | idle_gpus=0,1 | reserved_gpus=none | available_gpus=0,1
2026-06-17T08:00:00+00:00 | gpu_allocated | task=perf_qwen3_32b_seq1_i1024_o1024 | gpus=0,1 | port=8000
2026-06-17T08:00:01+00:00 | server_starting | task=perf_qwen3_32b_seq1_i1024_o1024 | model=qwen3_32b | port=8000 | gpus=0,1
2026-06-17T08:03:10+00:00 | server_ready | task=perf_qwen3_32b_seq1_i1024_o1024 | url=http://127.0.0.1:8000/v1/models
2026-06-17T08:03:11+00:00 | perf_start | task=perf_qwen3_32b_seq1_i1024_o1024 | raw_json=...
2026-06-17T08:08:30+00:00 | perf_result_written | task=perf_qwen3_32b_seq1_i1024_o1024 | status=ok | csv=...
2026-06-17T08:08:35+00:00 | task_done | task=perf_qwen3_32b_seq1_i1024_o1024
```

常见事件：

- `task_queued`：任务已展开并进入调度队列。
- `waiting_gpu` / `gpu_wait`：资源不足，正在等待空闲 GPU。
- `gpu_snapshot`：当前空闲、已预留和可用 GPU。
- `gpu_allocated` / `gpu_released`：GPU 和端口分配/释放。
- `server_starting` / `server_wait_ready` / `server_ready`：服务端启动和健康检查。
- `server_command`：完整服务端命令，包含 `CUDA_VISIBLE_DEVICES`。
- `perf_client_command`：性能测试客户端命令，即 `vllm bench serve ...`。
- `stability_client_command`：稳定性测试客户端命令，即等价 `curl ...`。
- `perf_result_written`：性能 CSV 和原始 JSON 写入完成。
- `stability_report_written`：稳定性 report 写入完成。
- `task_done` / `task_failed`：任务完成或失败。

每个 task 的命令和日志也会落盘到：

```text
results/runs/<task_id>/
  manifest.json
  server_command.txt
  client_command.txt
  server.stdout.log
  server.stderr.log
  bench.stdout.log
  bench.stderr.log
```

## 稳定性验证

稳定性测试的目标是检查：同一个模型配置、同一个 prompt 或 messages、同一套
采样参数下，模型输出是否和历史 baseline 一致。

执行流程：

- 启动对应模型的 `vllm serve`。
- 按 `STABILITY_CASES` 构造请求，发送到 `/v1/completions` 或
  `/v1/chat/completions`。
- 从 OpenAI-compatible response 中提取输出文本。
- 对输出做 normalize：去掉首尾空白、统一换行。
- 计算 normalized output 的 hash。
- 和 baseline JSON 中保存的 hash 或文本进行比较。
- 如果发生变化，保存 current JSON、report JSON 和 unified diff。

baseline 不只保存 hash。hash 适合快速判断“有没有变化”，但只保存 hash 会导致
无法定位变化内容，也无法区分换行、标点、小格式变化和真正的语义变化。因此
本项目保存的是完整结构化 baseline：

```json
{
  "schema_version": 2,
  "task_id": "stability_dense_qwen_7b_basic_qa",
  "model_id": "dense_qwen_7b",
  "model_config_hash": "...",
  "case_id": "basic_qa",
  "case_config_hash": "...",
  "request_hash": "...",
  "output_hash": "...",
  "request": {},
  "response": {},
  "raw_output": "...",
  "normalized_output": "...",
  "environment": {
    "python": "...",
    "platform": "...",
    "vllm_version": "..."
  }
}
```

默认比较模式是 `exact_normalized`，即比较 normalize 后输出的 hash。也可以在
case 里指定：

```python
"compare": {
    "mode": "exact_raw"          # 或 exact_normalized / hash
}
```

推荐稳定性 case 使用确定性采样参数，例如：

```python
"sampling": {
    "temperature": 0,
    "top_p": 1,
    "max_tokens": 128,
    "seed": 42,
}
```

## 结果目录

默认输出到 `results/`：

```text
results/
  raw/                         # vllm bench serve 生成的原始 JSON
  csv/perf_results.csv         # 追加写入的性能结果汇总
  runs/<task_id>/              # server 日志、bench 日志、manifest
  stability/
    baselines/                 # 稳定性测试 baseline JSON
    current/                   # 本次输出 JSON
    reports/                   # 对比报告和 diff
```

`perf_results.csv` 会保留原始 JSON 路径，方便每一行汇总结果都能追溯到
`vllm bench serve` 的原始输出。

## 说明

- 本项目不会自动调优 TP、DP、EP、dtype 或 MTP；配置文件是唯一来源。
- 如果不同 vLLM 版本的参数名有变化，可以把精确参数放到
  `server_args` 或 `bench.extra_args`。
- `PERF_CASES` 支持设置 `temperature`、`top_p`、`request_rate` 等采样和压测
  参数，这些参数会转发给 `vllm bench serve`。
- 稳定性测试会使用 case 中的采样参数。为了做严格回归检查，建议使用
  `temperature=0`，并在模型支持时固定 `seed`。
- 旧版本生成的 `.txt` baseline 仍可读取对比；刷新 baseline 后会写成新的
  JSON 格式。
