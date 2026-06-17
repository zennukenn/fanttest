"""Example benchmark configuration.

Edit this file to describe the models and workloads you want to run.
The runner intentionally keeps resource scheduling simple: it only looks at
currently idle GPUs and the configured TP/DP/PP/EP values.
"""

GLOBAL = {
    "host": "127.0.0.1",
    "base_port": 8000,
    "result_dir": "results",
    "poll_interval_sec": 30,
    "ready_timeout_sec": 1800,
    "idle_memory_threshold_mb": 1024,
    "default_backend": "vllm",
    "default_endpoint": "/v1/completions",
    "save_detailed": True,
}

MODELS = [
    {
        "id": "dense_qwen_7b",
        "type": "dense",
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "served_model_name": "dense_qwen_7b",
        "tp": 1,
        "dp": 1,
        "dtype": "bfloat16",
        "server_args": {
            "gpu_memory_utilization": 0.9,
            "max_model_len": 32768,
        },
        "extra_server_args": [],
    },
    {
        "id": "moe_with_ep_mtp",
        "type": "moe",
        "model": "deepseek-ai/DeepSeek-V3-0324",
        "served_model_name": "moe_with_ep_mtp",
        "tp": 1,
        "dp": 8,
        "enable_ep": True,
        "dtype": "bfloat16",
        "server_args": {
            "enable_eplb": True,
        },
        "mtp": {
            "enabled": True,
            "spec_method": "deepseek_mtp",
            "spec_tokens": 2,
            # You can use speculative_config instead when your vLLM version
            # expects one JSON argument:
            # "speculative_config": {"method": "deepseek_mtp", "num_speculative_tokens": 2},
        },
        "extra_server_args": [],
    },
    {
        "id": "vl_qwen_7b",
        "type": "vl",
        "model": "Qwen/Qwen2.5-VL-7B-Instruct",
        "served_model_name": "vl_qwen_7b",
        "tp": 2,
        "dp": 1,
        "dtype": "bfloat16",
        "server_args": {
            "max_model_len": 32768,
            "limit_mm_per_prompt": {"image": 4},
        },
        "bench": {
            "backend": "openai-chat",
            "endpoint": "/v1/chat/completions",
            "dataset_name": "random-mm",
            "extra_args": [
                "--random-mm-base-items-per-request", "1",
                "--random-mm-limit-mm-per-prompt", '{"image": 2, "video": 0}',
            ],
        },
    },
]

PERF_CASES = [
    {
        "id": "i512_o128_c16",
        "input_len": 512,
        "output_len": 128,
        "num_prompts": 200,
        "max_concurrency": 16,
        "temperature": 0,
    },
    {
        "id": "i2048_o256_c16",
        "input_len": 2048,
        "output_len": 256,
        "num_prompts": 200,
        "max_concurrency": 16,
        "temperature": 0,
    },
]

STABILITY_CASES = [
    {
        "id": "basic_qa",
        "prompt": "请用三句话解释 KV cache 的作用。",
        "sampling": {
            "temperature": 0,
            "top_p": 1,
            "max_tokens": 128,
            "seed": 42,
        },
        "compare": {
            "mode": "exact_normalized",
        },
    },
    # For VL/chat models you can provide OpenAI-compatible messages directly:
    # {
    #     "id": "vl_image_case",
    #     "endpoint": "/v1/chat/completions",
    #     "messages": [
    #         {
    #             "role": "user",
    #             "content": [
    #                 {"type": "text", "text": "Describe this image."},
    #                 {"type": "image_url", "image_url": {"url": "file:///path/to/image.jpg"}},
    #             ],
    #         }
    #     ],
    #     "sampling": {"temperature": 0, "max_tokens": 128},
    # },
]
