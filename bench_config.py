"""Benchmark configuration.

MODELS is the source of truth.

perf_cases shorthand:
    [max_concurrency, input_len, output_len]
    [max_concurrency, input_len, output_len, num_prompts]
    [max_concurrency, input_len, output_len, num_prompts, temperature]

The runner auto-generates perf case ids such as seq8_i1024_o1024 and also
sets server-side --max-num-seqs from max_concurrency.
"""

GLOBAL = {
    "host": "127.0.0.1",
    "base_port": 8000,
    "result_dir": "results",
    "poll_interval_sec": 30,
    "ready_timeout_sec": 1800,
    "gpu_indices": [0, 1],
    "gpu_visible_devices_env": "CUDA_VISIBLE_DEVICES",
    "default_backend": "vllm",
    "default_endpoint": "/v1/completions",
    "save_detailed": True,
}


MODELS = [
    {
        "id": "qwen3_32b",
        "type": "dense",
        "model": "/models/Qwen3-32B",
        "served_model_name": "/models/Qwen3-32B",
        "tp": 2,
        "dp": 1,
        "dtype": "float16",
        "enable_ep": False,
        "server_args": ["--enforce-eager"],
        "perf_cases": [
            [1, 1024, 1024],
            [1, 3024, 1024],
            [8, 1024, 1024],
            [8, 3024, 1024],
        ],
        "stability_cases": [
            {
                "id": "qa_1_max128",
                "prompt": "中国的首都是哪里？",
                "expected": "北京",
                "sampling": {"temperature": 0, "top_p": 1, "max_tokens": 128, "seed": 42},
                "compare": {"mode": "exact_normalized"},
            },
            {
                "id": "qa_2_max128",
                "prompt": "1+1等于几？",
                "expected": "2",
                "sampling": {"temperature": 0, "top_p": 1, "max_tokens": 128, "seed": 42},
                "compare": {"mode": "exact_normalized"},
            },
        ],
    },
    {
        "id": "qwen3_32b_gptq_int8",
        "type": "dense",
        "model": "/models/Qwen3-32B-GPTQ-Int8",
        "served_model_name": "/models/Qwen3-32B-GPTQ-Int8",
        "tp": 2,
        "dp": 1,
        "dtype": "float16",
        "enable_ep": False,
        "server_args": ["--enforce-eager"],
        "perf_cases": [
            [1, 1024, 1024],
            [1, 3024, 1024],
            [8, 1024, 1024],
            [8, 3024, 1024],
        ],
        "stability_cases": [
            {
                "id": "qa_1_max128",
                "prompt": "中国的首都是哪里？",
                "expected": "北京",
                "sampling": {"temperature": 0, "top_p": 1, "max_tokens": 128, "seed": 42},
                "compare": {"mode": "exact_normalized"},
            },
            {
                "id": "qa_2_max128",
                "prompt": "1+1等于几？",
                "expected": "2",
                "sampling": {"temperature": 0, "top_p": 1, "max_tokens": 128, "seed": 42},
                "compare": {"mode": "exact_normalized"},
            },
        ],
    },
    {
        "id": "qwen3_next_80b_a3b_instruct",
        "type": "moe",
        "model": "/models/Qwen3-Next-80B-A3B-Instruct",
        "served_model_name": "/models/Qwen3-Next-80B-A3B-Instruct",
        "tp": 4,
        "dp": 1,
        "dtype": "float16",
        "enable_ep": False,
        "server_args": ["--enforce-eager"],
        "perf_cases": [
            [1, 1024, 1024],
            [1, 3024, 1024],
            [8, 1024, 1024],
            [8, 3024, 1024],
        ],
        "stability_cases": [
            {
                "id": "qa_1_max128",
                "prompt": "中国的首都是哪里？",
                "expected": "北京",
                "sampling": {"temperature": 0, "top_p": 1, "max_tokens": 128, "seed": 42},
                "compare": {"mode": "exact_normalized"},
            },
            {
                "id": "qa_2_max128",
                "prompt": "1+1等于几？",
                "expected": "2",
                "sampling": {"temperature": 0, "top_p": 1, "max_tokens": 128, "seed": 42},
                "compare": {"mode": "exact_normalized"},
            },
        ],
    },
    {
        "id": "qwen3_vl_30b_a3b_instruct",
        "type": "vl",
        "model": "/models/Qwen3-VL-30B-A3B-Instruct",
        "served_model_name": "/models/Qwen3-VL-30B-A3B-Instruct",
        "tp": 2,
        "dp": 1,
        "dtype": "float16",
        "enable_ep": False,
        "server_args": [
            "--allowed-local-media-path",
            "/workspace",
        ],
        "bench": {
            "backend": "openai-chat",
            "endpoint": "/v1/chat/completions",
            "dataset_name": "random-mm",
            "extra_args": [
                "--random-mm-base-items-per-request",
                "1",
                "--random-mm-limit-mm-per-prompt",
                '{"image": 2, "video": 0}',
            ],
        },
        "perf_cases": [
            [8, 1024, 1024],
            [8, 3024, 1024],
        ],
        "stability_cases": [
            {
                "id": "vl_1_max128",
                "endpoint": "/v1/chat/completions",
                "expected": "一只小狗站在草地上",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "请详细描述下这个图片。"},
                            {
                                "type": "image_url",
                                "image_url": {"url": "file:///workspace/paddleocr_vl_demo_s.png"},
                            },
                        ],
                    }
                ],
                "sampling": {"temperature": 0, "top_p": 1, "max_tokens": 128, "seed": 42},
                "compare": {"mode": "exact_normalized"},
            }
        ],
    },
    {
        "id": "qwen3_vl_32b_instruct",
        "type": "vl",
        "model": "/models/Qwen3-VL-32B-Instruct",
        "served_model_name": "/models/Qwen3-VL-32B-Instruct",
        "tp": 2,
        "dp": 1,
        "dtype": "float16",
        "enable_ep": False,
        "server_args": [
            "--allowed-local-media-path",
            "/workspace",
        ],
        "bench": {
            "backend": "openai-chat",
            "endpoint": "/v1/chat/completions",
            "dataset_name": "random-mm",
            "extra_args": [
                "--random-mm-base-items-per-request",
                "1",
                "--random-mm-limit-mm-per-prompt",
                '{"image": 2, "video": 0}',
            ],
        },
        "perf_cases": [
            [8, 1024, 1024],
            [8, 3024, 1024],
        ],
        "stability_cases": [
            {
                "id": "vl_1_max128",
                "endpoint": "/v1/chat/completions",
                "expected": "一只小狗站在草地上",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "请详细描述下这个图片。"},
                            {
                                "type": "image_url",
                                "image_url": {"url": "file:///workspace/paddleocr_vl_demo_s.png"},
                            },
                        ],
                    }
                ],
                "sampling": {"temperature": 0, "top_p": 1, "max_tokens": 128, "seed": 42},
                "compare": {"mode": "exact_normalized"},
            }
        ],
    },
    {
        "id": "qwen3_6_35b_a3b_workspace",
        "type": "moe",
        "model": "/models/Qwen3.6-35B-A3B",
        "served_model_name": "/models/Qwen3.6-35B-A3B",
        "tp": 2,
        "dp": 1,
        "dtype": "bfloat16",
        "enable_ep": True,
        "server_args": [
            "--allowed-local-media-path",
            "/workspace",
        ],
        "perf_cases": [
            [8, 1024, 1024],
            [8, 3024, 1024],
        ],
        "stability_cases": [
            {
                "id": "qa_1_max128",
                "prompt": "中国的首都是哪里？",
                "expected": "北京",
                "sampling": {"temperature": 0, "top_p": 1, "max_tokens": 128, "seed": 42},
                "compare": {"mode": "exact_normalized"},
            },
            {
                "id": "qa_2_max128",
                "prompt": "1+1等于几？",
                "expected": "2",
                "sampling": {"temperature": 0, "top_p": 1, "max_tokens": 128, "seed": 42},
                "compare": {"mode": "exact_normalized"},
            },
        ],
    },
    {
        "id": "qwen3_6_27b",
        "type": "dense",
        "model": "/models/Qwen3.6-27B",
        "served_model_name": "/models/Qwen3.6-27B",
        "tp": 2,
        "dp": 1,
        "dtype": "bfloat16",
        "enable_ep": False,
        "server_args": [],
        "perf_cases": [
            [1, 1024, 1024],
            [1, 3024, 1024],
            [8, 1024, 1024],
            [8, 3024, 1024],
        ],
        "stability_cases": [
            {
                "id": "qa_1_max128",
                "prompt": "中国的首都是哪里？",
                "expected": "北京",
                "sampling": {"temperature": 0, "top_p": 1, "max_tokens": 128, "seed": 42},
                "compare": {"mode": "exact_normalized"},
            },
            {
                "id": "qa_2_max128",
                "prompt": "1+1等于几？",
                "expected": "2",
                "sampling": {"temperature": 0, "top_p": 1, "max_tokens": 128, "seed": 42},
                "compare": {"mode": "exact_normalized"},
            },
        ],
    },
    {
        "id": "qwen3_6_35b_a3b_ep",
        "type": "moe",
        "model": "/models/Qwen3.6-35B-A3B",
        "served_model_name": "/models/Qwen3.6-35B-A3B",
        "tp": 2,
        "dp": 1,
        "dtype": "bfloat16",
        "enable_ep": True,
        "server_args": [],
        "perf_cases": [
            [1, 1024, 1024],
            [1, 3024, 1024],
            [8, 1024, 1024],
            [8, 3024, 1024],
        ],
        "stability_cases": [
            {
                "id": "qa_1_max128",
                "prompt": "中国的首都是哪里？",
                "expected": "北京",
                "sampling": {"temperature": 0, "top_p": 1, "max_tokens": 128, "seed": 42},
                "compare": {"mode": "exact_normalized"},
            },
            {
                "id": "qa_2_max128",
                "prompt": "1+1等于几？",
                "expected": "2",
                "sampling": {"temperature": 0, "top_p": 1, "max_tokens": 128, "seed": 42},
                "compare": {"mode": "exact_normalized"},
            },
        ],
    },
]

PERF_CASES = []
STABILITY_CASES = []
