"""Benchmark online serving throughput.

On the server side, run one of the following commands:
    vLLM OpenAI API server
    python -m vllm.entrypoints.openai.api_server \
        --model <your_model> --swap-space 16 \
        --disable-log-requests

    (TGI backend)
    ./launch_tgi_server.sh <your_model> <max_batch_total_tokens>

On the client side, run:
    python benchmarks/benchmark_serving.py \
        --backend <backend> \
        --model <your_model> \
        --dataset-name sharegpt \
        --dataset-path <path to dataset> \
        --request-rate <request_rate> \ # By default <request_rate> is inf
        --num-prompts <num_prompts> # By default <num_prompts> is 1000
"""
import argparse
import asyncio
import json
import os
import random
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import AsyncGenerator, List, Tuple

import numpy as np
from backend_request_func import (ASYNC_REQUEST_FUNCS, RequestFuncInput,
                                  RequestFuncOutput)
from tqdm.asyncio import tqdm
from transformers import PreTrainedTokenizerBase, AutoTokenizer


@dataclass
class BenchmarkMetrics:
    completed: int
    total_input: int
    total_output: int
    request_throughput: float
    input_throughput: float
    output_throughput: float
    mean_input_len: int
    median_input_len: int
    max_input_len: int
    mean_output_len: int
    median_output_len: int
    max_output_len: int
    mean_latency_ms: float
    median_latency_ms: float
    p99_latency_ms: float
    mean_ttft_ms: float
    median_ttft_ms: float
    p99_ttft_ms: float
    mean_tpot_ms: float
    median_tpot_ms: float
    p99_tpot_ms: float

def build_chat_prompt(prompt, tokenizer):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    return text

# def build_chat_prompt2(prompt, context, tokenizer):
#     messages = [
#       {"role": "system", "content": "You need to read the text and anwser question: " + prompt},
#       {"role": "user", "content": context}
#     ]
#     text = tokenizer.apply_chat_template(
#         messages,
#         tokenize=False,
#         add_generation_prompt=True
#     )
#     return text

def sample_custom_requests(
    dataset_path: str,
    num_requests: int,
    max_output_len: int,
    tokenizer: PreTrainedTokenizerBase,
) -> List[Tuple[str, int, int]]:
    # Load the dataset.
    dataset=[]
    with open(dataset_path, "r") as f:
        for line in f:
            data = json.loads(line)
            dataset.append(data["query"])

    # Tokenize the prompts and completions.
    prompts = [build_chat_prompt(prompt, tokenizer) for prompt in dataset]
    prompt_token_ids = tokenizer(prompts).input_ids
    tokenized_dataset = []
    for i in range(len(dataset)):
        tokenized_dataset.append((prompts[i], len(prompt_token_ids[i]), max_output_len))

    # Sample the requests.
    sampled_requests = random.sample(tokenized_dataset, num_requests)
    # sampled_requests = tokenized_dataset[:num_requests]
    return sampled_requests

def sample_rand_requests(
    input_len: str,
    num_requests: int,
    max_output_len: int,
    tokenizer: PreTrainedTokenizerBase,
) -> List[Tuple[str, int, int]]:
    # Load the dataset.
    tokenized_dataset = []
    for i in range(1000):
        random_token_ids = np.random.randint(10000,size=(input_len))
        prompts = tokenizer.decode(random_token_ids)
        random_token_ids=tokenizer(prompts).input_ids
        tokenized_dataset.append((prompts, len(random_token_ids), max_output_len))

    sampled_requests = tokenized_dataset[:num_requests]
    return sampled_requests


def sample_gsm_requests(
    dataset_path: str,
    num_requests: int,
    max_output_len: int,
    tokenizer: PreTrainedTokenizerBase,
) -> List[Tuple[str, int, int]]:
    # Load the dataset.
    dataset=[]
    with open(dataset_path, "r") as f:
        for line in f:
            data = json.loads(line)
            dataset.append(data["question"])

    # Tokenize the prompts and completions.
    #prompts = [prompt for prompt in dataset]
    prompts = [build_chat_prompt(prompt,tokenizer) for prompt in dataset]
    prompt_token_ids = tokenizer(prompts).input_ids
    tokenized_dataset = []
    for i in range(len(dataset)):
        tokenized_dataset.append((prompts[i], len(prompt_token_ids[i]), max_output_len))

    # Sample the requests.
    # sampled_requests = random.sample(filtered_dataset, num_requests)
    sampled_requests = tokenized_dataset[:num_requests]
    return sampled_requests


def sample_long_requests(
    dataset_path: str,
    num_requests: int,
    max_output_len: int,
    tokenizer: PreTrainedTokenizerBase,
) -> List[Tuple[str, int, int]]:
    # Load the dataset.
    dataset=[]
    # for filepath in os.listdir(dataset_path):
    with open(dataset_path, "r") as f:
        for line in f:
            data = json.loads(line)
            prompt_clip = data["input"] + '\n' + data["context"]
            prompt_clip = prompt_clip[:max_output_len]
            dataset.append(prompt_clip)
        # data = json.load(f)
        # for elem in data:
        #     dataset.append(elem["instruction"])

    # Tokenize the prompts and completions.
    #prompts = [prompt for prompt in dataset]
    prompts = [build_chat_prompt(prompt,tokenizer) for prompt in dataset]
    prompt_token_ids = tokenizer(prompts).input_ids
    tokenized_dataset = []
    for i in range(len(dataset)):
        # if len(prompt_token_ids[i]) <= max_output_len:
        if len(prompt_token_ids[i]) >= 512:
            tokenized_dataset.append((prompts[i], len(prompt_token_ids[i]), max_output_len))

    # Sample the requests.
    # sampled_requests = random.sample(filtered_dataset, num_requests)
    sampled_requests = tokenized_dataset[:num_requests]
    return sampled_requests

def sample_sharegpt_requests(
    dataset_path: str,
    num_requests: int,
    tokenizer: PreTrainedTokenizerBase,
) -> List[Tuple[str, int, int]]:
    # Load the dataset.
    with open(dataset_path) as f:
        dataset = json.load(f)
    # Filter out the conversations with less than 2 turns.
    dataset = [data for data in dataset if len(data["conversations"]) >= 2]
    # Only keep the first two turns of each conversation.
    dataset = [(data["conversations"][0]["value"],
                data["conversations"][1]["value"]) for data in dataset]

    # some of these will be filtered out, so sample more than we need
    sampled_indices = random.sample(range(len(dataset)),
                                    int(num_requests * 1.2))
    dataset = [dataset[i] for i in sampled_indices]

    # Tokenize the prompts and completions.
    prompts = [prompt for prompt, _ in dataset]
    prompt_token_ids = tokenizer(prompts).input_ids
    completions = [completion for _, completion in dataset]
    completion_token_ids = tokenizer(completions).input_ids
    tokenized_dataset = []
    for i in range(len(dataset)):
        output_len = len(completion_token_ids[i])
        tokenized_dataset.append((prompts[i], prompt_token_ids[i], output_len))

    # Filter out too long sequences.
    filtered_dataset: List[Tuple[str, int, int]] = []
    for prompt, prompt_token_ids, output_len in tokenized_dataset:
        prompt_len = len(prompt_token_ids)
        if prompt_len < 4 or output_len < 4:
            # Prune too short sequences.
            # This is because TGI causes errors when the input or output length
            # is too short.
            continue
        if prompt_len > 1024 or prompt_len + output_len > 2048:
            # Prune too long sequences.
            continue
        filtered_dataset.append((prompt, prompt_len, output_len))

    # Sample the requests.
    sampled_requests = random.sample(filtered_dataset, num_requests)
    return sampled_requests


def sample_sonnet_requests(
    dataset_path: str,
    num_requests: int,
    input_len: int,
    output_len: int,
    prefix_len: int,
    tokenizer: PreTrainedTokenizerBase,
) -> List[Tuple[str, str, int, int]]:
    assert input_len > prefix_len, "input_len must be greater than prefix_len."

    # Load the dataset.
    with open(dataset_path) as f:
        poem_lines = f.readlines()

    # Tokenize the poem lines.
    poem_token_ids = tokenizer(poem_lines).input_ids
    average_poem_len = sum(
        len(token_ids) for token_ids in poem_token_ids) / len(poem_token_ids)

    # Base prefix for all requests.
    base_prompt = "Pick as many lines as you can from these poem lines:\n"
    base_message = [{
        "role": "user",
        "content": base_prompt,
    }]
    base_prompt_formatted = tokenizer.apply_chat_template(
        base_message, add_generation_prompt=True, tokenize=False)
    base_prompt_offset = len(tokenizer(base_prompt_formatted).input_ids)

    assert (input_len > base_prompt_offset
            ), f"Please set 'args.input-len' higher than {base_prompt_offset}."
    num_input_lines = round(
        (input_len - base_prompt_offset) / average_poem_len)

    # First approximately `prefix_len` number of tokens in the
    # prompt are fixed poem lines.
    assert (
        prefix_len > base_prompt_offset
    ), f"Please set 'args.prefix-len' higher than {base_prompt_offset}."

    num_prefix_lines = round(
        (prefix_len - base_prompt_offset) / average_poem_len)
    prefix_lines = poem_lines[:num_prefix_lines]

    # Sample the rest of lines per request.
    sampled_requests: List[Tuple[str, int, int]] = []
    for _ in range(num_requests):
        sampled_lines = "".join(
            prefix_lines +
            random.sample(poem_lines, num_input_lines - num_prefix_lines))

        prompt = f"{base_prompt}{sampled_lines}"
        message = [
            {
                "role": "user",
                "content": prompt,
            },
        ]
        prompt_formatted = tokenizer.apply_chat_template(
            message, add_generation_prompt=True, tokenize=False)
        prompt_len = len(tokenizer(prompt_formatted).input_ids)
        sampled_requests.append(
            (prompt, prompt_formatted, prompt_len, output_len))

    return sampled_requests


async def get_request(
    input_requests: List[Tuple[str, int, int]],
    request_rate: float,
) -> AsyncGenerator[Tuple[str, int, int], None]:
    input_requests = iter(input_requests)
    for request in input_requests:
        yield request

        if request_rate == float("inf"):
            # If the request rate is infinity, then we don't need to wait.
            continue
        # Sample the request interval from the exponential distribution.
        interval = np.random.exponential(1.0 / request_rate)
        # The next request will be sent after the interval.
        await asyncio.sleep(interval)


def calculate_metrics(
    input_requests: List[Tuple[str, int, int]],
    outputs: List[RequestFuncOutput],
    dur_s: float,
    tokenizer: PreTrainedTokenizerBase,
) -> Tuple[BenchmarkMetrics, List[int]]:
    actual_output_lens = []
    input_lens = []
    total_input = 0
    completed = 0
    tpots = []
    ttfts = []
    latency = []
    for i in range(len(outputs)):
        if outputs[i].success:
            output_len = len(outputs[i].itl) + 1
            actual_output_lens.append(output_len)
            input_lens.append(input_requests[i][1])
            total_input += input_requests[i][1]
            for tpot in outputs[i].itl:
                tpots.append(tpot)
            ttfts.append(outputs[i].ttft)
            latency.append(outputs[i].latency)
            completed += 1
        else:
            actual_output_lens.append(0)

    metrics = BenchmarkMetrics(
        completed=completed,
        total_input=total_input,
        total_output=sum(actual_output_lens),
        mean_input_len=np.mean(input_lens),
        median_input_len=np.median(input_lens),
        max_input_len=max(input_lens),
        mean_output_len=np.mean(actual_output_lens),
        median_output_len=np.median(actual_output_lens),
        max_output_len=max(actual_output_lens),
        request_throughput=completed / dur_s,
        input_throughput=total_input / dur_s,
        output_throughput=sum(actual_output_lens) / dur_s,
        mean_latency_ms=np.mean(latency) * 1000,
        median_latency_ms=np.median(latency) * 1000,
        p99_latency_ms=np.percentile(latency, 99) * 1000,
        mean_ttft_ms=np.mean(ttfts or 0) *
        1000,  # ttfts is empty if streaming is not supported by backend
        median_ttft_ms=np.median(ttfts or 0) * 1000,
        p99_ttft_ms=np.percentile(ttfts or 0, 99) * 1000,
        mean_tpot_ms=np.mean(tpots) * 1000,
        median_tpot_ms=np.median(tpots) * 1000,
        p99_tpot_ms=np.percentile(tpots, 99) * 1000,
    )

    return metrics, actual_output_lens


async def benchmark(
    backend: str,
    api_url: str,
    model_id: str,
    tokenizer: PreTrainedTokenizerBase,
    input_requests: List[Tuple[str, int, int]],
    best_of: int,
    use_beam_search: bool,
    request_rate: float,
    disable_tqdm: bool,
):
    if backend in ASYNC_REQUEST_FUNCS:
        request_func = ASYNC_REQUEST_FUNCS.get(backend)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    print(f"Traffic request rate: {request_rate}")

    pbar = None if disable_tqdm else tqdm(total=len(input_requests))

    benchmark_start_time = time.perf_counter()
    tasks = []
    async for request in get_request(input_requests, request_rate):
        prompt, prompt_len, output_len = request
        request_func_input = RequestFuncInput(
            model=model_id,
            prompt=prompt,
            api_url=api_url,
            prompt_len=prompt_len,
            output_len=output_len,
            best_of=best_of,
            use_beam_search=use_beam_search,
        )
        tasks.append(
            asyncio.create_task(
                request_func(request_func_input=request_func_input,
                             pbar=pbar)))
    outputs: List[RequestFuncOutput] = await asyncio.gather(*tasks)

    if not disable_tqdm:
        pbar.close()

    benchmark_duration = time.perf_counter() - benchmark_start_time

    metrics, actual_output_lens = calculate_metrics(
        input_requests=input_requests,
        outputs=outputs,
        dur_s=benchmark_duration,
        tokenizer=tokenizer,
    )

    print("{s:{c}^{n}}".format(s=' Serving Benchmark Result ', n=50, c='='))
    print("{:<40} {:<10}".format("Successful requests:", metrics.completed))
    print("{:<40} {:<10.2f}".format("Benchmark duration (s):",
                                    benchmark_duration))
    print("{:<40} {:<10}".format("Total input tokens:", metrics.total_input))
    print("{:<40} {:<10}".format("Total generated tokens:",
                                 metrics.total_output))
    print("{:<40} {:<10}".format("Mean input tokens:", metrics.mean_input_len))
    print("{:<40} {:<10}".format("Median input tokens:", metrics.median_input_len))
    print("{:<40} {:<10}".format("Max input tokens:", metrics.max_input_len))
    print("{:<40} {:<10}".format("Mean generated tokens:",
                                 metrics.mean_output_len))
    print("{:<40} {:<10}".format("Median generated tokens:",
                                 metrics.median_output_len))
    print("{:<40} {:<10}".format("Max generated tokens:",
                                 metrics.max_output_len))
    print("{s:{c}^{n}}".format(s='Throughput', n=50, c='-'))
    print("{:<40} {:<10.2f}".format("Request throughput (req/s):",
                                    metrics.request_throughput))
    print("{:<40} {:<10.2f}".format("Input token throughput (tok/s):",
                                    metrics.input_throughput))
    print("{:<40} {:<10.2f}".format("Output token throughput (tok/s):",
                                    metrics.output_throughput))
    print("{s:{c}^{n}}".format(s='Time to Latency', n=50, c='-'))
    print("{:<40} {:<10.2f}".format("Mean Latency (ms):", metrics.mean_latency_ms))
    print("{:<40} {:<10.2f}".format("Median Latency (ms):", metrics.median_latency_ms))
    print("{:<40} {:<10.2f}".format("P99 Latency (ms):", metrics.p99_latency_ms))
    print("{s:{c}^{n}}".format(s='Time to First Token', n=50, c='-'))
    print("{:<40} {:<10.2f}".format("Mean TTFT (ms):", metrics.mean_ttft_ms))
    print("{:<40} {:<10.2f}".format("Median TTFT (ms):",
                                    metrics.median_ttft_ms))
    print("{:<40} {:<10.2f}".format("P99 TTFT (ms):", metrics.p99_ttft_ms))
    print("{s:{c}^{n}}".format(s='Time per Output Token (excl. 1st token)',
                               n=50,
                               c='-'))
    print("{:<40} {:<10.2f}".format("Mean TPOT (ms):", metrics.mean_tpot_ms))
    print("{:<40} {:<10.2f}".format("Median TPOT (ms):",
                                    metrics.median_tpot_ms))
    print("{:<40} {:<10.2f}".format("P99 TPOT (ms):", metrics.p99_tpot_ms))
    print("=" * 50)

    result = {
        "duration": benchmark_duration,
        "completed": metrics.completed,
        "total_input_tokens": metrics.total_input,
        "total_output_tokens": metrics.total_output,
        "mean_input_len": metrics.mean_input_len,
        "median_input_len": metrics.median_input_len,
        "max_input_len": metrics.max_input_len,
        "mean_output_len": metrics.mean_output_len,
        "median_output_len": metrics.median_output_len,
        "max_output_len": metrics.max_output_len,
        "request_throughput": metrics.request_throughput,
        "input_throughput": metrics.input_throughput,
        "output_throughput": metrics.output_throughput,
        "mean_latency_ms": metrics.mean_latency_ms,
        "median_latency_ms": metrics.median_latency_ms,
        "p99_latency_ms": metrics.p99_latency_ms,
        "mean_ttft_ms": metrics.mean_ttft_ms,
        "median_ttft_ms": metrics.median_ttft_ms,
        "p99_ttft_ms": metrics.p99_ttft_ms,
        "mean_tpot_ms": metrics.mean_tpot_ms,
        "median_tpot_ms": metrics.median_tpot_ms,
        "p99_tpot_ms": metrics.p99_tpot_ms,
        "input_lens": [output.prompt_len for output in outputs],
        "output_lens": actual_output_lens,
        "ttfts": [output.ttft for output in outputs],
        #"itls": [output.itl for output in outputs],
        "generated_texts": [output.generated_text for output in outputs],
        "errors": [output.error for output in outputs],
    }
    return result


def main(args: argparse.Namespace):
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)

    backend = args.backend
    model_id = args.model
    tokenizer_id = args.tokenizer if args.tokenizer is not None else args.model

    if args.base_url is not None:
        api_url = f"{args.base_url}{args.endpoint}"
    else:
        api_url = f"http://{args.host}:{args.port}{args.endpoint}"

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id,
                              trust_remote_code=args.trust_remote_code)

    if args.dataset is not None:
        warnings.warn(
            "The '--dataset' argument will be deprecated in the next "
            "release. Please use '--dataset-name' and "
            "'--dataset-path' in the future runs.",
            stacklevel=2)
        input_requests = sample_sharegpt_requests(
            dataset_path=args.dataset,
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
        )

    elif args.dataset_name == "random":
        input_requests = sample_rand_requests(
            input_len=args.input_len,
            num_requests=args.num_prompts,
            max_output_len=args.output_len,
            tokenizer=tokenizer,
        )

    elif args.dataset_name == "GSM":
        input_requests = sample_gsm_requests(
            dataset_path=args.dataset_path,
            num_requests=args.num_prompts,
            max_output_len=args.output_len,
            tokenizer=tokenizer,
        )

    elif args.dataset_name == 'long':
        input_requests = sample_long_requests(
            dataset_path=args.dataset_path,
            num_requests=args.num_prompts,
            max_output_len=args.output_len,
            tokenizer=tokenizer,
        )

    elif args.dataset_name == "custom":
        input_requests = sample_custom_requests(
            dataset_path=args.dataset_path,
            num_requests=args.num_prompts,
            max_output_len=args.output_len,
            tokenizer=tokenizer,
        )

    elif args.dataset_name == "sharegpt":
        input_requests = sample_sharegpt_requests(
            dataset_path=args.dataset_path,
            num_requests=args.num_prompts,
            tokenizer=tokenizer,
        )

    elif args.dataset_name == "sonnet":
        # Do not format the prompt, pass to message directly
        if args.backend == "openai-chat":
            input_requests = sample_sonnet_requests(
                dataset_path=args.dataset_path,
                num_requests=args.num_prompts,
                input_len=args.input_len,
                output_len=args.output_len,
                prefix_len=args.prefix_len,
                tokenizer=tokenizer,
            )
            input_requests = [(prompt, prompt_len, output_len)
                              for prompt, prompt_formatted, prompt_len,
                              output_len in input_requests]
        else:
            assert (
                tokenizer.chat_template or tokenizer.default_chat_template
            ), "Tokenizer/model must have chat template for sonnet dataset."
            input_requests = sample_sonnet_requests(
                dataset_path=args.dataset_path,
                num_requests=args.num_prompts,
                input_len=args.input_len,
                output_len=args.output_len,
                prefix_len=args.prefix_len,
                tokenizer=tokenizer,
            )
            input_requests = [(prompt_formatted, prompt_len, output_len)
                              for prompt, prompt_formatted, prompt_len,
                              output_len in input_requests]

    else:
        raise ValueError(f"Unknown dataset: {args.dataset_name}")

    benchmark_result = asyncio.run(
        benchmark(
            backend=backend,
            api_url=api_url,
            model_id=model_id,
            tokenizer=tokenizer,
            input_requests=input_requests,
            best_of=args.best_of,
            use_beam_search=args.use_beam_search,
            request_rate=args.request_rate,
            disable_tqdm=args.disable_tqdm,
        ))

    # Save config and results to json
    if args.save_result:
        result_json = {}

        # Setup
        current_dt = datetime.now().strftime("%Y%m%d-%H%M%S")
        result_json["date"] = current_dt
        result_json["backend"] = backend
        result_json["model_id"] = model_id
        result_json["tokenizer_id"] = tokenizer_id
        result_json["best_of"] = args.best_of
        result_json["use_beam_search"] = args.use_beam_search
        result_json["num_prompts"] = args.num_prompts

        # Metadata
        if args.metadata:
            for item in args.metadata:
                if "=" in item:
                    kvstring = item.split("=")
                    result_json[kvstring[0].strip()] = kvstring[1].strip()
                else:
                    raise ValueError(
                        "Invalid metadata format. Please use KEY=VALUE format."
                    )

        # Traffic
        result_json["request_rate"] = (
            args.request_rate if args.request_rate < float("inf") else "inf")

        # Merge with benchmark result
        result_json = {**result_json, **benchmark_result}

        # Save to file
        base_model_id = model_id.split("/")[-1]
        file_name = f"{backend}-{args.request_rate}qps-{args.num_prompts}prompts-{base_model_id}-{current_dt}.json"  #noqa
        if args.result_dir:
            file_name = os.path.join(args.result_dir, file_name)
        with open(file_name, "w") as outfile:
            json.dump(result_json, outfile, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the online serving throughput.")
    parser.add_argument(
        "--backend",
        type=str,
        default="vllm",
        choices=list(ASYNC_REQUEST_FUNCS.keys()),
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Server or API base url if not using http host and port.",
    )
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--endpoint",
        type=str,
        default="/v1/completions",
        help="API endpoint.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to the ShareGPT dataset, will be deprecated in the "
        "next release.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="sharegpt",
        choices=["random","sharegpt", "sonnet", "GSM", "custom", "long"],
        help="Name of the dataset to benchmark on.",
    )
    parser.add_argument("--dataset-path",
                        type=str,
                        default=None,
                        help="Path to the dataset.")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Name of the model.",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="",
        help="MindIE-LLM Model ame",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        help=
        "Name or path of the tokenizer, if not using the default tokenizer.",
    )
    parser.add_argument(
        "--best-of",
        type=int,
        default=1,
        help="Generates `best_of` sequences per prompt and "
        "returns the best one.",
    )
    parser.add_argument("--use-beam-search", action="store_true")
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=1000,
        help="Number of prompts to process.",
    )
    parser.add_argument(
        "--input-len",
        type=int,
        default=256,
        help=
        "Number of input tokens per request",
    )
    parser.add_argument(
        "--output-len",
        type=int,
        default=2048,
        help=
        "Number of output tokens per request, used only for sonnet dataset.",
    )
    parser.add_argument(
        "--sonnet-prefix-len",
        type=int,
        default=200,
        help=
        "Number of prefix tokens per request, used only for sonnet dataset.",
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Number of requests per second. If this is inf, "
        "then all the requests are sent at time 0. "
        "Otherwise, we use Poisson process to synthesize "
        "the request arrival times.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code from huggingface",
    )
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        help="Specify to disable tqdm progress bar.",
    )
    parser.add_argument(
        "--save-result",
        action="store_true",
        help="Specify to save benchmark results to a json file",
    )
    parser.add_argument(
        "--metadata",
        metavar="KEY=VALUE",
        nargs="*",
        help="Key-value pairs (e.g, --metadata version=0.3.3 tp=1) "
        "for metadata of this run to be saved in the result JSON file "
        "for record keeping purposes.",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default=None,
        help="Specify directory to save benchmark json results."
        "If not specified, results are saved in the current directory.",
    )

    args = parser.parse_args()
    main(args)
