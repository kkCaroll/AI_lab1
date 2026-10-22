"""在固定通用文本上测量语言模型困惑度，不载入任何任务数据。

语料按不重叠窗口切分，每窗长度 1024 token，窗口内逐位预测下一位，首位因无上下文
不计。五个模型读同一份文本、同一组分词与同一组窗口，故困惑度可直接横向比较。绝对
值受预训练重叠影响，只解释相对差异。

损失按 float32 分块求和，不取模型自带的均值损失，以免不同框架版本的归约口径不同。
困惑度取负对数似然总和除以计分 token 数后再取指数。

用法示例
    cd exp1
    python -m scripts.measure_ppl --model-path models/Qwen3-0.6B \
        --corpus ../data/raw/wikitext2_train_2m.txt \
        --output-dir runs/ppl/M0 --dtype float32 --device cpu \
        --max-tokens 8192 --batch-size 1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F

DEFAULT_CORPUS = "../data/raw/wikitext2_train_2m.txt"
DEFAULT_MAX_TOKENS = 2_000_000
DEFAULT_WINDOW = 1024
DEFAULT_BATCH = 8
DEFAULT_CHUNK = 64


def plan_windows(total_tokens: int, window: int) -> tuple[int, int]:
    """返回完整窗口数与实际用到的 token 数，末尾不足一窗的余数丢弃。"""
    if window < 2:
        raise ValueError("窗口长度至少为 2")
    windows = total_tokens // window
    if windows == 0:
        raise ValueError(f"token 数 {total_tokens} 不足一个窗口 {window}")
    return windows, windows * window


def perplexity(nll_sum: float, scored_tokens: int) -> float:
    """由负对数似然总和与计分 token 数算困惑度。"""
    if scored_tokens <= 0:
        raise ValueError("计分 token 数为零")
    exponent = nll_sum / scored_tokens
    if exponent > 700.0:
        return float("inf")
    return math.exp(exponent)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="固定窗口困惑度测量")
    parser.add_argument("--model-path", required=True, help="基座或完整微调检查点目录")
    parser.add_argument("--adapter", default=None, help="PEFT 适配器目录，基座测量时留空")
    parser.add_argument("--corpus", default=DEFAULT_CORPUS, help="纯文本语料")
    parser.add_argument("--output-dir", required=True, help="本次测量的输出目录")
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--chunk", type=int, default=DEFAULT_CHUNK, help="损失求和的序列分块")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--dtype", default="float16", choices=["float32", "bfloat16", "float16"]
    )
    return parser.parse_args()


def _load(model_path: str, adapter: str | None, device, dtype):
    """载入分词器、模型与可选适配器，返回三者与加载秒数。"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=dtype)
    if adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter)
    model.to(device).eval()
    model.config.use_cache = False
    return tokenizer, model, time.perf_counter() - started


def _nll_sum(model, input_ids, chunk: int) -> torch.Tensor:
    """一个批次的负对数似然总和，按序列分块以 float32 计算。"""
    outputs = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids), use_cache=False)
    logits = outputs.logits[:, :-1]
    targets = input_ids[:, 1:]
    total = torch.zeros((), dtype=torch.float64, device=input_ids.device)
    for start in range(0, logits.shape[1], chunk):
        stop = start + chunk
        block = logits[:, start:stop].float()
        total += F.cross_entropy(
            block.reshape(-1, block.shape[-1]),
            targets[:, start:stop].reshape(-1),
            reduction="sum",
        ).double()
    return total


def main() -> int:
    args = _parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("未检测到可用的 CUDA 设备")
        return 2
    dtype = getattr(torch, args.dtype)
    if device.type == "cpu" and dtype is not torch.float32:
        print("处理器上的矩阵乘法不支持该精度，请显式传 --dtype float32")
        return 2

    corpus = Path(args.corpus)
    if not corpus.is_file():
        print(f"语料不存在  {corpus}")
        return 1
    model_dir = Path(args.model_path)
    if not (model_dir / "config.json").is_file():
        print(f"模型目录缺少 config.json  {model_dir}")
        return 1
    if args.adapter and not (Path(args.adapter) / "adapter_config.json").is_file():
        print(f"适配器目录缺少 adapter_config.json  {args.adapter}")
        return 1

    print(f"设备 {device}  精度 {dtype}  窗口 {args.window}  批 {args.batch_size}")
    print(f"语料 {corpus}")
    tokenizer, model, load_seconds = _load(args.model_path, args.adapter, device, dtype)
    print(f"  模型加载 {load_seconds:.1f} 秒")

    text = corpus.read_text(encoding="utf-8")
    ids = tokenizer(text, add_special_tokens=False, return_attention_mask=False)["input_ids"]
    available = len(ids)
    if available < args.max_tokens:
        print(f"语料仅 {available:,} token，不足上限 {args.max_tokens:,}，按实际长度测量")
    used_limit = min(available, args.max_tokens)

    try:
        windows, used = plan_windows(used_limit, args.window)
    except ValueError as error:
        print(f"窗口规划失败  {error}")
        return 1
    scored = windows * (args.window - 1)
    print(f"  可用 {available:,}  使用 {used:,}  窗口 {windows:,}  计分 token {scored:,}")

    batch_count = math.ceil(windows / args.batch_size)
    report_every = max(1, batch_count // 10)
    nll_total = 0.0
    processed = 0
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    try:
        with torch.inference_mode():
            for batch_index in range(batch_count):
                first = batch_index * args.batch_size
                last = min(first + args.batch_size, windows)
                rows = [
                    ids[item * args.window : (item + 1) * args.window] for item in range(first, last)
                ]
                input_ids = torch.tensor(rows, dtype=torch.long, device=device)
                nll_total += float(_nll_sum(model, input_ids, args.chunk))
                processed += input_ids.shape[0]
                if (batch_index + 1) % report_every == 0 or last == windows:
                    elapsed = time.perf_counter() - started
                    done_tokens = processed * (args.window - 1)
                    current = perplexity(nll_total, done_tokens)
                    speed = processed * args.window / elapsed
                    eta = (windows - processed) * args.window / speed if speed > 0 else 0.0
                    print(
                        f"  批 {batch_index + 1}/{batch_count}  窗口 {processed}/{windows}"
                        f"  当前困惑度 {current:.4f}  速度 {speed:.0f} token/s  余 {eta:.0f} 秒"
                    )
    except torch.cuda.OutOfMemoryError:
        print("显存不足，请下调 --batch-size 或 --window")
        return 1

    elapsed = time.perf_counter() - started
    value = perplexity(nll_total, scored)
    peak = torch.cuda.max_memory_allocated() / 1024**3 if device.type == "cuda" else 0.0
    result = {
        "model_path": args.model_path,
        "adapter": args.adapter,
        "corpus": str(corpus),
        "corpus_sha256": _sha256(corpus),
        "available_tokens": available,
        "used_tokens": used,
        "window": args.window,
        "stride": args.window,
        "windows": windows,
        "scored_tokens": scored,
        "nll_sum": nll_total,
        "perplexity": value,
        "elapsed_seconds": round(elapsed, 1),
        "tokens_per_second": round(used / elapsed, 1),
        "device": str(device),
        "dtype": str(dtype),
        "load_seconds": round(load_seconds, 1),
        "peak_memory_gib": round(peak, 2),
        "measured_at": datetime.now().isoformat(timespec="seconds"),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "ppl.json", "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)

    print("===== 结果 =====")
    print(f"  困惑度 {value:.4f}  负对数似然 {nll_total:.2f}  计分 token {scored:,}")
    print(f"  用时 {elapsed:.1f} 秒  速度 {used / elapsed:.0f} token/s  显存峰值 {peak:.2f} GiB")
    print(f"  已写出 {output_dir / 'ppl.json'}")
    if not math.isfinite(value):
        print("困惑度非有限值，判失败")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
