"""从 WikiText-2 原始划分抽取固定长度的通用文本，供困惑度测量。

取 train 划分的前若干 token 而非 test 或 validation。train 约二百一十万 token，
足以取满二百万；test 与 validation 合计仅约四十六万。该划分属预训练语料，绝对
困惑度会偏低，故结果只用于五个模型之间的相对比较，不与文献绝对值对照。

输出为纯文本，测量端用同一分词器重新编码并截断到同一 token 数。写盘前把 token
序列解码回文本，使文件大小与目标 token 数大致相称，同时保留可读性。

用法示例
    cd exp1
    python -m scripts.make_ppl_corpus \
        --input ../data/external/wikitext2/train-00000-of-00001.parquet \
        --output ../data/raw/wikitext2_train_2m.txt
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

DEFAULT_INPUT = "../data/external/wikitext2/train-00000-of-00001.parquet"
DEFAULT_OUTPUT = "../data/raw/wikitext2_train_2m.txt"
DEFAULT_MODEL = "models/Qwen3-0.6B"
DEFAULT_TOKENS = 2_000_000


def _read_parquet(path: Path) -> str:
    """读出 text 列并用换行连接。WikiText 原文自带段落换行，空行保留。"""
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    if "text" not in table.column_names:
        raise ValueError(f"{path} 缺少 text 列，实际列为 {table.column_names}")
    return "\n".join(table.column("text").to_pylist())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="抽取 WikiText-2 固定 token 数的通用文本")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="WikiText-2 parquet 文件")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="输出纯文本")
    parser.add_argument("--model-path", default=DEFAULT_MODEL, help="提供分词器的模型目录")
    parser.add_argument("--tokens", type=int, default=DEFAULT_TOKENS, help="目标 token 数")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    source = Path(args.input)
    if not source.is_file():
        print(f"输入不存在  {source}")
        return 1

    print(f"读取 {source}")
    text = _read_parquet(source)
    print(f"  字符 {len(text):,}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    ids = tokenizer(text, add_special_tokens=False, return_attention_mask=False)["input_ids"]
    print(f"  可用 token {len(ids):,}")
    if len(ids) < args.tokens:
        print(f"可用 token 不足目标 {args.tokens:,}，不写出")
        return 1

    selected = ids[: args.tokens]
    decoded = tokenizer.decode(
        selected, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(decoded, encoding="utf-8")

    reencoded = tokenizer(decoded, add_special_tokens=False, return_attention_mask=False)[
        "input_ids"
    ]
    print(f"写出 {output}")
    print(f"  目标 token {args.tokens:,}  回编码 {len(reencoded):,}  字符 {len(decoded):,}")
    print(f"  sha256 {_sha256(output)}")
    print(
        "  "
        + json.dumps(
            {
                "source": source.name,
                "source_chars": len(text),
                "target_tokens": args.tokens,
                "reencoded_tokens": len(reencoded),
                "output_chars": len(decoded),
                "sha256": _sha256(output),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
