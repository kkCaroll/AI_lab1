"""开发集划分脚本。

从 GSM8K 训练集以固定种子划出开发集，只记索引，不复制样本。开发集供选择学习率、
轮数与正则系数，不取测试集，以免测试集信息经选择过程泄漏进模型。

原始文件不改动，正式训练集由 src/data.py 的 load_train_split 按索引剔除。索引文件
同时记下源文件的 SHA256，供载入端核对索引与源文件是否仍然对齐。

2026-11-11 按课程要求改为 9 比 1，开发集 747 条。为不使已选定的超参失效，取扩展
而非重抽，即保留原 400 条索引，再从未入选的样本中补抽 347 条，使旧开发集始终不进入
训练。记录中的 previous_dev_size 与 kept_count 两项留作审计。

本脚本只在本机运行，产物 exp1/configs/dev_split.json 入仓库，容器内不再执行，故
直接复用 src/utils.py 的报告机制而不另写一份。

用法示例
    cd exp1
    python -m scripts.make_dev_split --data-dir ../data/raw --extend-from configs/dev_split.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from src.utils import FAIL, OK, report, section, sha256_of, summarize, write_json

SEED = 42
DEV_SIZE = 747
SOURCE_FILE = "gsm8k_train.jsonl"
EXPECTED_TOTAL = 7473
PREVIEW = 8


def main() -> int:
    parser = argparse.ArgumentParser(description="从 GSM8K 训练集划出开发集索引")
    parser.add_argument("--data-dir", default="../data/raw", help="原始 jsonl 目录")
    parser.add_argument("--output", default="configs/dev_split.json", help="索引文件写出位置")
    parser.add_argument("--seed", type=int, default=SEED, help="划分所用随机种子")
    parser.add_argument("--dev-size", type=int, default=DEV_SIZE, help="开发集条数")
    parser.add_argument(
        "--extend-from",
        default=None,
        help="在已有索引文件基础上扩抽，保留其全部索引，只补足差额",
    )
    args = parser.parse_args()

    checks = []
    section("源文件")

    source = Path(args.data_dir) / SOURCE_FILE
    if not source.is_file():
        report(checks, "源文件可读", FAIL, str(source))
        return summarize(checks)

    checksum = sha256_of(source)
    with source.open(encoding="utf-8") as handle:
        total = sum(1 for _ in handle)
    report(checks, "源文件行数", OK if total == EXPECTED_TOTAL else FAIL, f"{total}，期望 {EXPECTED_TOTAL}")
    print(f"  源文件校验和 {checksum}")

    section("划分")
    if not 0 < args.dev_size < total:
        report(checks, "开发集规模", FAIL, f"{args.dev_size} 不在 0 与 {total} 之间")
        return summarize(checks)

    # 用 random.Random 的独立实例，不用 datasets 的 shuffle，令划分只依赖种子与
    # Python 版本，不依赖第三方库的行为。索引排序后写出，使文件内容与 diff 稳定。
    rng = random.Random(args.seed)
    kept: list[int] = []
    if args.extend_from:
        previous = json.loads(Path(args.extend_from).read_text(encoding="utf-8"))
        if previous.get("source_sha256") != checksum:
            report(checks, "原索引与源文件一致", FAIL, "校验和不同")
            return summarize(checks)
        kept = list(previous["indices"])
        if len(kept) != previous.get("dev_size") or len(set(kept)) != len(kept):
            report(checks, "原索引自洽", FAIL, f"{len(kept)} 条")
            return summarize(checks)
        if not 0 < len(kept) < args.dev_size:
            report(checks, "扩抽额度", FAIL, f"原 {len(kept)} 条，目标 {args.dev_size} 条")
            return summarize(checks)
        report(checks, "原索引可复用", OK, f"{len(kept)} 条")

    pool = [i for i in range(total) if i not in set(kept)]
    indices = sorted(kept + rng.sample(pool, args.dev_size - len(kept)))
    report(checks, "开发集规模", OK, f"{args.dev_size} 条")

    section("写出")
    record = {
        "dataset": "gsm8k",
        "split": "train",
        "source_file": SOURCE_FILE,
        "source_sha256": checksum,
        "seed": args.seed,
        "total": total,
        "dev_size": args.dev_size,
        "train_size": total - args.dev_size,
        "previous_dev_size": len(kept) if args.extend_from else None,
        "kept_count": len(kept),
        "indices": indices,
    }
    output = Path(args.output)
    write_json(record, output)
    print(f"  已写出 {output}")
    print(f"  索引前 {PREVIEW} 项 {indices[:PREVIEW]}")

    return summarize(checks)


if __name__ == "__main__":
    raise SystemExit(main())
