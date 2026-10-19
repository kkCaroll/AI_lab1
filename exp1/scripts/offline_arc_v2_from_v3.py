"""由 v3 逐条预测还原 v2 严格口径的 ARC 准确率，不加载模型。

v3 的前四级与 v2 逐条相同，第五与第六级只接管 v2 会判失败的样本，故 v2 的判对数
等于 v3 的判对数减去由后两级接住且判对者。v3 判失败的样本在 v2 下同样失败。

用法示例
    cd exp1
    python -m scripts.offline_arc_v2_from_v3 --pred runs/P3-3/M4-eval-f16/predictions_arc.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

V3_ONLY = {"lead-bare", "body-anywhere"}


def load(path: Path) -> list[dict]:
    """读取逐条明细。"""
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True, type=Path)
    args = parser.parse_args()

    records = load(args.pred)
    v3_correct = sum(record["correct"] for record in records)
    rescued = [record for record in records if record["extract_method"] in V3_ONLY]
    rescued_correct = sum(record["correct"] for record in rescued)
    v2_correct = v3_correct - rescued_correct
    methods = Counter(record["extract_method"] for record in records)

    print(f"样本 {len(records)}")
    print(f"v3 判对 {v3_correct}，准确率 {v3_correct / len(records):.4f}")
    print(f"后两级接住 {len(rescued)} 条，其中判对 {rescued_correct} 条")
    print(f"v2 判对 {v2_correct}，准确率 {v2_correct / len(records):.4f}")
    print("规则命中 " + "，".join(f"{name} {count}" for name, count in methods.most_common()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
