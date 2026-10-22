"""汇总五种方法的逐条预测，计算区间与配对检验，不加载模型。

先核对行数、样本集合与判对数，再按数据集计算 Wilson 区间、McNemar 精确检验
与 Holm 校正，写出 JSON 与 Markdown 两份结果。

用法示例
    cd exp1
    python -m scripts.analyze_results
    python -m scripts.analyze_results --manifest configs/formal_runs.json
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime
from pathlib import Path

Z95 = 1.959963984540054
EXP1_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXP1_ROOT.parent
REQUIRED_FIELDS = {"index", "id", "correct", "gen_tokens", "truncated", "extract_status"}
FAMILY_NAMES = {"baseline": "基线", "methods": "方法间"}


def load_json(path: Path) -> dict:
    """读取 JSON。"""
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict]:
    """读取逐条明细。"""
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def resolve(raw: str | Path) -> Path:
    """清单内的相对路径以 exp1 为根。"""
    path = Path(raw)
    return path if path.is_absolute() else EXP1_ROOT / path


def wilson(correct: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson 得分区间。"""
    p = correct / n
    denominator = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denominator
    low = 0.0 if correct == 0 else max(0.0, center - half)
    high = 1.0 if correct == n else min(1.0, center + half)
    return low, high


def mcnemar_exact(a_only: int, b_only: int) -> float:
    """McNemar 精确检验，双侧。a_only 为仅前者判对，b_only 为仅后者判对。"""
    n = a_only + b_only
    if n == 0:
        return 1.0
    k = min(a_only, b_only)
    log_denominator = n * math.log(2.0)
    terms = (
        math.exp(
            math.lgamma(n + 1)
            - math.lgamma(i + 1)
            - math.lgamma(n - i + 1)
            - log_denominator
        )
        for i in range(k + 1)
    )
    return min(1.0, 2.0 * math.fsum(terms))


def paired_diff_ci(
    a_only: int, b_only: int, n: int, z: float = Z95
) -> tuple[float, float, float]:
    """配对比例差及其 95% 区间。"""
    diff = (a_only - b_only) / n
    variance = (a_only + b_only) - (a_only - b_only) ** 2 / n
    standard_error = math.sqrt(max(variance, 0.0)) / n
    return diff, max(-1.0, diff - z * standard_error), min(1.0, diff + z * standard_error)


def holm(pvalues: list[float]) -> list[float]:
    """Holm 校正，返回与输入同序的校正 p 值。"""
    m = len(pvalues)
    order = sorted(range(m), key=lambda index: pvalues[index])
    adjusted = [1.0] * m
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[index]))
        adjusted[index] = running
    return adjusted


def check_metrics(
    path: Path, n: int, correct: int, dataset: str, method: str, failures: list[str]
) -> None:
    """核对指标文件与逐条明细是否一致。"""
    if not path.exists():
        failures.append(f"{dataset} {method} 指标文件不存在 {path}")
        return
    metrics = load_json(path)
    if metrics.get("samples") != n:
        failures.append(f"{dataset} {method} 指标样本数 {metrics.get('samples')} 期望 {n}")
    if metrics.get("correct") != correct:
        failures.append(f"{dataset} {method} 指标判对数 {metrics.get('correct')} 期望 {correct}")


def check_dataset(
    dataset: str, spec: dict, runs: dict, methods: list[str]
) -> tuple[dict[str, list[dict]], list[str]]:
    """核对一个数据集下各方法的逐条预测。"""
    n = spec["n"]
    expected = spec["expected_correct"]
    records: dict[str, list[dict]] = {}
    failures: list[str] = []
    reference_ids: list[str] | None = None

    for method in methods:
        path = resolve(runs[method][dataset])
        if not path.exists():
            failures.append(f"{dataset} {method} 文件不存在 {path}")
            continue
        rows = load_jsonl(path)
        records[method] = rows
        if not rows:
            failures.append(f"{dataset} {method} 文件为空")
            continue
        if len(rows) != n:
            failures.append(f"{dataset} {method} 行数 {len(rows)} 期望 {n}")
        indices = [row.get("index") for row in rows]
        if sorted(indices) != list(range(n)):
            failures.append(f"{dataset} {method} index 不是 0 至 {n - 1} 的排列")
        ids = [row.get("id") for row in rows]
        if len(set(ids)) != len(ids):
            failures.append(f"{dataset} {method} id 有重复")
        missing = REQUIRED_FIELDS - set(rows[0])
        if missing:
            failures.append(f"{dataset} {method} 缺字段 {sorted(missing)}")
        if any(row.get("dataset") != dataset for row in rows):
            failures.append(f"{dataset} {method} dataset 字段不一致")
        correct = sum(bool(row.get("correct")) for row in rows)
        if correct != expected[method]:
            failures.append(f"{dataset} {method} 判对 {correct} 期望 {expected[method]}")
        if any(row.get("extract_status") not in {"OK", "WARN", "FAIL"} for row in rows):
            failures.append(f"{dataset} {method} extract_status 取值非法")
        if any(not isinstance(row.get("truncated"), bool) for row in rows):
            failures.append(f"{dataset} {method} truncated 字段非布尔")
        if any(not isinstance(row.get("gen_tokens"), int) or row["gen_tokens"] < 0 for row in rows):
            failures.append(f"{dataset} {method} gen_tokens 字段非法")
        if reference_ids is None:
            reference_ids = ids
        elif ids != reference_ids:
            failures.append(f"{dataset} {method} id 顺序与 {methods[0]} 不一致")

        metrics_path = runs[method].get("metrics", {}).get(dataset)
        if metrics_path:
            check_metrics(resolve(metrics_path), n, correct, dataset, method, failures)

    return records, failures


def check_prompt_consistency(runs: dict, methods: list[str], failures: list[str]) -> None:
    """GSM8K 各方法的模板指纹应相同。"""
    shas = {}
    for method in methods:
        raw = runs[method].get("metrics", {}).get("gsm8k")
        if not raw:
            continue
        path = resolve(raw)
        if not path.exists():
            continue
        shas[method] = load_json(path).get("prompt_sha256")
    if any(value is None for value in shas.values()):
        failures.append("GSM8K 指标缺 prompt_sha256")
    if len(set(shas.values())) > 1:
        failures.append("GSM8K 各方法的 prompt_sha256 不一致")


def check_arc_rule_version(runs: dict, methods: list[str], failures: list[str]) -> None:
    """ARC 的 v3 指标文件应统一标 v3。"""
    for method in methods:
        raw = runs[method].get("metrics", {}).get("arc")
        if not raw:
            continue
        path = resolve(raw)
        if not path.exists():
            continue
        version = load_json(path).get("extract_rules_version")
        if version != "v3":
            failures.append(f"ARC {method} 抽取规则版本 {version} 非 v3")


def pair_stats(a: str, b: str, n: int, records: dict[str, list[dict]]) -> dict:
    """两个方法在同一批样本上的配对统计。"""
    a_correct = [bool(row["correct"]) for row in records[a]]
    b_correct = [bool(row["correct"]) for row in records[b]]
    a_only = sum(x and not y for x, y in zip(a_correct, b_correct))
    b_only = sum((not x) and y for x, y in zip(a_correct, b_correct))
    diff, low, high = paired_diff_ci(a_only, b_only, n)
    return {
        "a": a,
        "b": b,
        "diff": diff,
        "diff_ci": [low, high],
        "a_only": a_only,
        "b_only": b_only,
        "p_raw": mcnemar_exact(a_only, b_only),
    }


def analyze_dataset(dataset: str, methods: list[str], records: dict[str, list[dict]], n: int) -> dict:
    """计算单个数据集的描述统计与两族配对检验。"""
    stats = {}
    for method in methods:
        rows = records[method]
        correct = sum(bool(row["correct"]) for row in rows)
        low, high = wilson(correct, n)
        stats[method] = {
            "correct": correct,
            "accuracy": correct / n,
            "wilson_ci": [low, high],
            "mean_gen_tokens": sum(row["gen_tokens"] for row in rows) / n,
            "truncation_rate": sum(bool(row["truncated"]) for row in rows) / n,
            "extract_failure_rate": sum(row["extract_status"] == "FAIL" for row in rows) / n,
            "extract_warn_rate": sum(row["extract_status"] == "WARN" for row in rows) / n,
        }

    baseline = [(method, "M0") for method in methods if method != "M0"]
    method_pairs = [
        ("M4", "M2"),
        ("M4", "M1"),
        ("M4", "M3"),
        ("M2", "M1"),
        ("M3", "M1"),
        ("M3", "M2"),
    ]
    pairs = []
    for family, definitions in (("baseline", baseline), ("methods", method_pairs)):
        group = [pair_stats(a, b, n, records) for a, b in definitions]
        for item, adjusted in zip(group, holm([item["p_raw"] for item in group])):
            item["family"] = family
            item["p_holm"] = adjusted
            item["significant"] = adjusted < 0.05
        pairs.extend(group)

    return {"n": n, "methods": stats, "pairs": pairs}


def format_p(value: float) -> str:
    """p 值格式，极小值不写零。"""
    return "<0.0001" if value < 1e-4 else f"{value:.4f}"


def build_markdown(summary: dict, manifest: dict, generated_at: str) -> str:
    """生成人可读的统计表。"""
    labels = manifest["labels"]
    methods = manifest["methods"]
    lines = [
        "# 统计分析与配对检验",
        "",
        f"本文件由 `scripts/analyze_results.py` 于 {generated_at} 生成，输入清单为 `configs/formal_runs.json`。数值与 `runs/analysis/summary.json` 一致，正文不手工修改。",
        "",
        "## 一 素材核对",
        "",
        "| 数据集 | 方法 | 行数 | 判对数 | 期望判对数 | 状态 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for dataset in ("gsm8k", "arc"):
        result = summary["datasets"][dataset]
        for method in methods:
            correct = result["methods"][method]["correct"]
            expected = result["expected_correct"][method]
            lines.append(f"| {dataset} | {labels[method]} | {result['n']} | {correct} | {expected} | OK |")

    lines += ["", "## 二 准确率与置信区间", ""]
    for index, dataset in enumerate(("gsm8k", "arc"), start=1):
        title = "GSM8K" if dataset == "gsm8k" else "ARC-Challenge"
        lines += [
            f"### 2.{index} {title}",
            "",
            "| 方法 | 判对 | 准确率 | Wilson 95% 区间 | 生成均长 | 截断率 | 抽取失败率 |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        result = summary["datasets"][dataset]
        for method in methods:
            item = result["methods"][method]
            low, high = item["wilson_ci"]
            lines.append(
                f"| {labels[method]} | {item['correct']} | {item['accuracy']:.4f} | "
                f"{low:.4f} 至 {high:.4f} | {item['mean_gen_tokens']:.1f} | "
                f"{item['truncation_rate']:.4f} | {item['extract_failure_rate']:.4f} |"
            )
        lines.append("")

    lines += [
        "## 三 配对检验",
        "",
        "差值一律为前者减后者，单位为百分点。p 为 McNemar 精确检验双侧值，Holm 列为同一数据集同一家族内的校正值。",
        "",
    ]
    for index, dataset in enumerate(("gsm8k", "arc"), start=1):
        title = "GSM8K" if dataset == "gsm8k" else "ARC-Challenge"
        lines += [
            f"### 3.{index} {title}",
            "",
            "| 家族 | 对比 | 差值 pp | 95% 区间 pp | 仅前者对 | 仅后者对 | p | Holm p | 判定 |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for item in summary["datasets"][dataset]["pairs"]:
            low, high = item["diff_ci"]
            verdict = "显著" if item["significant"] else "不显著"
            lines.append(
                f"| {FAMILY_NAMES[item['family']]} | {labels[item['a']]} 与 {labels[item['b']]} | "
                f"{item['diff'] * 100:+.2f} | {low * 100:+.2f} 至 {high * 100:+.2f} | "
                f"{item['a_only']} | {item['b_only']} | {format_p(item['p_raw'])} | "
                f"{format_p(item['p_holm'])} | {verdict} |"
            )
        lines.append("")

    lines += [
        "## 四 口径说明",
        "",
        "- Wilson 区间按得分法计算，比例接近 0 或 1 时优于正态近似。",
        "- McNemar 精确检验只使用两法判断不一致的样本，仅前者对与仅后者对分别记 a_only 与 b_only。",
        "- 差值区间按配对比例差的标准误计算，标准误为 sqrt((a_only+b_only)-(a_only-b_only)^2/n)/n。",
        "- Holm 校正分两个家族，基线家族为四种微调方法各与 Base Model 比较，方法间家族为四种微调方法两两比较，每个数据集分别校正。",
        "- 截断率与抽取失败率只作描述，不作显著性推断。",
        "- ARC 一律取 v3 逐条预测，v2 严格口径不入本表。",
        "- GSM8K 的抽取规则版本在 M0 至 M3 为 v2、M4 为 v3，v3 只改 ARC 级联，GSM8K 判对不受影响。",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=EXP1_ROOT / "configs/formal_runs.json")
    parser.add_argument("--output-dir", type=Path, default=EXP1_ROOT / "runs/analysis")
    parser.add_argument(
        "--record", type=Path, default=REPO_ROOT / "RECORD/03-统计分析与配对检验.md"
    )
    args = parser.parse_args()

    manifest = load_json(args.manifest)
    methods = manifest["methods"]
    records: dict[str, dict[str, list[dict]]] = {}
    failures: list[str] = []
    for dataset, spec in manifest["datasets"].items():
        dataset_records, dataset_failures = check_dataset(
            dataset, spec, manifest["runs"], methods
        )
        records[dataset] = dataset_records
        failures.extend(dataset_failures)
    check_prompt_consistency(manifest["runs"], methods, failures)
    check_arc_rule_version(manifest["runs"], methods, failures)

    if failures:
        for failure in failures:
            print(f"FAIL {failure}")
        print(f"核对未通过，共 {len(failures)} 项")
        return 1

    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    datasets = {}
    for dataset, spec in manifest["datasets"].items():
        datasets[dataset] = analyze_dataset(dataset, methods, records[dataset], spec["n"])
        datasets[dataset]["expected_correct"] = spec["expected_correct"]

    summary = {
        "generated_at": generated_at,
        "manifest": str(args.manifest),
        "methods": methods,
        "labels": manifest["labels"],
        "datasets": datasets,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "summary.json"
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(build_markdown(summary, manifest, generated_at), encoding="utf-8")

    print(f"OK 核对 {len(methods) * len(manifest['datasets'])} 份预测")
    for dataset, result in datasets.items():
        print(f"{dataset} 样本 {result['n']}")
        for method in methods:
            item = result["methods"][method]
            low, high = item["wilson_ci"]
            print(
                f"  {manifest['labels'][method]} 判对 {item['correct']} "
                f"准确率 {item['accuracy']:.4f} 区间 {low:.4f} 至 {high:.4f}"
            )
    print(f"已写出 {json_path}")
    print(f"已写出 {args.record}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
