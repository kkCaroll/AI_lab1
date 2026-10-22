"""由统计产物生成报告所需的表格片段与矢量图。

表三至表七与三幅图取自 `runs/analysis/summary.json`、`runs/P4-ppl/*/ppl.json`
与 `运行输出日志/P3/` 的五份任务日志。脚本只做取位与排版，不改数值；缺文件或
读数与预期不符即非零退出，不写半套产物。

用法示例
    cd exp1
    python -m scripts.export_tables
    python -m scripts.export_tables --out-dir ../REPORT/tables --fig-dir ../REPORT/figures

表一、表二与表八为人工记录，直接在正文排版，不由本脚本生成。
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

EXP1_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXP1_ROOT.parent

METHODS = ["M0", "M1", "M2", "M3", "M4"]
LABELS = {
    "M0": "基座",
    "M1": "全参微调",
    "M2": "LoRA",
    "M3": "P-Tuning v2",
    "M4": "自研方法",
}
DATASET_NAMES = {"gsm8k": "GSM8K", "arc": "ARC-Challenge"}
# 参考调色板前五槽，已通过相邻对 CVD 校验
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
PPL_METHOD_ORDER = METHODS

BETA_READINGS = {0.05: 196, 0.2: 195, 0.8: 243, 1.6: 241, 3.2: 233}
BETA_LOG_PATTERN = re.compile(
    r"开发集\s+准确率\s+([\d.]+)\s+正确\s+(\d+)/400\s+均长\s+([\d.]+)\s+token"
)
BETA_PATTERN = re.compile(r"kl([0-9.]+)")


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def fmt4(value: float) -> str:
    return f"{value:.4f}"


def fmt_pp(value: float) -> str:
    return f"{value * 100:.2f}"


def fmt_p(value: float) -> str:
    return "$<$0.0001" if value < 0.0001 else f"{value:.4f}"


def fmt_ci(low: float, high: float) -> str:
    return f"{low:.4f}--{high:.4f}"


def parse_beta_logs(log_dir: Path) -> list[dict]:
    """解析 P3 各臂开发集读数，返回按系数升序的五条。"""
    found: dict[float, dict] = {}
    for path in sorted(log_dir.glob("*.txt")):
        match = BETA_PATTERN.search(path.name)
        if not match:
            continue
        beta = float(match.group(1))
        text = path.read_text(encoding="utf-8", errors="replace")
        lines = BETA_LOG_PATTERN.findall(text)
        if not lines:
            continue
        accuracy, correct, length = lines[-1]
        record = {
            "beta": beta,
            "correct": int(correct),
            "accuracy": float(accuracy),
            "mean_gen_tokens": float(length),
            "source": path.name,
        }
        if beta in found and found[beta]["source"].startswith("ai-lab1-R3"):
            continue
        if beta in found and not path.name.startswith("ai-lab1-R3"):
            raise ValueError(f"系数 {beta} 有两份读数  {found[beta]['source']} 与 {path.name}")
        found[beta] = record

    missing = set(BETA_READINGS) - set(found)
    if missing:
        raise ValueError(f"缺少系数 {sorted(missing)} 的日志")
    rows = [found[beta] for beta in sorted(found)]
    for row in rows:
        if row["correct"] != BETA_READINGS[row["beta"]]:
            raise ValueError(
                f"系数 {row['beta']} 的正确数 {row['correct']} 与记录的 "
                f"{BETA_READINGS[row['beta']]} 不符  {row['source']}"
            )
    return rows


def table_main(datasets: dict) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{五种方法在两个测试集上的准确率与 Wilson 区间}",
        r"\label{tab:main}",
        r"\begin{tabular}{lSlSl}",
        r"\toprule",
        r"\multirow{2}{*}{方法} & \multicolumn{2}{c}{GSM8K} & \multicolumn{2}{c}{ARC-Challenge} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r" & {准确率} & {95\% 区间} & {准确率} & {95\% 区间} \\",
        r"\midrule",
    ]
    for method in METHODS:
        gsm = datasets["gsm8k"]["methods"][method]
        arc = datasets["arc"]["methods"][method]
        lines.append(
            f"{LABELS[method]} & {fmt4(gsm['accuracy'])} & "
            f"{fmt_ci(*gsm['wilson_ci'])} & {fmt4(arc['accuracy'])} & "
            f"{fmt_ci(*arc['wilson_ci'])} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    return "\n".join(lines)


def table_paired(datasets: dict) -> str:
    def pair(dataset: str, a: str, b: str) -> dict:
        for item in datasets[dataset]["pairs"]:
            if item["a"] == a and item["b"] == b:
                return item
        raise KeyError(f"缺少配对 {a} 与 {b}")

    rows = [
        ("M1", "M0", r"全参微调\ 相对\ 基座"),
        ("M2", "M0", r"LoRA 相对\ 基座"),
        ("M3", "M0", r"P-Tuning v2 相对\ 基座"),
        ("M4", "M0", r"自研方法\ 相对\ 基座"),
        ("M4", "M2", r"自研方法\ 相对\ LoRA"),
        ("M4", "M1", r"自研方法\ 相对\ 全参微调"),
        ("M4", "M3", r"自研方法\ 相对\ P-Tuning v2"),
    ]
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{同批测试样本上的配对检验，差值为前者减后者}",
        r"\label{tab:paired}",
        r"\begin{tabular}{lSlSl}",
        r"\toprule",
        r"\multirow{2}{*}{对比} & \multicolumn{2}{c}{GSM8K} & \multicolumn{2}{c}{ARC-Challenge} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
        r" & {差值} & {Holm $p$} & {差值} & {Holm $p$} \\",
        r"\midrule",
    ]
    for a, b, label in rows:
        gsm = pair("gsm8k", a, b)
        arc = pair("arc", a, b)
        lines.append(
            f"{label} & {fmt_pp(gsm['diff'])} & {fmt_p(gsm['p_holm'])} & "
            f"{fmt_pp(arc['diff'])} & {fmt_p(arc['p_holm'])} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{minipage}{\linewidth}",
        r"\vspace{0.3em}\footnotesize 差值单位为百分点。$p$ 为 McNemar 精确检验双侧值，",
        r"经同一数据集同一家族内的 Holm 校正。",
        r"\end{minipage}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def table_behavior(datasets: dict) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{输出行为指标，截断与抽取失败均按样本计}",
        r"\label{tab:behavior}",
        r"\begin{tabular}{lSSSS}",
        r"\toprule",
        r"方法 & {GSM8K 均长} & {ARC 均长} & {ARC 截断率} & {ARC 抽取失败率} \\",
        r"\midrule",
    ]
    for method in METHODS:
        gsm = datasets["gsm8k"]["methods"][method]
        arc = datasets["arc"]["methods"][method]
        lines.append(
            f"{LABELS[method]} & {gsm['mean_gen_tokens']:.1f} & "
            f"{arc['mean_gen_tokens']:.1f} & {arc['truncation_rate']:.4f} & "
            f"{arc['extract_failure_rate']:.4f} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{minipage}{\linewidth}",
        r"\vspace{0.3em}\footnotesize 均长单位为 token。抽取失败率取 v3 口径。",
        r"\end{minipage}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def table_beta(rows: list[dict]) -> str:
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{正则系数五档扫描在 400 条开发集上的读数}",
        r"\label{tab:beta}",
        r"\begin{tabular}{SSSS}",
        r"\toprule",
        r"{系数} & {正确数} & {准确率} & {生成均长} \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['beta']:g} & {row['correct']} & {row['accuracy']:.4f} & "
            f"{row['mean_gen_tokens']:.1f} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{minipage}{\linewidth}",
        r"\vspace{0.3em}\footnotesize 正确数以 400 条为分母，均长单位为 token。",
        r"\end{minipage}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def table_ppl(ppl_root: Path) -> str:
    values = {}
    for method in PPL_METHOD_ORDER:
        path = ppl_root / method / "ppl.json"
        if not path.is_file():
            raise FileNotFoundError(f"缺少困惑度结果  {path}")
        values[method] = float(load_json(path)["perplexity"])
    base = values["M0"]
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{五个模型在 WikiText-2 前二百万 token 上的困惑度}",
        r"\label{tab:ppl}",
        r"\begin{tabular}{lSS}",
        r"\toprule",
        r"方法 & {困惑度} & {相对基座} \\",
        r"\midrule",
    ]
    for method in PPL_METHOD_ORDER:
        lines.append(
            f"{LABELS[method]} & {values[method]:.2f} & {values[method] / base:.3f} \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{minipage}{\linewidth}",
        r"\vspace{0.3em}\footnotesize P-Tuning v2 的前缀仅在对话模板格式的输入上训练，",
        r"与无模板语料不匹配，其读数只作描述，不参与方法排序。",
        r"\end{minipage}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def _setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "legend.frameon": False,
        }
    )
    return plt


def figure_accuracy(datasets: dict, path: Path) -> None:
    plt = _setup_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for axis, dataset in zip(axes, ("gsm8k", "arc")):
        for index, method in enumerate(METHODS):
            item = datasets[dataset]["methods"][method]
            low, high = item["wilson_ci"]
            axis.errorbar(
                index,
                item["accuracy"],
                yerr=[[item["accuracy"] - low], [high - item["accuracy"]]],
                fmt="o",
                color=COLORS[index],
                markersize=4.5,
                capsize=2.5,
                elinewidth=1.2,
                markeredgecolor="white",
                markeredgewidth=0.6,
                zorder=3,
            )
            axis.annotate(
                f"{item['accuracy']:.4f}",
                (index, high),
                textcoords="offset points",
                xytext=(0, 4),
                ha="center",
                fontsize=6.5,
                color="#52514e",
            )
        axis.set_xticks(range(len(METHODS)))
        axis.set_xticklabels([LABELS[m] for m in METHODS], fontsize=8)
        axis.set_title(DATASET_NAMES[dataset], fontsize=9, pad=6)
        axis.set_ylim(0.20, 0.68)
        axis.grid(axis="y", color="#d9d8d4", linewidth=0.6, linestyle=(0, (1, 3)))
        axis.set_axisbelow(True)
    axes[1].axhline(0.2502, color="#8a8984", linewidth=0.8, linestyle="--", zorder=1)
    axes[1].annotate(
        "随机猜测 0.2502",
        (0.98, 0.2502),
        xycoords=("axes fraction", "data"),
        xytext=(0, 3),
        textcoords="offset points",
        ha="right",
        fontsize=6.5,
        color="#52514e",
    )
    axes[0].set_ylabel("准确率")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def figure_beta(rows: list[dict], path: Path) -> None:
    plt = _setup_matplotlib()
    betas = [row["beta"] for row in rows]
    fig, (top, bottom) = plt.subplots(
        2, 1, figsize=(4.6, 3.9), sharex=True, gridspec_kw={"hspace": 0.18}
    )
    top.plot(betas, [row["correct"] for row in rows], marker="o", color=COLORS[0],
             linewidth=1.6, markersize=4.5)
    bottom.plot(betas, [row["mean_gen_tokens"] for row in rows], marker="o",
                color=COLORS[1], linewidth=1.6, markersize=4.5)
    for axis in (top, bottom):
        axis.set_xscale("log")
        axis.set_xticks(betas)
        axis.set_xticklabels([f"{b:g}" for b in betas])
        axis.grid(axis="y", color="#d9d8d4", linewidth=0.6, linestyle=(0, (1, 3)))
        axis.set_axisbelow(True)
        axis.axvline(1.6, color="#8a8984", linewidth=0.9, linestyle="--", zorder=1)
    top.set_ylabel("开发集正确数")
    bottom.set_ylabel("生成均长")
    bottom.set_xlabel("正则系数")
    top.set_ylim(150, 270)
    bottom.set_ylim(80, 220)
    top.axhline(254, color="#8a8984", linewidth=0.8, linestyle=(0, (4, 3)), zorder=1)
    bottom.axhline(208.4, color="#8a8984", linewidth=0.8, linestyle=(0, (4, 3)), zorder=1)
    top.annotate("基座 254", (0.05, 254), xycoords=("data", "data"),
                 xytext=(0, 4), textcoords="offset points", ha="left",
                 fontsize=7, color="#52514e")
    bottom.annotate("基座 208.4", (0.05, 208.4), xycoords=("data", "data"),
                    xytext=(0, 4), textcoords="offset points", ha="left",
                    fontsize=7, color="#52514e")
    top.annotate("选用 1.6", (1.6, top.get_ylim()[1]), xytext=(4, -12),
                 textcoords="offset points", ha="left", fontsize=7, color="#52514e")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def figure_behavior(datasets: dict, path: Path) -> None:
    plt = _setup_matplotlib()
    import numpy as np

    truncation = [datasets["arc"]["methods"][m]["truncation_rate"] for m in METHODS]
    failure = [datasets["arc"]["methods"][m]["extract_failure_rate"] for m in METHODS]
    positions = np.arange(len(METHODS))
    width = 0.36
    fig, axis = plt.subplots(figsize=(5.6, 2.8))
    bars = [
        axis.bar(positions - width / 2 - 0.01, truncation, width, label="截断率",
                 color=COLORS[0], edgecolor="white", linewidth=0.8),
        axis.bar(positions + width / 2 + 0.01, failure, width, label="抽取失败率",
                 color=COLORS[1], edgecolor="white", linewidth=0.8),
    ]
    for group in bars:
        for bar in group:
            height = bar.get_height()
            axis.annotate(f"{height:.3f}", (bar.get_x() + bar.get_width() / 2, height),
                          xytext=(0, 2), textcoords="offset points",
                          ha="center", fontsize=6.5, color="#52514e")
    axis.set_xticks(positions)
    axis.set_xticklabels([LABELS[m] for m in METHODS], fontsize=8)
    axis.set_ylabel("比例")
    axis.set_ylim(0, 0.52)
    axis.grid(axis="y", color="#d9d8d4", linewidth=0.6, linestyle=(0, (1, 3)))
    axis.set_axisbelow(True)
    axis.legend(fontsize=7.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="生成报告的表格与图")
    parser.add_argument("--summary", default=str(EXP1_ROOT / "runs/analysis/summary.json"))
    parser.add_argument("--ppl-root", default=str(EXP1_ROOT / "runs/P4-ppl"))
    parser.add_argument("--p3-logs", default=str(REPO_ROOT / "运行输出日志/P3"))
    parser.add_argument("--out-dir", default=str(REPO_ROOT / "REPORT/tables"))
    parser.add_argument("--fig-dir", default=str(REPO_ROOT / "REPORT/figures"))
    args = parser.parse_args()

    summary = load_json(Path(args.summary))
    datasets = summary["datasets"]
    beta_rows = parse_beta_logs(Path(args.p3_logs))

    out_dir = Path(args.out_dir)
    fig_dir = Path(args.fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    tables = {
        "03-main.tex": table_main(datasets),
        "04-paired.tex": table_paired(datasets),
        "05-behavior.tex": table_behavior(datasets),
        "06-beta.tex": table_beta(beta_rows),
        "07-ppl.tex": table_ppl(Path(args.ppl_root)),
    }
    for name, text in tables.items():
        (out_dir / name).write_text(text, encoding="utf-8")

    figure_accuracy(datasets, fig_dir / "fig1-accuracy.pdf")
    figure_beta(beta_rows, fig_dir / "fig3-beta.pdf")
    figure_behavior(datasets, fig_dir / "fig2-behavior.pdf")

    print(f"表 {len(tables)} 份写入 {out_dir}")
    for row in beta_rows:
        print(f"  系数 {row['beta']:g}  正确 {row['correct']}  均长 {row['mean_gen_tokens']}  取自 {row['source']}")
    print(f"图 3 幅写入 {fig_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
