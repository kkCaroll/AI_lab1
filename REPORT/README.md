# 报告编译说明

本目录为实验一报告源文件。正文五部分置于 `sections/`，表格与图分别由脚本生成至 `tables/` 与 `figures/`，后者不手工编辑。

内容先在 `draft/` 下以 Markdown 写定并修订，确认后再转排为 LaTeX。两份文件的对应关系与撰写口径见 `draft/00-说明.md`。

## 环境

TeX Live 2025，XeLaTeX，`fontset=windows` 使用本机宋体与黑体。所需宏包为 `ctex`、`booktabs`、`multirow`、`graphicx`、`amsmath`、`siunitx`、`caption`、`hyperref`、`gbt7714`。

## 编译

在 `REPORT/` 下执行

```bash
latexmk -xelatex main.tex
```

中间文件与成品统一写入 `build/`，由 `.latexmkrc` 设定。成品为 `build/main.pdf`，源文件目录只留源码与图表。清理中间文件

```bash
latexmk -c
```

只编一次的应急方式，适用于现场不便使用 latexmk 的情形

```bash
xelatex -output-directory=build -interaction=nonstopmode main.tex
xelatex -output-directory=build -interaction=nonstopmode main.tex
```

两次编译用于生成目录与交叉引用。参考文献首次新增条目时须另跑 `bibtex main`，用 latexmk 时自动完成。

## 表格与图

`exp1/scripts/export_tables.py` 生成 `tables/` 下的五份表与 `figures/` 下的三幅图，命令为

```bash
cd exp1
python -m scripts.export_tables
```

脚本读取 `exp1/runs/analysis/summary.json`、`exp1/runs/P4-ppl/*/ppl.json` 与 `运行输出日志/P3/` 的五份任务日志，只做取位与排版，不改数值。表一、表二与表八为人工记录，直接在正文中排版，数值取自 `RECORD/01-实验配置.md` 与 `RECORD/00-结果表.md`。

## 目录

| 路径 | 内容 |
| --- | --- |
| `main.tex` | 主文件，加载宏包并按序引入各节 |
| `draft/` | Markdown 草稿，定稿后转排为 `sections/` 下的 TeX |
| `sections/` | 摘要、实验原理、实验设置、实验结果、结果分析、实验总结、附录 |
| `build/` | 编译中间文件与 `main.pdf`，不入仓库 |
| `tables/` | 脚本生成的表格片段 |
| `figures/` | 脚本生成的矢量图 |
| `refs.bib` | 参考文献条目，按 GB/T 7714 数字制排版 |
| `验收操作.md` | 现场验收三项操作的命令、耗时与回退路径 |
| `问答要点.md` | 现场问答的四个方向与五个高概率难题 |
