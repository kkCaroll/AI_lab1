"""报告表格取位与日志解析的单元测试，不依赖运行产物。"""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.export_tables import (
    BETA_READINGS,
    fmt4,
    fmt_ci,
    fmt_p,
    fmt_pp,
    parse_beta_logs,
)

LOG_LINE = (
    "2026-10-19T10:57:13.5+08:00 dlcxxtpu3ax53i0m-master-0 pytorch "
    "  开发集  准确率 {acc}  正确 {correct}/400  均长 {length} token  用时 397.0 秒\n"
)


def _write_log(directory: Path, beta: float, correct: int, acc: float, length: float) -> None:
    text = LOG_LINE.format(acc=f"{acc:.4f}", correct=correct, length=f"{length:.1f}")
    (directory / f"ai-lab1-P3-1-kl{beta:g}-成功.txt").write_text(text, encoding="utf-8")


def test_fmt4_truncates_to_four_places() -> None:
    assert fmt4(0.44351781652767247) == "0.4435"
    assert fmt4(0.6020) == "0.6020"


def test_fmt_pp_reports_points_with_sign() -> None:
    assert fmt_pp(-0.08946171341925702) == "-8.95"
    assert fmt_pp(0.12657922668688398) == "12.66"


def test_fmt_p_switches_at_threshold() -> None:
    assert fmt_p(2.384727840076348e-09) == "$<$0.0001"
    assert fmt_p(0.0166) == "0.0166"


def test_fmt_ci_uses_double_hyphen() -> None:
    assert fmt_ci(0.5752955652507104, 0.6280545778872377) == "0.5753--0.6281"


def test_parse_beta_logs_returns_sorted_readings(tmp_path: Path) -> None:
    lengths = {0.05: 106.0, 0.2: 110.1, 0.8: 153.3, 1.6: 180.7, 3.2: 191.7}
    for beta, correct in BETA_READINGS.items():
        _write_log(tmp_path, beta, correct, correct / 400, lengths[beta])
    rows = parse_beta_logs(tmp_path)
    assert [row["beta"] for row in rows] == [0.05, 0.2, 0.8, 1.6, 3.2]
    assert [row["correct"] for row in rows] == [196, 195, 243, 241, 233]


def test_parse_beta_logs_rejects_mismatch(tmp_path: Path) -> None:
    for beta, correct in BETA_READINGS.items():
        _write_log(tmp_path, beta, correct + 7, correct / 400, 100.0)
    with pytest.raises(ValueError):
        parse_beta_logs(tmp_path)


def test_parse_beta_logs_requires_all_five(tmp_path: Path) -> None:
    _write_log(tmp_path, 0.8, 243, 0.6075, 153.3)
    with pytest.raises(ValueError):
        parse_beta_logs(tmp_path)


def test_parse_beta_logs_prefers_canonical_run(tmp_path: Path) -> None:
    for beta, correct in BETA_READINGS.items():
        _write_log(tmp_path, beta, correct, correct / 400, 100.0)
    canonical = LOG_LINE.format(acc="0.6025", correct=241, length="180.7")
    (tmp_path / "ai-lab1-R3-4-kl1.6_成功.txt").write_text(canonical, encoding="utf-8")
    rows = parse_beta_logs(tmp_path)
    assert len(rows) == 5
    assert rows[3]["source"].startswith("ai-lab1-R3")
