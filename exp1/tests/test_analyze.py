"""统计脚本的纯函数测试，不加载模型。"""

from __future__ import annotations

import math

import pytest

from scripts.analyze_results import (
    format_p,
    holm,
    mcnemar_exact,
    paired_diff_ci,
    wilson,
)


def test_wilson_at_bounds() -> None:
    low, _ = wilson(0, 100)
    _, high = wilson(100, 100)
    assert low == 0.0
    assert high == 1.0


def test_wilson_centered_on_proportion() -> None:
    low, high = wilson(794, 1319)
    assert low < 794 / 1319 < high
    half_width = (high - low) / 2
    expected = 1.959963984540054 * math.sqrt((794 / 1319) * (1 - 794 / 1319) / 1319)
    assert half_width == pytest.approx(expected, abs=0.001)


def test_mcnemar_exact_known_values() -> None:
    assert mcnemar_exact(10, 0) == pytest.approx(2 / 1024)
    assert mcnemar_exact(9, 1) == pytest.approx(22 / 1024)
    assert mcnemar_exact(5, 5) == pytest.approx(1.0)
    assert mcnemar_exact(0, 0) == pytest.approx(1.0)


def test_mcnemar_symmetric() -> None:
    assert mcnemar_exact(30, 7) == pytest.approx(mcnemar_exact(7, 30))


def test_paired_diff_ci() -> None:
    diff, low, high = paired_diff_ci(20, 5, 100)
    standard_error = math.sqrt(25 - 225 / 100) / 100
    assert diff == pytest.approx(0.15)
    assert low == pytest.approx(0.15 - 1.959963984540054 * standard_error)
    assert high == pytest.approx(0.15 + 1.959963984540054 * standard_error)


def test_paired_diff_ci_no_discordant() -> None:
    assert paired_diff_ci(0, 0, 100) == (0.0, 0.0, 0.0)


def test_holm_monotone() -> None:
    assert holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_holm_caps_at_one() -> None:
    assert holm([0.9, 0.9]) == pytest.approx([1.0, 1.0])


def test_format_p() -> None:
    assert format_p(0.0) == "<0.0001"
    assert format_p(0.02) == "0.0200"
