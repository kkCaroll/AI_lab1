"""困惑度脚本的纯函数测试，不加载模型。"""

from __future__ import annotations

import math

import pytest

from scripts.measure_ppl import perplexity, plan_windows


def test_plan_windows_drops_remainder() -> None:
    assert plan_windows(2048, 1024) == (2, 2048)
    assert plan_windows(2050, 1024) == (2, 2048)
    assert plan_windows(1024, 1024) == (1, 1024)


def test_plan_windows_rejects_short_input() -> None:
    with pytest.raises(ValueError):
        plan_windows(1023, 1024)
    with pytest.raises(ValueError):
        plan_windows(2048, 1)


def test_perplexity_inverts_mean_nll() -> None:
    assert perplexity(1000 * math.log(10.0), 1000) == pytest.approx(10.0)


def test_perplexity_rejects_empty() -> None:
    with pytest.raises(ValueError):
        perplexity(1.0, 0)


def test_perplexity_guards_overflow() -> None:
    assert perplexity(1e9, 1) == float("inf")
