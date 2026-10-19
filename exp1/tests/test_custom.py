"""M4 正则项的单元测试。

只测纯张量部分，不加载模型。散度的方向、掩码与分块三处是训练能否按设计进行的直接
依据，三处都不报错而结果错误，故各设一条对照。
"""

from __future__ import annotations

import torch
from torch.distributions import Categorical

from src.methods.custom import IGNORE_INDEX, _masked_kl


def _independent_kl(policy_logits, reference_logits):
    """以 torch.distributions 独立算一遍基座在前的散度，逐位置求和不取均值。"""
    reference = Categorical(logits=reference_logits.float())
    policy = Categorical(logits=policy_logits.float())
    return torch.distributions.kl_divergence(reference, policy).sum()


def test_identical_distributions_give_zero():
    logits = torch.randn(2, 5, 7)
    labels = torch.arange(5).repeat(2, 1)
    total, count = _masked_kl(logits, logits.clone(), labels, 2)
    assert count == 8
    assert abs(float(total)) < 1e-5


def test_direction_is_reference_first():
    policy = torch.randn(1, 4, 6)
    reference = torch.randn(1, 4, 6)
    labels = torch.arange(4).unsqueeze(0)
    total, count = _masked_kl(policy, reference, labels, 64)
    expected = _independent_kl(policy[:, :-1], reference[:, :-1])
    assert count == 3
    assert torch.allclose(total, expected, atol=1e-5)


def test_ignored_positions_do_not_contribute():
    policy = torch.randn(1, 4, 6)
    reference = torch.randn(1, 4, 6)
    labels = torch.tensor([[9, 8, IGNORE_INDEX, 7]])
    total, count = _masked_kl(policy, reference, labels, 64)

    disturbed = policy.clone()
    disturbed[0, 1] = 100.0
    total_disturbed, count_disturbed = _masked_kl(disturbed, reference, labels, 64)

    assert count == 2
    assert count_disturbed == 2
    assert torch.allclose(total, total_disturbed, atol=1e-5)


def test_shift_uses_next_position_label():
    policy = torch.randn(1, 5, 6)
    reference = torch.randn(1, 5, 6)
    labels = torch.tensor([[5, IGNORE_INDEX, 6, IGNORE_INDEX, 7]])
    _, count = _masked_kl(policy, reference, labels, 64)
    assert count == 2


def test_chunk_width_does_not_change_result():
    policy = torch.randn(2, 9, 8)
    reference = torch.randn(2, 9, 8)
    labels = torch.randint(0, 8, (2, 9))
    wide, wide_count = _masked_kl(policy, reference, labels, 64)
    narrow, narrow_count = _masked_kl(policy, reference, labels, 1)
    assert wide_count == narrow_count
    assert torch.allclose(wide, narrow, atol=1e-4)
