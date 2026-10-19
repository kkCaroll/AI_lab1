"""M4 自研方法。LoRA 旁路加基座输出上的 KL 正则。

与 M2 的唯一差别是损失里多一项 β 乘 KL，参照分布为未经训练的基座。参考模型在装配时
深拷贝得到，不注册为子模块，因而不进参数统计、不进检查点、不进优化器。

散度只算在完成段上，取基座在前的方向，即用基座软标签做蒸馏。方向与分块的取舍见
PLAN/04-M4方案.md 第二节与第四节。
"""

from __future__ import annotations

import copy
from pathlib import Path

import torch
import torch.nn.functional as F

from src.utils import parameter_counts

METHOD = "custom"
CONFIG_SCHEMA: dict[str, tuple[object, type]] = {
    "r": (8, int),
    "alpha": (16, int),
    "dropout": (0.05, float),
    "target_modules": (["q_proj", "k_proj", "v_proj", "o_proj"], list),
    "kl_weight": (None, float),
    "kl_chunk": (128, int),
}

IGNORE_INDEX = -100
REFERENCE_ATTRIBUTE = "_reference_model"
KL_LOG_EVERY = 512

_kl_calls = 0


def resolve_weight(config):
    """取正则系数，未定即报错，不使未填的取值静默跑成一次无正则的训练。"""
    weight = config["kl_weight"]
    if weight is None:
        raise ValueError("custom.kl_weight 未填，须由配置或 --set 给出")
    return weight


def build(model, config):
    """深拷贝一份冻结基座作参照，再在基座上挂 LoRA 旁路。

    参考模型以 object.__setattr__ 挂到实例字典，绕开 nn.Module 的模块注册。注册会使
    它的参数进入统计与检查点，而它只在训练期使用。
    """
    from peft import LoraConfig, get_peft_model

    resolve_weight(config)

    reference = copy.deepcopy(model)
    reference.requires_grad_(False)
    reference.eval()

    settings = LoraConfig(
        r=config["r"],
        lora_alpha=config["alpha"],
        lora_dropout=config["dropout"],
        target_modules=list(config["target_modules"]),
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, settings)
    object.__setattr__(peft_model, REFERENCE_ATTRIBUTE, reference)
    return peft_model


def describe(model, config):
    """返回本方法的描述。参考模型未注册，故两项参数量与 M2 同值。"""
    return {
        "method": METHOD,
        "checkpoint_kind": "adapter",
        "config": dict(config),
        "targeted_module_names": sorted(set(getattr(model, "targeted_module_names", []))),
        **parameter_counts(model),
    }


def extra_loss(model, batch, outputs, config):
    """返回 β 乘完成段上的每 token 散度，策略与参照以基座在前的方向计算。"""
    global _kl_calls

    weight = config["kl_weight"]
    reference = getattr(model, REFERENCE_ATTRIBUTE)
    with torch.no_grad():
        reference_logits = reference(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            use_cache=False,
        ).logits

    total, count = _masked_kl(
        outputs.logits, reference_logits, batch["labels"], config["kl_chunk"]
    )
    if count == 0:
        return None
    per_token = total / count

    _kl_calls += 1
    if _kl_calls <= 2 or _kl_calls % KL_LOG_EVERY == 0:
        # 散度非负，浮点求和可略低于零，显示时截到零以免误读
        shown = max(0.0, float(per_token.detach()))
        print(f"  正则项 每 token 散度 {shown:.4f}  系数 {weight:g}")
    return weight * per_token


def _masked_kl(policy_logits, reference_logits, labels, chunk):
    """按完成段逐位求基座在前的散度，沿序列维分块以压住词表投影的峰值。

    第 i 位的输出预测第 i 加 1 位的标签，故两者同时错开一位。填充位与提示段的标签为
    -100，自然排除。分块是必需的，整段上转 float32 时两份张量合计可达数 GiB。
    """
    policy = policy_logits[:, :-1]
    reference = reference_logits[:, :-1]
    mask = labels[:, 1:] != IGNORE_INDEX

    total = policy.new_zeros((), dtype=torch.float32)
    count = 0
    for start in range(0, policy.shape[1], chunk):
        stop = start + chunk
        log_policy = F.log_softmax(policy[:, start:stop].float(), dim=-1)
        prob_reference = F.softmax(reference[:, start:stop].float(), dim=-1)
        divergence = F.kl_div(log_policy, prob_reference, reduction="none").sum(-1)
        block = mask[:, start:stop]
        total = total + (divergence * block).sum()
        count += int(block.sum())
    return total, count


def save(model, tokenizer, output_dir):
    """写出适配器权重与配置，返回所写文件的相对路径。参考模型不落盘。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    return sorted(str(item.relative_to(output_dir)) for item in output_dir.rglob("*") if item.is_file())
