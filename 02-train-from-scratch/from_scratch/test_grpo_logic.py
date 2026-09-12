# -*- coding: utf-8 -*-
"""
GRPO 核心数值逻辑自检（纯 tensor，CPU 秒级，不需要加载 63M 模型、不需要在线生成）。
逻辑与 MiniMind trainer/train_grpo.py / trainer_utils.py 严格对齐：
  - 规则奖励 calculate_rewards（长度 / thinking 标签 / 3-gram 重复惩罚）
  - 纯规则奖励版：学习型 reward model 的 get_score 恒为 0（不下载 internlm2-1_8b-reward）
  - 组内相对优势 advantages = (r - 组内均值) / (组内标准差 + 1e-4)
  - KL 正则 per_token_kl = exp(kl) - kl - 1
  - CISPO / GRPO 两种 loss 与可反传性
运行：python test_grpo_logic.py
"""
import re
import torch

torch.manual_seed(0)
PASS, FAIL = 0, 0


def check(name, cond):
    global PASS, FAIL
    if bool(cond):
        PASS += 1
        print(f"[PASS] {name}")
    else:
        FAIL += 1
        print(f"[FAIL] {name}")


# ---------- 1) 与官方一致的 3-gram 重复惩罚 ----------
def rep_penalty(text, n=3, cap=0.5):
    toks = re.findall(r"\w+|[^\w\s]", text.lower())
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return min(cap, (len(grams) - len(set(grams))) * cap * 2 / len(grams)) if grams else 0.0


def test_rep_penalty():
    repetitive = "the cat the cat the cat the cat the cat"
    diverse = "the quick brown fox jumps over lazy dogs happily"
    pr, pd = rep_penalty(repetitive), rep_penalty(diverse)
    check("重复文本惩罚 > 多样文本", pr > pd)
    check("惩罚值落在 [0, 0.5]", 0.0 <= pd <= pr <= 0.5)


# ---------- 2) 纯规则奖励（学习型 reward model 贡献恒为 0） ----------
def rule_reward(response, learned_score=0.0):
    """对齐 calculate_rewards；learned_score 即 reward_model.get_score，纯规则版恒 0。"""
    r = 0.0
    r += 0.5 if 20 <= len(response.strip()) <= 800 else -0.5
    if "</think>" in response:
        thinking, answer = response.split("</think>", 1)
        r += 1.0 if 20 <= len(thinking.strip()) <= 300 else -0.5
        r += 0.25 if response.count("</think>") == 1 else -0.25
    r -= rep_penalty(response)
    return r + learned_score  # 纯规则版 learned_score=0


def test_rule_reward():
    too_short = "ok"  # 长度 <20 -> -0.5
    good_len = "这是一个长度合适、内容正常的回答，用来拿到长度奖励分。"
    r_short = rule_reward(too_short)
    r_len = rule_reward(good_len)
    check("过短回答长度项为 -0.5（无重复惩罚时总分=-0.5）", abs(r_short - (-0.5)) < 1e-6)
    check("合适长度拿到 +0.5", r_len >= 0.5 - 1e-6)

    think = "我需要先分析问题再给出结论，步骤是先读题后计算" + "</think>" + good_len
    r_think = rule_reward(think)
    check("规范 thinking 拿到 +1.0 思考分与 +0.25 标签分", r_think >= 0.5 + 1.0 + 0.25 - 0.01)
    check("纯规则版学习型 reward 贡献为 0", rule_reward(good_len, learned_score=0.0) == r_len)


# ---------- 3) 组内相对优势（GRPO 的核心，替代 PPO 的 value 网络） ----------
def test_group_advantage():
    B, G = 2, 3
    rewards = torch.tensor([0.2, 0.8, 0.5, -0.3, 0.1, 0.4])  # 两组，每组 G=3
    grouped = rewards.view(B, G)
    mean = grouped.mean(dim=1).repeat_interleave(G)
    std = grouped.std(dim=1, unbiased=False).repeat_interleave(G)
    adv = (rewards - mean) / (std + 1e-4)
    adv_g = adv.view(B, G)
    check("每组优势均值约为0（组内中心化）", torch.allclose(adv_g.mean(dim=1), torch.zeros(B), atol=1e-5))
    check("组内高分优势>0、低分优势<0", adv[1] > 0 and adv[0] < 0)
    check("优势按组分别归一化（跨组不比较）", adv[1].item() != adv[5].item())


# ---------- 4) KL 正则 exp(kl)-kl-1：非负、kl=0 时为 0 ----------
def test_kl():
    kl = torch.randn(4096, dtype=torch.float32) * 0.5
    per_tok_kl = torch.exp(kl) - kl - 1.0
    check("KL 惩罚项恒非负", (per_tok_kl >= -1e-6).all())
    zero = torch.zeros(8)
    check("kl=0 时惩罚精确为 0（policy 未偏离 reference）",
          torch.allclose(torch.exp(zero) - zero - 1, zero, atol=1e-7))


# ---------- 5) CISPO loss 可反传；初始 ratio=1 ----------
def test_loss_and_backward():
    B, R, V = 4, 16, 100
    logits = torch.nn.Parameter(torch.randn(B, R, V) * 0.1)
    labels = torch.randint(0, V, (B, R))
    per_token_logps = torch.log_softmax(logits, dim=-1).gather(2, labels.unsqueeze(-1)).squeeze(-1)
    old_per_token_logps = per_token_logps.detach().clone()
    ref_per_token_logps = per_token_logps.detach().clone()
    advantages = torch.randn(B).unsqueeze(1).repeat(1, R)
    mask = torch.ones(B, R)
    beta, eps_high = 0.1, 5.0

    ratio = torch.exp(per_token_logps - old_per_token_logps)
    check("训练第一步 policy==old 时 ratio 恒为 1", torch.allclose(ratio, torch.ones_like(ratio), atol=1e-5))

    kl_div = ref_per_token_logps - per_token_logps
    per_tok_kl = torch.exp(kl_div) - kl_div - 1
    clamped = torch.clamp(ratio, max=eps_high).detach()
    cispo = -(clamped * advantages * per_token_logps - beta * per_tok_kl)
    loss = ((cispo * mask).sum(1) / mask.sum(1)).mean()
    loss.backward()
    check("CISPO loss 为有限标量", torch.isfinite(loss))
    check("反传后 policy 参数梯度存在且非全 0", logits.grad is not None and logits.grad.abs().sum() > 0)


if __name__ == "__main__":
    test_rep_penalty()
    test_rule_reward()
    test_group_advantage()
    test_kl()
    test_loss_and_backward()
    print("-" * 60)
    print(f"GRPO logic self-test: {PASS} passed, {FAIL} failed")
    raise SystemExit(1 if FAIL else 0)