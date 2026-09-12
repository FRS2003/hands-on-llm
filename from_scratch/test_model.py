# -*- coding: utf-8 -*-
"""from_scratch/test_model.py —— 不依赖 GPU，纯 CPU 用数值性质验证手写组件正确性。
运行: python test_model.py
"""
import torch
from model import (RMSNorm, precompute_rope, apply_rope, SwiGLU,
                   repeat_kv, GQACausalSelfAttention, DecoderBlock)

torch.manual_seed(0)
PASS = 0
def ok(name, cond):
    assert cond, f"FAIL: {name}"
    global PASS; PASS += 1; print(f"[PASS] {name}")

# 1) RMSNorm：gamma=1 时输出每行 RMS≈1，且与手写公式一致
def test_rmsnorm():
    norm = RMSNorm(16); norm.eval()
    x = torch.randn(4, 10, 16)
    y = norm(x)
    rms = y.pow(2).mean(-1).sqrt()
    ok("RMSNorm 归一化后 RMS≈1", torch.allclose(rms, torch.ones_like(rms), atol=1e-5))
    ref = x * (x.pow(2).mean(-1, keepdim=True) + 1e-6).rsqrt()
    ok("RMSNorm 与手写公式一致", torch.allclose(y, ref, atol=1e-6))

# 2) RoPE：内积只依赖相对位置；旋转保持模长
def test_rope():
    Dh, M = 8, 64
    cos, sin = precompute_rope(Dh, M)
    q = torch.randn(1, 1, 1, Dh); k = torch.randn(1, 1, 1, Dh)
    def at(m):
        return apply_rope(q, cos, sin, torch.tensor([[m]]))[0, 0, 0]
    def dot(m, n):
        qm = at(m)
        kn = apply_rope(k, cos, sin, torch.tensor([[n]]))[0, 0, 0]
        return float((qm * kn).sum())
    a, b = dot(2, 5), dot(10, 13)       # 相对距离都是 3
    ok("RoPE 内积只依赖相对位置 (2,5)≈(10,13)", abs(a - b) < 1e-4)
    ok("RoPE 保持向量模长", abs(float(at(7).norm()) - float(q.norm())) < 1e-4)

# 3) repeat_kv 形状；GQA 因果性：改未来 token 不影响过去位置的输出
def test_gqa_causal():
    x = torch.randn(2, 4, 4, 6)
    z = repeat_kv(x, 3)
    ok("repeat_kv 头数复制正确", z.shape == (2, 12, 4, 6))
    attn = GQACausalSelfAttention(hidden=32, n_heads=4, n_kv_heads=2).eval()
    cos, sin = precompute_rope(8, 16)
    x1 = torch.randn(1, 5, 32)
    with torch.no_grad():
        o1 = attn(x1, cos, sin)
        x2 = x1.clone(); x2[:, 3:, :] += 5.0      # 只改动位置 3、4（相对 0/1 是未来）
        o2 = attn(x2, cos, sin)
    ok("GQA 因果遮蔽：未来改动不影响位置0/1", torch.allclose(o1[:, :2], o2[:, :2], atol=1e-5))
    changed = (o1[:, 2] - o2[:, 2]).abs().max()   # 位置2能看到位置3吗？不能（3是未来）
    ok("GQA 位置2同样看不到位置3", float(changed) < 1e-5)
    ok("GQA 输出形状正确", o1.shape == x1.shape)

# 4) SwiGLU：形状正确；up 权重清零后输出为 0（门控乘法生效）
def test_swiglu():
    ffn = SwiGLU(32); ffn.eval()
    x = torch.randn(2, 5, 32)
    ok("SwiGLU 形状保持", ffn(x).shape == x.shape)
    with torch.no_grad():
        ffn.up.weight.zero_(); o = ffn(x)
    ok("SwiGLU 门控分支生效(up=0=>out=0)", torch.allclose(o, torch.zeros_like(o), atol=1e-6))

# 5) DecoderBlock：形状保持且梯度可回传到所有子层
def test_block():
    blk = DecoderBlock(hidden=32, n_heads=4, n_kv_heads=2)
    cos, sin = precompute_rope(8, 16)
    x = torch.randn(2, 6, 32, requires_grad=True)
    y = blk(x, cos, sin)
    ok("DecoderBlock 形状保持", y.shape == x.shape)
    y.sum().backward()
    ok("DecoderBlock 梯度可回传", x.grad is not None and torch.isfinite(x.grad).all())

if __name__ == "__main__":
    test_rmsnorm(); test_rope(); test_gqa_causal(); test_swiglu(); test_block()
    print(f"\nALL {PASS} CHECKS PASSED")
