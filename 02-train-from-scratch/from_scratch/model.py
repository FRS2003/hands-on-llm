# -*- coding: utf-8 -*-
"""
from_scratch/model.py
=====================
不看官方实现、用原生 PyTorch 独立手写的现代 Decoder-only LLM 核心组件。
覆盖：RMSNorm、RoPE、SwiGLU FFN、GQA(MHA/MQA 的统一形式) 因果自注意力、Pre-Norm DecoderBlock。

统一维度记号
------------
B=batch, T=seq_len, D=hidden_size, H=n_heads, Hkv=n_kv_heads, Dh=head_dim = D//H
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------
# 1. RMSNorm：相比 LayerNorm 去掉「均值中心化」和偏置 beta，只用均方根做缩放
#    公式:  y = x / sqrt( mean(x^2) + eps ) * gamma
#    复杂度从 O(2n) 降到 O(n)，且实测训练更稳（Llama / Qwen 均采用）
# ----------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))  # 可学习缩放 gamma

    def forward(self, x: torch.Tensor) -> torch.Tensor:           # x: [..., D]
        norm = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()  # [...,1]
        return x * norm * self.weight


# ----------------------------------------------------------------------
# 2. RoPE 旋转位置编码：把位置 m 编码成对 q/k 两两维度的 2D 旋转
#    频率: theta_i = base^(-2i/D)；位置 m 上第 i 对维度旋转角 m*theta_i
#    关键性质: R_m 正交且 R_{m+n}=R_m R_n  => 内积 <R_m q, R_n k> 只依赖相对位置 (n-m)
# ----------------------------------------------------------------------
def precompute_rope(head_dim: int, max_seq_len: int, base: float = 10000.0,
                    device=None, dtype=torch.float32):
    # 每个通道对共享一个频率，长度 head_dim//2
    half = head_dim // 2
    idx = torch.arange(0, half, dtype=torch.float32, device=device)
    freqs = 1.0 / (base ** (idx / half))                         # [Dh/2]
    pos = torch.arange(max_seq_len, dtype=torch.float32, device=device)
    ang = torch.outer(pos, freqs)                                # [T, Dh/2]
    # 重复成 [T, Dh]：第 2i、2i+1 列相同，配合 rotate_half 的前后半拆分
    cos = torch.cat([ang.cos(), ang.cos()], dim=-1).to(dtype)    # [T, Dh]
    sin = torch.cat([ang.sin(), ang.sin()], dim=-1).to(dtype)
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
               position_ids: torch.Tensor = None) -> torch.Tensor:
    # x: [B, H, T, Dh]；cos/sin: [Tmax, Dh]
    B, H, T, Dh = x.shape
    if position_ids is None:
        c, s = cos[:T], sin[:T]                                  # [T, Dh]
    else:                                                        # 自定义位置
        c = cos[position_ids]                                    # [B,T,Dh]
        s = sin[position_ids]
        c = c[:, None, :, :]                                     # [B,1,T,Dh]
        s = s[:, None, :, :]
        x1, x2 = x[..., :Dh // 2], x[..., Dh // 2:]
        rot = torch.cat([-x2, x1], dim=-1)
        return x * c + rot * s
    c = c[None, None, :, :]                                      # [1,1,T,Dh]
    s = s[None, None, :, :]
    x1, x2 = x[..., :Dh // 2], x[..., Dh // 2:]
    rot = torch.cat([-x2, x1], dim=-1)                           # rotate_half
    return x * c + rot * s


# ----------------------------------------------------------------------
# 3. SwiGLU 前馈网络:  down( SiLU(gate(x)) * up(x) )
#    比标准 ReLU FFN 多一条门控分支；中间层维度按 Llama 取 2/3·D 并向上取整
# ----------------------------------------------------------------------
class SwiGLU(nn.Module):
    def __init__(self, hidden: int, multiple_of: int = 64, multiplier: float = 2 / 3):
        super().__init__()
        inner = int(2 * hidden * multiplier)
        inner = multiple_of * ((inner + multiple_of - 1) // multiple_of)
        self.gate = nn.Linear(hidden, inner, bias=False)
        self.up = nn.Linear(hidden, inner, bias=False)
        self.down = nn.Linear(inner, hidden, bias=False)
        self.inner = inner

    def forward(self, x):                                         # [B,T,D] -> [B,T,D]
        return self.down(F.silu(self.gate(x)) * self.up(x))


# ----------------------------------------------------------------------
# 4. GQA：query 有 H 个头，key/value 只有 Hkv 个头（Hkv<=H），
#    先把每个 KV 头复制 n_rep=H/Hkv 份再做标准 MHA，从而显著压缩 KV-Cache。
#    Hkv=H 即 MHA；Hkv=1 即 MQA。这里给出教学用的显式实现。
# ----------------------------------------------------------------------
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:       # [B,Hkv,T,Dh]
    if n_rep == 1:
        return x
    B, Hkv, T, Dh = x.shape
    return x[:, :, None, :, :].expand(B, Hkv, n_rep, T, Dh).reshape(B, Hkv * n_rep, T, Dh)


class GQACausalSelfAttention(nn.Module):
    def __init__(self, hidden: int = 768, n_heads: int = 8, n_kv_heads: int = 4):
        super().__init__()
        assert hidden % n_heads == 0
        self.H, self.Hkv = n_heads, n_kv_heads
        self.n_rep = n_heads // n_kv_heads
        self.Dh = hidden // n_heads
        self.wq = nn.Linear(hidden, n_heads * self.Dh, bias=False)
        self.wk = nn.Linear(hidden, n_kv_heads * self.Dh, bias=False)
        self.wv = nn.Linear(hidden, n_kv_heads * self.Dh, bias=False)
        self.wo = nn.Linear(n_heads * self.Dh, hidden, bias=False)

    def forward(self, x: torch.Tensor, cos=None, sin=None) -> torch.Tensor:
        B, T, D = x.shape
        q = self.wq(x).view(B, T, self.H, self.Dh).transpose(1, 2)    # [B,H,T,Dh]
        k = self.wk(x).view(B, T, self.Hkv, self.Dh).transpose(1, 2)  # [B,Hkv,T,Dh]
        v = self.wv(x).view(B, T, self.Hkv, self.Dh).transpose(1, 2)
        if cos is not None:                                           # 对 q,k 加 RoPE
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        k = repeat_kv(k, self.n_rep)                                  # [B,H,T,Dh]
        v = repeat_kv(v, self.n_rep)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.Dh)       # [B,H,T,T]
        causal = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(causal, float("-inf"))            # 屏蔽未来
        attn = torch.softmax(scores, dim=-1)
        out = attn @ v                                                # [B,H,T,Dh]
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.wo(out)                                          # [B,T,D]


# ----------------------------------------------------------------------
# 5. Pre-Norm DecoderBlock：先归一化再进子层 + 残差（现代 LLM 标配，比 Post-Norm 更易训练）
#    x = x + Attn(RMSNorm(x));  x = x + FFN(RMSNorm(x))
# ----------------------------------------------------------------------
class DecoderBlock(nn.Module):
    def __init__(self, hidden=768, n_heads=8, n_kv_heads=4):
        super().__init__()
        self.norm1 = RMSNorm(hidden)
        self.attn = GQACausalSelfAttention(hidden, n_heads, n_kv_heads)
        self.norm2 = RMSNorm(hidden)
        self.ffn = SwiGLU(hidden)

    def forward(self, x, cos=None, sin=None):
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.ffn(self.norm2(x))
        return x
