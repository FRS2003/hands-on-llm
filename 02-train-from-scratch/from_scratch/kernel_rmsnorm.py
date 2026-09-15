# -*- coding: utf-8 -*-
"""
from_scratch/kernel_rmsnorm.py
==============================
用 OpenAI Triton 手写 RMSNorm 的融合 GPU Kernel，对标 model.RMSNorm（纯 PyTorch 版）。

RMSNorm 公式（与 model.py 完全一致）：
    rms = sqrt( mean(x^2) + eps )          # 每个 token 沿最后一维 D 算一个标量
    y   = (x / rms) * gamma                # gamma 是逐通道可学习缩放

为什么值得写成 Kernel
---------------------
PyTorch 原生写法 x.pow(2).mean().rsqrt() 会启动多个 kernel、中间张量反复读写 HBM；
Triton 让一个 program 负责一行（一个 token 的 D 个分量），在片上完成
「归约求均方根 → 归一化 → 乘 gamma」的融合，减少 HBM 往返（这正是 FlashAttention
「分块 + 融合」思想在归一化算子上的同款应用）。

实现要点
--------
- forward：两遍扫描。第 1 遍分块累加 x^2 求 1/rms（fp32 累加保精度），第 2 遍写 y；
  D 超过 BLOCK_D 时循环分块，故支持任意 hidden 维度。
- backward：RMSNorm 没有「去均值」，反向比 LayerNorm 简单。记 x_hat = x/rms：
      dgamma = sum_rows(dy * x_hat)
      dx     = (dy*gamma - x_hat * mean(dy*gamma * x_hat)) * (1/rms)
  dgamma 用 atomic_add 跨行归约；同样两遍扫描。
- 封装成 torch.autograd.Function，可直接插进训练图反向传播。

维度记号：M = 除最后一维外所有维度展平后的行数（B*T），D = hidden_size。
"""
import torch
import triton
import triton.language as tl


# ----------------------------------------------------------------------
# 前向融合 Kernel：每个 program 处理一行（长度 D）
# ----------------------------------------------------------------------
@triton.jit
def _rmsnorm_fwd_kernel(
    X, W, Y, RSTD,
    stride_row,            # 相邻两行在内存中的间隔（= D，连续布局）
    D: tl.constexpr,       # hidden 维度（编译期常量，便于循环自动展开）
    BLOCK_D: tl.constexpr, # 单次处理的列数 = next_power_of_2(D)
    EPS: tl.constexpr,
):
    row = tl.program_id(0)
    X += row * stride_row
    Y += row * stride_row

    # —— 第 1 遍：分块累加 sum(x^2)，全程 fp32 —— #
    ss = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for off in range(0, D, BLOCK_D):
        cols = off + tl.arange(0, BLOCK_D)
        mask = cols < D
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        ss += x * x
    mean_sq = tl.sum(ss, axis=0) / D
    rstd = 1.0 / tl.sqrt(mean_sq + EPS)      # 1/rms，留给反向复用
    tl.store(RSTD + row, rstd)

    # —— 第 2 遍：y = (x * rstd) * gamma，写回原 dtype —— #
    for off in range(0, D, BLOCK_D):
        cols = off + tl.arange(0, BLOCK_D)
        mask = cols < D
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
        y = x * rstd * w
        tl.store(Y + cols, y.to(Y.dtype.element_ty), mask=mask)


# ----------------------------------------------------------------------
# 反向 Kernel：逐行算 dx，并用 atomic_add 跨行累加 dgamma
# ----------------------------------------------------------------------
@triton.jit
def _rmsnorm_bwd_kernel(
    X, W, DY, DX, DW, RSTD,
    stride_row,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    X += row * stride_row
    DY += row * stride_row
    DX += row * stride_row
    rstd = tl.load(RSTD + row)            # 这一行的 1/rms

    # —— 第 1 遍：累加 c = mean(dx_hat * x_hat)，同时 atomic 累加 dgamma —— #
    cc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for off in range(0, D, BLOCK_D):
        cols = off + tl.arange(0, BLOCK_D)
        mask = cols < D
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(DY + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
        x_hat = x * rstd                  # 归一化后、乘 gamma 前
        dx_hat = dy * w
        cc += dx_hat * x_hat
        tl.atomic_add(DW + cols, (dy * x_hat).to(tl.float32), mask=mask)
    c = tl.sum(cc, axis=0) / D

    # —— 第 2 遍：dx = (dx_hat - x_hat*c) * rstd —— #
    for off in range(0, D, BLOCK_D):
        cols = off + tl.arange(0, BLOCK_D)
        mask = cols < D
        x = tl.load(X + cols, mask=mask, other=0.0).to(tl.float32)
        dy = tl.load(DY + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
        x_hat = x * rstd
        dx_hat = dy * w
        dx = (dx_hat - x_hat * c) * rstd
        tl.store(DX + cols, dx.to(DX.dtype.element_ty), mask=mask)


# ----------------------------------------------------------------------
# 自动微分封装：前向走 Triton，反向也走 Triton
# ----------------------------------------------------------------------
class _RMSNormTritonFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6):
        assert x.is_cuda and weight.is_cuda, "Triton kernel 需要 CUDA 张量"
        assert x.shape[-1] == weight.shape[0]
        D = x.shape[-1]
        x_c = x.contiguous()
        M = x_c.numel() // D
        x2 = x_c.reshape(M, D)
        y = torch.empty_like(x2)
        rstd = torch.empty(M, dtype=torch.float32, device=x.device)
        BLOCK = triton.next_power_of_2(D)
        grid = (M,)
        _rmsnorm_fwd_kernel[grid](
            x2, weight, y, rstd,
            x2.stride(0), D=D, BLOCK_D=BLOCK, EPS=eps,
            num_warps=4,
        )
        ctx.save_for_backward(x2, weight, rstd)
        ctx.D = D
        ctx.stride = x2.stride(0)
        ctx.block = BLOCK
        return y.reshape_as(x)

    @staticmethod
    def backward(ctx, dy: torch.Tensor):
        x2, weight, rstd = ctx.saved_tensors
        D, stride, BLOCK = ctx.D, ctx.stride, ctx.block
        M = x2.shape[0]
        dy2 = dy.contiguous().reshape(M, D)
        dx = torch.empty_like(x2)
        dw = torch.zeros(D, dtype=torch.float32, device=x2.device)
        _rmsnorm_bwd_kernel[(M,)](
            x2, weight, dy2, dx, dw, rstd,
            stride, D=D, BLOCK_D=BLOCK, num_warps=4,
        )
        return dx.reshape(dy.shape), dw.to(weight.dtype), None


class RMSNormTriton(torch.nn.Module):
    """与 model.RMSNorm 同接口的 Triton 实现版，可直接替换进网络。"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))   # gamma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _RMSNormTritonFn.apply(x, self.weight, self.eps)


# 纯 PyTorch 参考实现（测试里当 ground truth）
def rmsnorm_ref(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6):
    norm = x.pow(2).mean(dim=-1, keepdim=True).add(eps).rsqrt()
    return x * norm * weight