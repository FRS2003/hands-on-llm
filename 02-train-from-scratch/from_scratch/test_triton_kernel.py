# -*- coding: utf-8 -*-
"""
from_scratch/test_triton_kernel.py
==================================
Triton RMSNorm Kernel 的数值正确性自检（需要 NVIDIA GPU + triton）。
- 无 GPU / 未装 triton 的环境（如 GitHub Actions CPU runner）自动 SKIP 并以 0 退出，不拖红 CI；
- 有 GPU 时严格对齐前向输出与反向梯度（对拍纯 PyTorch 参考实现）。
运行: python test_triton_kernel.py
"""
import sys

try:
    import triton  # noqa: F401
except Exception:
    print("SKIP: 未安装 triton（CPU 环境），跳过 Triton kernel 自检")
    sys.exit(0)
import torch

try:
    from kernel_rmsnorm import RMSNormTriton, rmsnorm_ref
except Exception:
    from .kernel_rmsnorm import RMSNormTriton, rmsnorm_ref  # 包内导入兜底

if not torch.cuda.is_available():
    print("SKIP: 无可用 CUDA GPU，跳过 Triton kernel 自检")
    sys.exit(0)

dev = "cuda"
PASS, FAIL = 0, 0
def check(name, cond):
    global PASS, FAIL
    if bool(cond):
        PASS += 1; print(f"[PASS] {name}")
    else:
        FAIL += 1; print(f"[FAIL] {name}")


def test_forward(dtype, atol):
    torch.manual_seed(0)
    D = 1024
    x = torch.randn(8, 16, D, device=dev, dtype=dtype)
    w = torch.randn(D, device=dev, dtype=dtype)
    # 随机 gamma：验证「归一化 + 逐通道缩放」整体与参考一致
    mod = RMSNormTriton(D).to(dev, dtype)
    with torch.no_grad():
        mod.weight.copy_(w)
    y_tri = mod(x)
    y_ref = rmsnorm_ref(x, w)
    check(f"前向对齐[{dtype}] Triton≈PyTorch", torch.allclose(y_tri.float(), y_ref.float(), atol=atol, rtol=atol))
    # gamma=全1 时，输出 RMS 才应为 1（归一化本身的性质）
    mod1 = RMSNormTriton(D).to(dev, dtype)   # 默认 weight=ones
    y1 = mod1(x).float()
    rms = y1.pow(2).mean(-1).sqrt()
    check(f"gamma=1 时归一化 RMS≈1[{dtype}]", torch.allclose(rms, torch.ones_like(rms), atol=atol * 5))


def test_backward(dtype, atol):
    torch.manual_seed(1)
    D = 768
    x = torch.randn(4, 10, D, device=dev, dtype=dtype, requires_grad=True)
    w = torch.randn(D, device=dev, dtype=dtype, requires_grad=True)
    xt = x.detach().clone().requires_grad_(True)
    wt = w.detach().clone()
    mod = RMSNormTriton(D).to(dev, dtype)
    with torch.no_grad():
        mod.weight.copy_(wt)
    mod(xt).float().pow(2).mean().backward()           # Triton 路径
    rmsnorm_ref(x, w).float().pow(2).mean().backward() # PyTorch 参考路径
    check(f"反向 dx 对齐[{dtype}]", torch.allclose(xt.grad.float(), x.grad.float(), atol=atol*5, rtol=atol*5))
    check(f"反向 dgamma 对齐[{dtype}]", torch.allclose(mod.weight.grad.float(), w.grad.float(), atol=atol*5, rtol=atol*5))


def test_nonpower2():
    # 非 2 次幂维度，验证分块循环 + mask 正确性
    torch.manual_seed(2)
    for D in (1000, 2049):
        x = torch.randn(2, 5, D, device=dev, dtype=torch.float32)
        w = torch.randn(D, device=dev, dtype=torch.float32)
        mod = RMSNormTriton(D).to(dev)
        with torch.no_grad():
            mod.weight.copy_(w)
        check(f"非2次幂 D={D} 前向对齐", torch.allclose(mod(x), rmsnorm_ref(x, w), atol=1e-4))


if __name__ == "__main__":
    print("device:", torch.cuda.get_device_name(0))
    test_forward(torch.float32, 1e-5)
    test_forward(torch.bfloat16, 1e-2)
    test_backward(torch.float32, 1e-4)
    test_backward(torch.bfloat16, 3e-2)
    test_nonpower2()
    print("-" * 60)
    print(f"Triton RMSNorm self-test: {PASS} passed, {FAIL} failed")
    raise SystemExit(1 if FAIL else 0)