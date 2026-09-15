# -*- coding: utf-8 -*-
"""bench.py —— Triton 融合 RMSNorm vs 原生 PyTorch 多 kernel 写法的延迟对比（需 GPU）。
结论（RTX 3080 Ti, bf16）：访存受限的大张量上融合 kernel 明显更快，小张量受 launch 开销影响略慢。
"""
import torch, time
from kernel_rmsnorm import RMSNormTriton, rmsnorm_ref

dev = "cuda"
def bench(fn, iters=200, warm=50):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); t0 = time.time()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / iters * 1e3

if __name__ == "__main__":
    assert torch.cuda.is_available(), "bench 需要 GPU"
    for (M, D) in [(4096, 1024), (8192, 4096), (16384, 4096)]:
        x = torch.randn(M, D, device=dev, dtype=torch.bfloat16)
        w = torch.randn(D, device=dev, dtype=torch.bfloat16)
        mod = RMSNormTriton(D).to(dev, torch.bfloat16)
        with torch.no_grad(): mod.weight.copy_(w)
        with torch.no_grad():
            tt = bench(lambda: mod(x))
            tr = bench(lambda: rmsnorm_ref(x, w))
        print(f"M={M} D={D}: triton={tt:.4f}ms  torch={tr:.4f}ms  speedup={tr/tt:.2f}x")