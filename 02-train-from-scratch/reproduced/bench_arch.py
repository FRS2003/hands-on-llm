import torch, time, csv, os, warnings
warnings.filterwarnings("ignore")
import torch.nn.functional as F
from model.model_minimind import MiniMindConfig, MiniMindForCausalLM

torch.backends.cuda.matmul.allow_tf32 = True
VOCAB, SEQ, NLAYER, HD, QH = 6400, 512, 8, 96, 8
ARCHS = [("MHA", 8), ("GQA", 4), ("MQA", 1)]
DTYPES = [("fp32", torch.float32), ("bf16", torch.bfloat16), ("fp16", torch.float16)]
BATCHES = [1, 4, 8]
os.makedirs("experiments/ablation", exist_ok=True)

def bench(kv, dtype, bs, iters=5, warm=2):
    cfg = MiniMindConfig(hidden_size=768, num_hidden_layers=8, use_moe=False,
                         num_attention_heads=QH, num_key_value_heads=kv)
    m = MiniMindForCausalLM(cfg).to(dtype).cuda().train()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    npar = sum(p.numel() for p in m.parameters())
    def step():
        X = torch.randint(0, VOCAB, (bs, SEQ), device="cuda")
        out = m(X); lg = out.logits if hasattr(out, "logits") else out
        loss = F.cross_entropy(lg.float().reshape(-1, VOCAB), X.reshape(-1))
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
    for _ in range(warm): step()
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    for _ in range(iters): step()
    torch.cuda.synchronize()
    dt = (time.time()-t0)/iters
    peak = torch.cuda.max_memory_allocated()/1e6
    toks = bs*SEQ/dt
    finite = True
    del m, opt; torch.cuda.empty_cache()
    return npar, dt*1000, toks, peak

rows = []
print(f"{'arch':4s}{'dtype':6s}{'bs':>3s}{'params(M)':>10s}{'step(ms)':>10s}{'tok/s':>9s}{'peak(MB)':>10s}")
for an, kv in ARCHS:
    for dn, dt in DTYPES:
        for bs in BATCHES:
            npar, ms, toks, peak = bench(kv, dt, bs)
            rows.append([an, kv, dn, bs, SEQ, round(npar/1e6,3), round(ms,1), round(toks,0), round(peak,1)])
            print(f"{an:4s}{dn:6s}{bs:3d}{npar/1e6:10.3f}{ms:10.1f}{toks:9.0f}{peak:10.1f}", flush=True)

with open("experiments/ablation/arch_ablation.csv","w",newline="",encoding="utf-8") as f:
    w=csv.writer(f); w.writerow(["arch","kv_heads","dtype","batch","seq","params_M","step_ms","tokens_s","peak_MB"]); w.writerows(rows)

# inference KV cache (fp16/bf16 = 2 bytes), whole-prefix cache
L=["", "Inference KV-cache size = 2(K,V)*n_layers*kv_heads*head_dim*seq*2bytes",
   f"{'arch':4s}" + "".join(f"{('seq='+str(s)):>12s}" for s in [512,1024,2048,4096])]
kvrows=[]
for an,kv in ARCHS:
    vals=[2*NLAYER*kv*HD*s*2/1e6 for s in [512,1024,2048,4096]]
    kvrows.append([an]+[round(v,2) for v in vals])
    L.append(f"{an:4s}" + "".join(f"{v:12.2f}" for v in vals))
with open("experiments/ablation/kv_cache.csv","w",newline="",encoding="utf-8") as f:
    w=csv.writer(f); w.writerow(["arch","seq512_MB","seq1024_MB","seq2048_MB","seq4096_MB"]); w.writerows(kvrows)
# param-only footprint by dtype for GQA
L+=["", "GQA model parameter footprint (MB)"]
for dn,dt in DTYPES:
    cfg=MiniMindConfig(hidden_size=768,num_hidden_layers=8,use_moe=False,num_attention_heads=8,num_key_value_heads=4)
    m=MiniMindForCausalLM(cfg).to(dt).cuda()
    L.append(f"{dn:6s}{torch.cuda.memory_allocated()/1e6:10.2f}")
    del m; torch.cuda.empty_cache()
rep="\n".join(L); print(rep)
open("experiments/ablation/arch_ablation_metrics.txt","w",encoding="utf-8").write(rep)
print("ABLATION_DONE")
