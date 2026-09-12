# reproduced（可复现流程）

本目录记录在单卡上把 MiniMind「继续预训练 → SFT」端到端跑通的**确切命令与踩坑**，脚本均可直接复用。
模型/训练源码来自 [jingyaogong/minimind](https://github.com/jingyaogong/minimind)（MIT），此处仅放复现编排脚本，不含其源码。

## 0. 硬件与环境

- GPU：RTX 3080 Ti 12GB；系统 Ubuntu 22.04；Python 3.10；CUDA 12.4；PyTorch 2.6.0
- 关键 Python 包：`transformers==4.57.6 datasets==3.6.0 modelscope==1.37.0 trl==0.13.0 peft accelerate sentencepiece einops safetensors numpy==1.26.4`

### 踩坑：pip 拉大包零增长卡死 → curl + 离线安装

在该网络环境下，`pip install torch` 一旦下载数百 MB 的 wheel（torch 766MB、nvidia-cudnn 665MB…）就会**零增长卡死**，
换清华/阿里源都一样（问题在 pip 的下载长连接，不在镜像）；多线程下载器又被镜像 403。可靠解法：

```bash
# 1) 纯 python 小依赖直接在线装（小包不卡）
pip install filelock typing-extensions "sympy==1.13.1" networkx jinja2 fsspec mpmath markupsafe -i https://pypi.tuna.tsinghua.edu.cn/simple
# 2) torch 与 13 个 nvidia-*-cu12 编译大包，全部用 curl 单线程下到 wheels/（可断点续传、逐个字节校验）
#    torch 2.6.0 依赖清单见 wheels-list.txt，注意别漏 nvidia-cusparselt-cu12==0.6.2，且 sympy 必须精确 1.13.1
# 3) 离线安装 torch 全家桶
pip install --no-index --find-links wheels/ torch==2.6.0 numpy==1.26.4
# 4) 训练依赖在线装
pip install transformers==4.57.6 datasets==3.6.0 modelscope==1.37.0 trl==0.13.0 peft accelerate sentencepiece einops safetensors
# 验证：应输出 cuda True、bf16 True
python -c "import torch;print(torch.cuda.is_available(),torch.cuda.is_bf16_supported(),torch.cuda.get_device_name(0))"
```

## 1. 拉代码与数据

```bash
git clone --depth 1 https://github.com/jingyaogong/minimind.git MiniMind && cd MiniMind
# 数据集（ModelScope 国内直连，无需代理）：pretrain_t2t_mini.jsonl 1.27M 条 / sft_t2t_mini.jsonl 905k 条
python - <<'PY'
from modelscope import snapshot_download
snapshot_download('gongjy/minimind_dataset', repo_type='dataset',
                  local_dir='./dataset',
                  allow_file_pattern=['pretrain_t2t_mini.jsonl','sft_t2t_mini.jsonl'])
PY
```

## 2. 等间隔抽样子集

`python make_subsets.py`（脚本内可改目标条数）。等间隔抽样覆盖整个语料、避免只取头部造成分布偏斜。

## 3. 训练（必须在 `trainer/` 目录下运行，相对路径才正确）

```bash
bash train_run.sh small     # 60k pretrain / 20k SFT，约 18 分钟
bash train_run.sh medium    # 150k pretrain / 50k SFT，约 45 分钟
```

脚本会顺序执行 pretrain → 从 pretrain 权重热启动 SFT，并每 5 秒采样一次显存/利用率到 `out/gpu_*.csv`。
核心等价命令：

```bash
cd trainer
python -u train_pretrain.py --data_path ../dataset/pretrain_med.jsonl --epochs 2 --save_weight pretrain        --batch_size 32 --accumulation_steps 8 --learning_rate 5e-4 --dtype bfloat16 --log_interval 100
python -u train_full_sft.py --data_path ../dataset/sft_med.jsonl --epochs 2 --from_weight pretrain        --batch_size 16 --accumulation_steps 1 --learning_rate 1e-5 --dtype bfloat16 --log_interval 100
```


### LoRA 参数高效微调（在 full-SFT 基座上）

```bash
cd trainer
python -u train_lora.py --data_path ../dataset/sft_sub.jsonl --lora_name lora_demo \
       --from_weight full_sft --epochs 3 --batch_size 32 --learning_rate 1e-4 \
       --log_interval 50 --num_workers 4
# 推理：基座 + 适配器叠加加载
cd .. && printf '0\n' | python eval_llm.py --weight full_sft --lora_weight lora_demo --max_new_tokens 120
```

实测：只训 0.393M（占 0.61%）、适配器 0.78MB、显存峰值 4.6GB、20k 数据 3 轮仅 256 秒。

## 4. 推理与解析

```bash
# 在仓库根目录，选 0 走内置 8 题自动测试；权重默认读 ./out/full_sft_768.pth
printf '0
' | python eval_llm.py --weight full_sft --max_new_tokens 160 --temperature 0.7
# 把带 \r 刷新的训练日志解析成 loss CSV + 汇总指标
python parse_logs.py train_med.log out/gpu_med.csv experiments/medium
```

## 5. 产物清单

- `out/pretrain_768.pth`、`out/full_sft_768.pth`：训练权重（约 132MB/个，按 .gitignore 不上传）；
- `*_curve.csv`：逐步 loss/lr；`metrics.txt`：耗时、loss 首尾、显存峰值、利用率；
- 结果汇总见 [`../experiments/`](../experiments/)。
