# PEFT 三种微调方式对照实验（Full / LoRA / QLoRA 实测）

本目录是 [`../peft_compare.csv`](../peft_compare.csv) 中 full / lora / qlora 三行数据的**完整可复现实验包**：配置、启动脚本、解析脚本、依赖版本、原始训练日志、逐秒显存采样、结构化结果一应俱全，做到表中每一个数字都能回溯到日志。

> Freeze / Adapter / Prefix-Tuning / P-Tuning / DoRA 等其余方法本次未实测，对应行在 csv 中如实留空，可沿用本目录的脚本与口径自行补测。

## 1. 实验要回答的问题

同样一个模型、同样一份数据，只改变微调方式，对比 **Full（全参）/ LoRA / QLoRA（4bit 量化 + LoRA）** 在四个工程维度上的差异：

1. 需要训练多少参数；
2. 训练时显存峰值是多少；
3. 训练要花多长时间；
4. 产出的权重文件有多大、效果（loss）差多少。

## 2. 控制变量（保证三组可比）

| 维度 | 统一设置 |
| --- | --- |
| 基座模型 | Qwen2.5-0.5B（494,032,768 参数） |
| 训练 / 评测数据 | 中文指令 3000 条 / 300 条（固定 seed=42 切分，见 data/） |
| 全局 batch size | 16（micro-batch 2 × 梯度累积 8，三组一致） |
| 训练轮数 | 1 epoch，优化步均为 188 步 |
| 最大长度 / 精度 | cutoff_len=512 / bf16 / 梯度检查点 |
| 学习率调度 | cosine，warmup 5%，seed=42 |
| LoRA 超参 | rank=8、alpha=16、dropout=0.05、target=all（LoRA 与 QLoRA 完全相同） |
| 硬件 / 框架 | 单卡 NVIDIA RTX 3080 Ti 12GB；LLaMA-Factory 0.9.3 + PyTorch 2.6 + CUDA 12.4 |

> 唯一变量是微调方式本身。全局 batch 固定为 16 是关键：micro-batch 只影响单次前向的显存，梯度累积保证三组每个优化步看到的样本数相同，梯度在数学上等价。

## 3. 目录结构

```
peft_lab/
├── yamls/                     # 三组实际跑通的最终配置（configs/ 里是带注释的教学模板）
│   ├── peft_full.yaml         # 全参微调
│   ├── peft_lora.yaml         # LoRA
│   └── peft_qlora.yaml        # QLoRA = bitsandbytes 4bit + LoRA
├── data/                      # 固定切分的数据与 LLaMA-Factory 注册文件
├── run_one.sh                 # 单组训练封装：训练同时用 nvidia-smi 每秒采样显存
├── run_all.sh                 # 依次跑三组并调用解析脚本
├── parse_results.py           # 从日志正则提取指标，汇总成 peft_compare_result.csv
├── prepare_data.py            # 从原始数据集可复现地切出 3000/300
├── requirements-lock.txt      # 隔离环境的精确依赖版本
├── peft_compare_result.csv    # 脚本自动解析出的汇总结果
├── logs/                      # 原始训练日志 + 逐秒显存采样（数字的证据来源）
└── results/<method>/          # 每组的 all/train/eval_results.json、trainer_state、adapter 配置
```

## 4. 复现步骤

```bash
# 1) 准备隔离环境（避免与其他项目的 transformers 版本冲突），按 requirements-lock.txt 安装
# 2) 用 ModelScope/HuggingFace 下载 Qwen2.5-0.5B，修改 yamls 里的 model_name_or_path
# 3) 准备数据（或直接使用 data/ 下已固定切分的 json）
python prepare_data.py
# 4) 逐组训练（也可直接 bash run_all.sh 一次跑完）
bash run_one.sh full
bash run_one.sh lora
bash run_one.sh qlora
# 5) 解析日志，生成汇总 csv
python parse_results.py
```

离线环境需在训练前导出，避免进程卡在联网校验：

```bash
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # 减少显存碎片
```

## 5. 实测结果

| 方法 | 可训参数 | 占比 | 显存峰值 | 纯训练时长 | 训练吞吐 | train_loss | eval_loss | 部署权重 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Full 全参 | 494,032,768 | 100% | 11081 MB | 4.53 min（271.8s） | 11.04 条/s | 1.806 | 1.826 | 1885 MB（fp32） |
| LoRA | 4,399,104 | 0.88% | 3699 MB | 7.61 min（456.9s） | 6.57 条/s | 1.840 | 2.007 | 16.8 MB |
| QLoRA | 4,399,104 | 0.88% | 3199 MB | 9.94 min（596.4s） | 5.03 条/s | 1.916 | 2.083 | 16.8 MB |

补充：LoRA 与 QLoRA 的前向浮点运算量（FLOPs）完全相同（均为 1.486e15），因为二者可训练结构一致，差别只在基座用 bf16 还是 4bit 存储；Full 为 1.468e15。

## 6. 结果怎么读

**（1）可训练参数：LoRA 只动 0.88%。** 498.4M 总参数里只有 439.9 万是 LoRA 注入的低秩矩阵（总参数 = 494.0M 基座 + 4.4M adapter），其余 99% 以上冻结。

**（2）显存：Full(11.1G) > LoRA(3.7G) > QLoRA(3.2G)，阶梯清晰。**
- Full 要为每个参数保存梯度 + Adam 一阶/二阶动量 + fp32 主权重，优化器状态是显存大头；
- LoRA 只对 adapter 维护优化器状态，显存降到约 1/3；
- QLoRA 进一步把基座用 NF4 4bit 压缩，显存最低。这正是消费级单卡能微调较大模型的关键。

**（3）产物体积：16.8 MB 对 1885 MB，相差约 112 倍。** LoRA/QLoRA 只产出一个小 adapter（两者结构相同故同为 16.8MB），基座共享，便于为不同任务各存一个小文件、热插拔切换；Full 直接改写整份权重。（Full 训练用 bf16 混合精度，框架默认按 fp32 落盘故为 1885MB，转成 bf16 存储可减半至约 943MB。）

**（4）效果：Full 略优、LoRA 次之、QLoRA 因量化噪声略高，但差距不大。** eval_loss 1.826 / 2.007 / 2.083 的排序与 train_loss 一致，符合“可训练容量越大、拟合越充分”的预期。

**（5）训练时长反直觉：Full 反而最快。** 三组优化步都是 188，但 Full(4.53min) < LoRA(7.61min) < QLoRA(9.94min)。原因有三：
- 为控制变量，三组 micro-batch 都被压到 2（Full 在 micro-batch=4 时第一步反向就 OOM），LoRA 没能用大 batch 发挥吞吐优势；
- 开启梯度检查点后，LoRA 的冻结层在反向时仍需重算前向，adapter 小算子串行调度有额外开销；
- QLoRA 每个前向都要把 4bit 权重反量化回 bf16，多一层计算，因此最慢。

> 工程上的真实取舍是：LoRA/QLoRA 用时间和极小的效果损失，换来了显存与存储数量级的下降；当用省下的显存把 micro-batch 开大、或换更大模型时，它们的吞吐与可行性优势才会真正体现。

## 7. 过程中踩到的坑（也是重要知识点）

1. **Full 在 micro-batch=4 时第一步反向即 OOM（峰值 11.85G）。** 解决：micro-batch 降到 2、梯度累积升到 8，全局 batch 仍是 16，数学等价但激活显存减半。
2. **训练能跑完，却在 epoch 末评测时 OOM。** Qwen 词表高达 151936，评测时 logits 张量大小约为 batch × seq × vocab，非常占显存。解决：单独把 per_device_eval_batch_size 设为 1（评测 batch 不影响训练，也不改变 eval_loss 的定义）。
3. **显存碎片会让“看起来够”的显存申请失败。** PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 可显著缓解。
4. **对照实验必须统一 micro-batch 与评测 batch。** 早期 LoRA 用 micro-batch=4 时采样峰值 11.3G，反而比 micro-batch=2 的 Full(11.1G) 高，这是 batch 不同造成的假象；统一口径后显存阶梯才恢复为 Full > LoRA > QLoRA。
5. **离线机器必须关闭 HF 联网校验**，否则数据加载阶段会空等网络（见第 4 节环境变量）。

## 8. 局限性（如何正确看待这些数字）

- 单卡、单模型（0.5B）、单随机种子、只训 1 个 epoch、评测集仅 300 条；
- 因此 eval_loss 的微小差异不足以给三种方法的“最终效果”排名，严谨结论需要更大数据、多个 seed 与下游任务评测；
- 本实验价值在于可复现地展示四组工程指标的数量级差异与取舍，而不是宣称某种微调方式“效果最好”；
- 未测推理延迟/吞吐（extra_infer_latency 留空），需要时可固定输入输出长度，用同一套推理后端补测。