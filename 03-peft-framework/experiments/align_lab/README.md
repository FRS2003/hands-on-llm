# 对齐对照实验：SFT vs DPO（DPO 偏好对齐实测）

本实验包用同一基座、同一套 LoRA 超参，对比「监督微调 SFT」与「直接偏好优化 DPO」两种训练阶段在**数据形态、是否需要参考模型、显存/耗时、损失曲线**上的差异。SFT 基线直接复用 [`../peft_lab/`](../peft_lab/README.md) 的 LoRA-SFT 结果，本目录新增一次真实的 DPO 训练。

## 1. 实验要回答的问题

- SFT 只需要「问题→标准答案」，为什么 DPO 必须成对给出「更好回答 chosen / 更差回答 rejected」？
- DPO 为什么需要一个 **reference model（参考模型）**？它的损失为什么训练一开始约等于 `0.693 = -ln2`？
- 训练过程中 `rewards/chosen`、`rewards/rejected`、`rewards/margins`、`rewards/accuracies` 分别怎么看，对齐"成功"长什么样？
- 在同一张卡上，DPO 比 SFT 多花多少显存、慢多少，慢在哪里？

## 2. 数据：一个真实的中文偏好数据集

- 来源：ModelScope 上的 `shareAI/shareAI-Llama3-DPO-zh-en-emoji`（Apache-2.0）。它对同一问题分别采样了中文回答与英文回答，构成「偏好中文、不偏好英文」的**语言偏好对**：`chosen = 中文回答`，`rejected = 英文回答`。
- 用 `prepare_pref.py` 把原始 jsonl 转成 LLaMA-Factory 的 alpaca 偏好格式（`instruction / chosen / rejected`，并在 `data/dataset_info.json` 里以 `ranking: true` 注册为偏好数据）。
- 长度用 **Qwen tokenizer 按真实 token 数**过滤（两条回答各自落在 cutoff 1024 内，留 24 token 模板余量），固定 `seed=42` 打乱后切分：**训练 2243 条 / 评测 150 条**，pair 长度中位数 485、p90 为 688。
- 选语言偏好对的原因：chosen/rejected 的差异清晰、可验证，便于把 DPO 的内部机制（奖励分离、margin 扩大）讲透；它不追求复杂任务上的最终胜率，只演示对齐流程。

## 3. 控制变量与公平性说明

| 项 | SFT 基线（peft_lab） | DPO（本目录） |
| --- | --- | --- |
| 基座 | Qwen2.5-0.5B | 同左 |
| 微调方式 | LoRA r=8 / alpha=16 / dropout=0.05 / target=all | 同左（可训参数均为 4,399,104，占 0.8826%） |
| 精度 / 梯度检查点 / 种子 | bf16 / 开 / 42 | 同左 |
| 全局 batch | 16（micro2×累积8） | 16（micro1×累积16，原因见第 7 节） |
| 数据 | 3000 条指令（instruction），cutoff 512 | 2243 对偏好（chosen+rejected），cutoff 1024 |
| 学习率 / epoch | 1e-4 / 1 | **5e-6**（对齐阶段更小，防灾难性遗忘）/ 1 |
| DPO 参数 | — | `pref_beta=0.1`，`pref_loss=sigmoid` |

> **公平性边界（重要）**：SFT 与 DPO 属于不同训练阶段，数据类型与序列长度本就不同，因此**显存、耗时的绝对值不能当作"谁更快"的严格结论**，这里对照的是两种算法的**机制差异与资源结构**。真正的端到端比较应当是「base → SFT → DPO」两阶段后看同一评测集，本实验不展开。

## 4. 目录结构

```
align_lab/
├── README.md                  # 本文档
├── prepare_pref.py            # 原始偏好数据 -> LLaMA-Factory 偏好格式 + 固定切分
├── run_dpo.sh                 # 训练 + 逐秒显存采样 + 记录墙钟时间
├── yamls/
│   ├── align_dpo.yaml         # 正式 DPO-LoRA 配置
│   └── align_dpo_smoke.yaml   # 3 步冒烟配置（先验证 loss≈0.693）
├── data/
│   ├── dataset_info.json      # 偏好数据注册（ranking=true）
│   ├── pref_dpo_train.json    # 2243 对
│   └── pref_dpo_eval.json     # 150 对
├── logs/
│   ├── dpo.log / dpo_smoke.log          # 完整训练日志
│   ├── dpo_gpu.csv                      # 逐秒显存采样（MB）
│   └── dpo_stat.txt / *_smoke_stat.txt  # 墙钟时间与返回码
└── results/dpo_lora/
    ├── all_results.json / train_results.json / eval_results.json
    ├── trainer_state.json / adapter_config.json
    └── training_loss.png / training_rewards_accuracies.png / training_eval_loss.png
```

权重文件（`*.safetensors`）与词表、断点不入库，见本目录 `.gitignore`。

## 5. 复现步骤

```bash
# 1) 激活隔离环境（与 peft_lab 同一环境，依赖见 ../peft_lab/requirements-lock.txt）
conda activate <你的环境>
# 2) 下载偏好数据到数据盘后，转换并固定切分（也可直接用 data/ 下已切好的 json）
python prepare_pref.py
# 3) 先跑 3 步冒烟，确认初始 DPO loss≈0.693、参考模型正常建立
bash run_dpo.sh yamls/align_dpo_smoke.yaml dpo_smoke
# 4) 正式训练（约 14 分钟，含两次评测）
bash run_dpo.sh yamls/align_dpo.yaml dpo
```

## 6. 实测结果（RTX 3080 Ti 12G）

| 指标 | SFT（LoRA，复用 peft_lab） | DPO（LoRA，本次） |
| --- | --- | --- |
| 需要 reward model | 否 | 否（DPO 用隐式奖励，无需训练 RM） |
| 需要 reference model | 否 | **是**（LoRA 下用"关掉 adapter 的基座"隐式充当，不额外占一整份模型） |
| 数据类型 | instruction | preference_pair（chosen/rejected） |
| 可训参数 / 占比 | 4,399,104 / 0.8826% | 同左 |
| 显存峰值 | 3699 MB | **8333 MB** |
| 纯训练时长 | 7.61 min | **13.81 min**（828.8 s，141 步） |
| 最终训练 loss | 1.8400（交叉熵） | **0.2816**（DPO 损失，从 0.6889 起步） |
| 评测指标 | eval_loss 2.0067 | eval_loss 0.0797；**偏好准确率 1.0** |
| 奖励结构 | 无 | chosen +0.84 / rejected −1.96 / **margin +2.80** |
| 产物 | adapter 16.8 MB | adapter 17 MB |

DPO 曲线（见 `results/dpo_lora/training_*.png`）的完整变化：

- **损失**：第一步约 `0.6889 ≈ -ln2 = 0.693`，随后单调降到 0.28。初始为 0.693 是因为训练刚开始时 policy 与 reference 完全相同，sigmoid 输入为 0，`-ln σ(0) = ln2`。这是判断 DPO 是否正确启动的关键信号。
- **奖励分离**：`rewards/chosen` 由 0 附近升到正的 +0.84，`rewards/rejected` 降到 −1.96，二者之差 `rewards/margins` 从 0 扩大到 +2.8，说明模型确实把更高的隐式奖励分给了 chosen。
- **偏好准确率**：`rewards/accuracies` 很快升到 1.0，评测集 150 对全部能正确排序（语言偏好任务本身较易，见第 8 节局限）。

## 7. 过程中踩到的坑（也是知识点）

1. **YAML 里裸写 `eval_strategy: no` 直接报错**：YAML 1.1 会把未加引号的 `no` 解析成布尔 `False`。需要禁用评测时要么加引号写成 `"no"`，要么直接删掉该字段。
2. **micro-batch2 时 DPO 峰值冲到 11.9G，几乎顶满 12G**：DPO 每步要对 chosen、rejected 分别在 policy 和 reference 上前向（相当于 4 次前向 + 两次反向），比 SFT 吃显存。改为 **micro1×累积16**（全局 batch 仍是 16，数学等价）后峰值降到 8.3G，留出安全余量。
3. **按字符数过滤长度会误杀英文**：同样 1024 token，英文对应的字符数远多于中文，用统一字符阈值会丢掉一大半样本。改为用 **tokenizer 真实编码后的 token 数**过滤，保留率从约 30% 提升到 97%。

## 8. 局限性（如何正确看待这些数字）

- 本次是**语言偏好**任务，chosen/rejected 差异明显，准确率很快到 1.0 并不代表在复杂推理/事实性偏好上也能满分；换更难的偏好对，margin 与准确率会更温和。
- 只实测了 SFT 与 DPO；`alignment_compare.csv` 中 PPO/KTO/ORPO/SimPO 的机制是确定的（是否需要 RM/ref 已按原理填好），但**资源数字一律留空**，未实测不臆造。
- 如第 3 节所述，SFT 与 DPO 数据、序列长度不同，不对"端到端谁效果好"下结论；规范做法是 base→SFT→DPO 两阶段后在同一评测集比较。
- 结果来自单张 RTX 3080 Ti、单一 0.5B 模型，绝对数值随硬件/模型规模变化，复现时请关注**趋势与相对关系**而非具体数值。