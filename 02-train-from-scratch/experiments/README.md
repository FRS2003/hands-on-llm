# 实验记录（Experiments）

> 单卡 **RTX 3080 Ti 12GB** 上，从零复现 MiniMind（63.9M）的 **继续预训练 → 监督微调（SFT）→ LoRA 参数高效微调 → DPO 偏好对齐 → GRPO 强化学习** 全流程。
> 所有数字均来自真实训练日志（`*/` 目录下的 `*_curve.csv`、`metrics.txt`），曲线图见 `assets/`。

## 1. 运行环境

| 项 | 配置 |
| --- | --- |
| GPU | 1 × NVIDIA RTX 3080 Ti（12 GB GDDR6X），驱动 580.76.05 |
| 框架 | Python 3.10、PyTorch 2.6.0+cu124、Transformers 4.57.6、datasets 3.6.0 |
| 精度 | BF16 混合精度（`torch.cuda.amp.autocast`） |
| 模型 | MiniMind，hidden=768、8 层、8 头、**GQA（KV 头=4）**、词表 6400，主干 **参数量 63.91M** |
| 数据 | ModelScope `gongjy/minimind_dataset`：pretrain 全量 127.0 万条、SFT 全量 90.6 万条多轮对话 |

> 复现踩坑：该机器上 `pip` 直接拉取数百 MB 的大 wheel（torch / nvidia-cudnn 等）会零增长卡死，
> 最终改用 **curl 逐包下载 + `pip install --no-index --find-links` 离线安装**解决，详见 `../reproduced/`。

## 2. 数据准备：等间隔抽样

全量数据在单卡上训练 2 轮需要十余小时。为在可控时间内完成端到端复现并对比**数据规模效应**，
采用**等间隔抽样**（每隔 k 行取 1 条，覆盖整个文件、避免头部数据分布偏斜）构造两档子集：

| 子集 | pretrain 条数 | SFT 条数 | 占全量比例（pretrain/SFT） |
| --- | --- | --- | --- |
| small | 60,000 | 20,000 | 4.7% / 2.2% |
| medium | 150,000 | 50,000 | 11.8% / 5.5% |

抽样脚本见 `../reproduced/make_subsets.py`。

## 3. 超参数

| 阶段 | batch | 梯度累积 | 有效 batch | 学习率 | 调度 | 最大长度 | epochs |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Pretrain | 32 | 8 | 256 | 5e-4 | cosine + warmup | 340 | 2 |
| SFT | 16 | 1 | 16 | 1e-5 | cosine | 768 | 2 |
| LoRA | 32 | 1 | 32 | 1e-4 | cosine | 340 | 3 |
| DPO | 4 | 1 | 4 | 4e-8 | — | 1024 | 1 |

优化器 AdamW、梯度裁剪 1.0；SFT 从上一阶段 pretrain 权重热启动，且**仅对 assistant 回复部分计算损失**；
LoRA 在 medium 的 full-SFT 基座上只训练注入的低秩适配器（rank 适配 Q/V 等投影），主干全部冻结。

## 4. Pretrain + SFT 结果

![loss curves](assets/loss_curves.png)

| 实验 | 阶段 | 训练步数 | 耗时 | 起始→最终 loss | 显存峰值 | 平均 GPU 利用率 |
| --- | --- | --- | --- | --- | --- | --- |
| **small** | Pretrain | 3,750 | 10.3 min | 7.73 → **3.35** | 7.36 GB | 97.1% |
| **small** | SFT | 2,500 | 8.0 min | 3.91 → **3.17** | 7.36 GB | 97.1% |
| **medium** | Pretrain | 9,376 | 25.5 min | 7.31 → **2.53** | 7.36 GB | 98.8% |
| **medium** | SFT | 6,250 | 19.6 min | 2.82 → **2.21** | 7.36 GB | 98.8% |

## 5. LoRA 参数高效微调（PEFT）

在 medium full-SFT 基座上，用同一份 2 万条 SFT 子集训练 LoRA 适配器（3 轮、1875 步、**256 秒**）：

![lora compare](assets/lora_compare.png)

| 指标 | 全量 SFT | LoRA | 对比 |
| --- | --- | --- | --- |
| 可训练参数量 | 63.91 M | **0.393 M** | LoRA 仅为全量的 **1/163（0.61%）** |
| 检查点体积 | 132 MB | **0.779 MB** | 缩小约 169 倍，便于多任务分发 |
| 显存峰值 | 7.36 GB | **4.60 GB** | 降低约 37%（冻结主干、优化器状态更少） |
| 20k 数据训练耗时 | 479 s（2 epoch） | **256 s（3 epoch）** | 轮次更多反而更快 |
| loss | 收敛到 2.21 | 在 2.15–2.57 间围绕基座波动 | 见下方分析 |

**分析（诚实）**：LoRA 从已经拟合好的 full-SFT 基座起步，loss 本就处于 2.2 水平，再用**同分布**数据训练不会显著下降，
而是围绕基座小幅波动——这正说明低秩适配**没有破坏基座已有能力**（无灾难性遗忘），其价值在于以极小代价做**增量/领域适配**。
推理时把 0.78MB 适配器叠加到冻结基座即可正常生成（`eval_llm.py --weight full_sft --lora_weight lora_demo`），
且模型对「你是谁」类身份认知回答更稳定。若要体现 LoRA 的领域增益，应换用医疗/身份等**专门领域小数据**（对应官方 `lora_medical`）。

## 6. DPO 偏好对齐（Direct Preference Optimization）

在 medium full-SFT 基座上，用 **17,166 对 chosen/rejected** 偏好数据做 DPO（β=0.15、lr=4e-8、batch=4、1 轮 4292 步、**744 秒**）。
DPO 需同时持有 policy 与冻结的 reference 两个模型，故显存略升到 5.71GB。

![dpo curve](assets/dpo_curve.png)

| 指标 | 数值 |
| --- | --- |
| 偏好对 / 步数 | 17,166 对 / 4,292 步 |
| 耗时 | 12.4 min |
| 理论初始 loss | -ln2 = **0.693**（chosen、rejected 等概率） |
| 区间均值 | 前半 0.6415 → 后半 **0.6201** |
| 末步 / 最低 loss | 0.4276 / 0.3908 |
| 显存峰值 / 利用率 | 5.71 GB / 98.4% |

**分析**：DPO loss 从理论值 -ln2≈0.693 起步，训练后移动平均缓慢下移，说明 chosen 相对 rejected 的对数概率差（隐式 reward margin）被逐步拉开；
因学习率刻意取极小（4e-8，防止灾难性遗忘）且 batch=4 噪声大，曲线呈「高噪声、缓慢下降」形态。
DPO **只调整输出偏好/风格、不增加知识**：对齐后回答更简短收敛（见 `generation_samples.md`），但 63M 的知识边界不变。

## 7. GRPO 强化学习（纯规则奖励 / CISPO）

在 full_sft 基座上做 GRPO，**移除官方无条件加载的 1.8B 学习型奖励模型**（fp16≈3.6GB），奖励完全由内置规则给出（回答长度 20–800、`</think>` 思维段 20–300、标签恰好 1 个、3-gram 重复惩罚），改造最小 diff 见 [`../reproduced/grpo_rule_README.md`](../reproduced/grpo_rule_README.md)。

![grpo curve](assets/grpo_curve.png)

| 指标 | 数值 |
| --- | --- |
| prompt / 步数 | 600 / 300（batch 2 × num_gen 4） |
| 耗时 / 速度 | 21.7 min / ≈4.33 s·step（在线 rollout 为主） |
| lr / β / loss 类型 | 3e-7→3e-8 余弦 / 0.1 / CISPO |
| Reward 均值 | 全程 0.391，中段(101–200) 0.494（batch=2 噪声大） |
| KL(ref) | 均值 -0.0042、平均 |KL| 0.0051，始终贴近 0 |
| 组内优势 | max|Adv Mean|=0（中心化）、Adv Std 均值 0.93 |
| 显存峰值 | 9.9 GB（policy+reference 双模型 + 8 路生成 KV） |

**前后对比**（同 5 个 prompt、同种子/采样，脚本 `../reproduced/compare_grpo_gen.py`）：平均规则分 full_sft **0.109 → grpo 0.290**，最佳一条 0.047→1.750（满分），长度更受控、思维链格式更规范、重复更少。

**分析（诚实）**：组内优势均值严格为 0、KL 贴近 0、LR 余弦衰减均与理论一致，证明 GRPO 机制正确跑通；但规则奖励只定义"格式/长度/不重复"、**不评判内容对错**，故 63M 模型事实正确性并未提升；batch=2 使 Reward 噪声较大、有 1 条在闭合 think 前触及 512 token 上限。原始日志/逐步 CSV/对比全文见 [`grpo/`](grpo/)。

**100 条 held-out 定量评测**：评测题从全量 19,502 条等间隔抽取、并**排除 GRPO 训练用的 600 条**（脚本 `../reproduced/eval_grpo_100.py`，逐条明细 `grpo/eval_100_detail.csv`）：

| 权重 | 规则分均值 | 长度合规(20–800) | think段合规 | 标签恰1个 | 3-gram重复惩罚 | 平均字符 |
| --- | ---:| ---:| ---:| ---:| ---:| ---:|
| full_sft | 0.255 | 1.00 | 0.05 | 0.77 | 0.120 | 411 |
| grpo | **0.339** | 0.98 | 0.10 | 0.75 | **0.096** | 427 |

样本扩到 100 条后结论更稳健：规则分 **+0.084**、3-gram 重复度下降（0.120→0.096）、think 段长度合规率翻倍；"标签恰 1 个"比例基本持平（-0.02，属采样噪声）。提升温和且集中在"格式/重复度"，再次印证纯规则奖励不改变内容正确性。注：本次 max_new_tokens=300，think 段容易超过规则设定的 300 字符上限，故 think 段合规率绝对值偏低。

## 8. 五阶段统一生成横评（pretrain → GRPO）

用**同一组 6 个 prompt、同一随机种子与采样参数**让五个权重分别生成（pretrain 走续写模式、其余走对话模板；LoRA 是叠加在 SFT 主干上的适配器）。完整对比见独立文档 [`stage_evolution.md`](stage_evolution.md)（逐条原文 `stage_evolution.txt`，脚本 `../reproduced/compare_5stages.py`）。

| 阶段 | 平均字符 | 闭合 think 标签 | 定位 |
| --- | ---:| ---:| --- |
| pretrain | 209 | 0/6 | 只会续写、不遵循指令、事实易错乱 |
| SFT | 544 | 4/6 | 学会助手角色与 markdown/代码块结构 |
| LoRA | 522 | 5/6 | 仅训 0.61% 参数即复现 SFT 的指令遵循 |
| DPO | 529 | 5/6 | 调偏好/措辞，不增加知识 |
| GRPO | 517 | 4/6 | 强 KL 约束下小幅优化格式 |

**核心结论**：能力跃迁最大的是 **pretrain → SFT**（从"不会答题"到"会答题"），之后三个阶段都在 SFT 基础上做精细化调整，五阶段不可互相替代。

## 9. GPU 画像与推理速度

![gpu profile](assets/gpu_profile.png)

- 训练阶段 GPU 利用率长期 96–99%、温度峰值 72–75°C；显存峰值 pretrain/SFT 7.36GB、LoRA 4.60GB。
- 推理（FP16，单条 `model.generate`）解码速度约 **47–100 tokens/s**。

## 10. 关键发现

1. **数据规模的边际收益清晰可见**：pretrain 数据量 ×2.5（6 万→15 万），同架构同超参下最终 loss 由 3.35 降到 2.53；
   SFT 由 3.17 降到 2.21，medium 模型生成的语句连贯度、指令遵循度明显更好（见 `generation_samples.md`）。
2. **Pretrain 与 SFT 分工明确（消融）**：只做 pretrain 的模型只会**续写**、无法遵循指令
   （让写斐波那契函数时输出一串数字）；经过 SFT 后才学会 chat template、以助手身份分点作答。
3. **PEFT 性价比**：LoRA 只训 0.61% 参数、检查点缩小两个数量级、显存降 37%，适合多领域低成本适配与热插拔。
4. **小模型单卡可训**：63.9M 模型在 12GB 卡上 bf16 训练毫无压力，瓶颈在**数据量与训练 token 数**而非显存/算力。
6. **DPO 对齐特性**：loss 从理论值 -ln2 起步、以极小学习率缓慢下移，验证了偏好目标；DPO 改变输出风格而非知识容量，需严格控制 lr 防止遗忘。
5. **loss 与生成质量并不完全等价**：SFT loss 降到 2.2 后模型能稳定输出结构化中文，但受 63M 参数 + 仅约 5% 全量语料限制，
   仍有事实错误、重复、代码语法错误——与 Chinchilla「小模型需要足够 token」的结论一致，是后续扩数据/扩参的方向。

## 11. 目录与复现

- `small/`、`medium/`、`lora/`、`dpo/`：各自的 `*_curve.csv`（逐步 loss/lr）、`metrics.txt`、原始生成记录、GPU 采样 CSV；
- `grpo/`：GRPO 原始训练日志、逐步指标 CSV、metrics 与前后生成对比文本；
- `grpo/eval_100_*`：100 条 held-out 定量评测（指标表 + 200 行逐条明细）；
- `stage_evolution.md / .txt`：五阶段同种子生成横评（解读文档 + 逐条原文）；
- `assets/`：loss 对比曲线、GPU 画像、LoRA 资源对比图；
- `generation_samples.md`：pretrain-only / small-SFT / medium-SFT / LoRA / DPO 生成样例对比；
- 完整复现命令见 [`../reproduced/`](../reproduced/)；逐步汇总见 [`training_log.csv`](training_log.csv)。

> 说明：模型权重（主干 .pth 约 132MB、LoRA 适配器 0.78MB）按 `.gitignore` 约定不上传；数据来自 ModelScope 公开数据集，按其许可使用。
