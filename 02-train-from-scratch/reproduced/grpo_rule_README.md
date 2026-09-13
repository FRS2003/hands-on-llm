# GRPO 复现说明：纯规则奖励版（Rule-based Reward，免 1.8B 奖励模型）

## 为什么要改造
官方 `trainer/train_grpo.py` 在主流程里**无条件**加载一个 1.8B 的学习型奖励模型：

```python
reward_model = LMForRewardModel(args.reward_model_path, device=args.device, dtype=torch.float16)
# 默认路径 ../../internlm2-1_8b-reward，fp16 约 3.6GB，且每个训练步都对生成结果 get_score
```

而 GRPO 本身并不依赖学习型奖励模型——它只需要一个标量奖励，可以来自**规则 / 可验证结果 / 奖励模型**任意一种。
官方 `calculate_rewards` 已内置规则奖励（回答长度、`</think>` 思考标签、3-gram 重复惩罚），
因此本复现把学习型奖励模型替换为一个恒返回 0 分的占位对象，**奖励完全由规则项给出**，
从而：免下载 3.6GB 模型、显存更省、机制完整可观察。仓库中如实标注这是 rule-based 变体。

## 精确改动（最小 diff，其余官方代码原样保留）
把 `train_grpo.py` 第 280 行那一句替换为：

```python
class _DummyRuleReward:
    """get_score 恒为 0；奖励只来自 calculate_rewards 的规则项。"""
    def get_score(self, messages, response):
        return 0.0
reward_model = _DummyRuleReward()  # 原为 LMForRewardModel(args.reward_model_path, ...)
```

规则奖励（官方原样，未改）：
- 回答长度落在 20–800 字符：+0.5，否则 -0.5
- 含 `</think>`：思考段长度 20–300 给 +1.0（否则 -0.5）；恰好 1 个标签 +0.25（否则 -0.25）
- 减去 3-gram 重复惩罚 `rep_penalty`（上不超过 0.5）
- 学习型奖励分：本变体恒为 0

## 数据
- 全量 `dataset/rlaif.jsonl`：来自 modelscope `gongjy/minimind_dataset`，23.75MB / 19,502 条，字段为 `conversations`。
- `rlaif_sub.jsonl`：等间隔抽样 600 条作为单卡训练子集（与 pretrain/SFT 子集同一抽样方法，保证分布覆盖）。
- GRPO 只用每条的问题部分生成 prompt，答案由模型在线 rollout 产生，不读 jsonl 里的回答。

## 训练配置与理由
| 参数 | 取值 | 说明 |
| --- | --- | --- |
| from_weight | full_sft | policy 与 reference 都从 SFT 权重热启动 |
| batch_size × num_generations | 2 × 4 | 每步在线生成 8 条，组内比较（官方默认 2×6） |
| max_seq_len / max_gen_len | 512 / 384 | 官方 768/1024，单卡缩短以控时长 |
| learning_rate / beta | 3e-7 / 0.1 | 官方默认；RL 阶段学习率极小，KL 约束防跑偏 |
| loss_type | cispo | 官方默认（GRPO 的 PPO-clip 变体亦可切换） |

## 验证方式（无卡模式下先把能验的都验掉）
1. **运行时**：无卡 CPU 上改造脚本能正常启动——policy/reference 两个 63.91M 模型初始化、
   RLAIFDataset 加载、进入训练循环并开始 rollout，全程不再请求 1.8B 奖励模型。
2. **静态核对**：`rollout_engine.RolloutResult` 的 6 个字段与训练脚本消费端逐一对应。
3. **数值自检**：`from_scratch/test_grpo_logic.py` 覆盖规则奖励、组内优势中心化、
   KL 非负、初始 ratio=1、CISPO loss 可反传，共 14 项断言全过（CPU 秒级）。
4. 已在有卡 RTX 3080 Ti 上执行 `bash start_grpo.sh` 跑完全部 300 步，结果见下节「实际训练结果」。

## 实际训练结果（2026-09-13，单卡 RTX 3080 Ti 12GB）
- 300 个 optimizer step，wall **21m40s**（≈4.33 s/step，在线 rollout 生成是主要开销），显存峰值约 **9.9 GB**。
- Reward（规则分；batch=2 噪声大）：全程均值 0.391，中段(step101–200) 0.494；LR 3e-7→3e-8 余弦衰减。
- KL(ref)：均值 -0.0042、平均 |KL|=0.0051、区间 [-0.024, 0.008]，始终贴近 0，**策略未漂移**。
- 组内优势：max|Adv Mean|=0（组内中心化正确）、Adv Std 均值 0.93（组内确有区分度、梯度有效）。
- Actor(policy) loss 0.069±0.091，围绕 0 波动（策略梯度目标本就不趋零）。
- **前后对比**（5 个相同 prompt、同种子/采样参数，脚本 `compare_grpo_gen.py`）：平均规则分 **full_sft 0.109 → grpo 0.290**，最佳一条 0.047→1.750（满分）。
- 产物：`../experiments/grpo/`（grpo_train.log、grpo_curve.csv、metrics.txt、gen_grpo_compare.txt），曲线图 `../experiments/assets/grpo_curve.png`。

## 客观局限
规则奖励只约束**长度/格式/不重复**，不判断内容正确性，因此本实验用于**完整走通 GRPO 机制、观察奖励/KL/优势的真实变化**，不声称内容正确性提升（实测前后对比中提升的正是格式维度）；batch=2 使 Reward 噪声较大、有 1 条生成在闭合 think 标签前触及 512 token 上限，均如实保留。