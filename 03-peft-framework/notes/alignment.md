# 对齐算法选型：SFT / DPO / KTO / ORPO / SimPO / PPO 怎么选

> **所属模块**：第 ③ 模块「工业框架」，公式推导见 ② 的 [`../../02-train-from-scratch/notes/alignment.md`](../../02-train-from-scratch/notes/alignment.md)，本篇聚焦**工程选型与框架配置**
> **前置知识**：交叉熵、SFT/DPO 的基本思想（② 已推过 DPO 闭式解与 GRPO）
> **文档定位**：② 回答"损失为什么这样推导"，本篇回答"**手上的数据和资源，到底该用哪种对齐方法、在 LLaMA-Factory 里怎么配**"，并讲清 KTO/ORPO/SimPO 相对 DPO 改了什么。

---

## 📖 写在前面：六种方法在一条演进线上

对齐方法看起来多，其实都在回答两个问题：**要不要成对偏好数据？要不要额外模型（reference / reward / critic）？** 沿着这条线，所有方法都是在给 PPO"做减法"或给 DPO"打补丁"：

```
PPO   ：要 RM + ref + critic，在线采样，最全也最重
DPO   ：去掉 RM/critic，用成对偏好 + ref，离线
KTO   ：DPO 的"数据减负版"——连成对都不要，只要好/坏二元标签
ORPO  ：SFT+偏好一步到位，连 ref 和独立 SFT 阶段都去掉
SimPO ：DPO 去掉 ref 模型，用自身平均对数概率当奖励
```

---

## 0. 总览对照表（选型先看这张）

| 方法 | stage 值 | 数据形式 | 要 ref? | 要 RM/critic? | 是否需先 SFT | 最适合 |
| --- | --- | --- | --- | --- | --- | --- |
| SFT | `sft` | 指令-回答 | 否 | 否 | —— | 一切对齐的地基 |
| DPO | `dpo` | 成对偏好(chosen/rejected) | 是 | 否 | 建议先 SFT | 有成对偏好、追求稳 |
| KTO | `kto` | 每条带好/坏二元标签 | 是 | 否 | 建议先 SFT | 只有点赞/点踩、难凑对 |
| ORPO | `orpo` | 指令-回答（无需偏好对） | 否 | 否 | 否（自带 SFT 项） | 想一步完成 SFT+偏好 |
| SimPO | `simpo` | 成对偏好 | **否** | 否 | 建议先 SFT | 显存紧张、想省 ref 前向 |
| PPO | `ppo` | prompt + reward model | 是 | RM + critic | 必须先 SFT | 有可交互奖励、追求上限 |

> ref = reference model（冻结的 SFT 模型，做 KL 锚点）。RM = reward model，critic = 价值网络。

---

## 1. 先回顾 DPO（后面都拿它当基准）

DPO 损失（推导见 ②）：

```
L_DPO = − log σ( β · [ (logπ(y_w)−logπ_ref(y_w)) − (logπ(y_l)−logπ_ref(y_l)) ] )
```

两个工程代价：① 要存/前向一个 **reference 模型**（多一份显存与计算）；② 数据必须**严格成对**（同一 prompt 的更好/更差回答）。KTO、ORPO、SimPO 正是分别从这两点动刀。

---

## 2. KTO：把"成对偏好"降成"好/坏二元标签"

### 2.1 解决什么痛点

现实中很多反馈是**不成对**的：用户对一条回答点了赞/踩、客服把一条回复标为"合格/不合格"，你很难保证每个点赞都恰好配一个同题的点踩。DPO 用不了这种数据，KTO（Kahneman-Tversky Optimization, 2024）可以。

### 2.2 改动点

- 输入只要每条样本带一个二元标签（好 / 坏），**不需要配对**；
- 仍保留 reference 模型；
- 用行为经济学的**前景理论**构造一个随批次动态估计的"期望锚点" z：好回答的对数似然要高于锚点、坏回答要低于锚点，损失形如两个 sigmoid 之和。

### 2.3 选型建议

当你能拿到大量"单条质量标签"却凑不齐 chosen/rejected 对时，KTO 几乎是唯一顺手的选择；代价是没有同题对比，信号噪声比成对数据略低，需要更关注批次内好坏样本的均衡。

---

## 3. ORPO：把 SFT 和偏好对齐合并成一步

### 3.1 改动点

ORPO（Odds-Ratio Preference Optimization, 2024）**不要 reference、也不要求先做独立 SFT**，在一个损失里同时放两项：

```
L_ORPO = L_SFT（标准交叉熵，学 chosen）  +  λ · L_OR（odds-ratio 惩罚项，拉开好坏）
```

- `L_SFT` 让模型学会生成正确回答；
- odds-ratio 项用"生成好回答相对坏回答的胜率（比值比）"做惩罚，一步把"会答"和"更偏好哪个"都学了。

### 3.2 优缺点

- 优点：**单阶段、单模型、无 ref**，训练流程最短、显存最省，不会出现"SFT 后再 DPO 导致轻微能力回退"的两段式问题；
- 代价：一项损失干两件事，λ 权重和数据质量更敏感，复杂任务上有时不如"先 SFT 再 DPO"稳。
---

## 4. SimPO：去掉 reference 模型的 DPO

### 4.1 改动点

DPO 里的 `log π_ref` 需要一个冻结参考模型常驻显存。SimPO（Simple Preference Optimization, 2024）观察到：可以直接用**策略模型自身对序列的平均对数概率**当"隐式奖励"，于是：

- **完全去掉 reference 模型**，省一次前向和一份显存；
- 对序列长度做**长度归一化**（平均 log prob），避免模型靠"写得短"投机取巧；
- 引入一个 **target margin γ**，要求好回答的奖励比坏回答至少高出一个间隔，而不是仅仅大于。

### 4.2 与 DPO 的取舍

SimPO 更省显存、实现更简单，在不少基准上与 DPO 持平甚至更好；但没有 ref 锚点，约束更弱，对学习率和 margin 更敏感，训练发散风险略高。**显存吃紧（小卡）时优先试 SimPO，求稳用 DPO。**

---

## 5. PPO 与 GRPO：在线强化这一路

- **PPO** 是最重的通用方案：需要 reward model + reference + critic，在线 rollout，四模型协作（② 已详述）。上限高但调参难、显存大。
- **GRPO**（DeepSeekMath）用"同 prompt 采样 G 条、组内标准化优势"替掉 critic，特别适合**答案可自动验证**的数学/代码任务（RLVR）。本学习库 ② 手写实现了 GRPO 并用一位加法任务验证（规则分 0.109→0.290）。
- 在 LLaMA-Factory 里，PPO/GRPO 走 `stage: ppo`，并需要额外配置奖励模型或可验证奖励函数，成本远高于上面四种离线方法。

> 经验顺序：**数据/资源有限时，SFT → DPO（或 SimPO/KTO/ORPO 视数据形态）通常就是性价比最高的组合**；只有当你确实有可在线交互的奖励信号、且离线偏好优化到了瓶颈，才值得上 PPO/GRPO。

---

## 6. 偏好数据怎么构造（所有离线方法的共同前提）

| 数据形态 | 长这样 | 适配方法 |
| --- | --- | --- |
| 指令-回答 | {instruction, output} | SFT / ORPO |
| 成对偏好 | {prompt, chosen, rejected} | DPO / SimPO |
| 二元反馈 | {prompt, response, label:好/坏} | KTO |
| prompt + 打分器 | {prompt} + reward model | PPO/GRPO |

常见的 chosen/rejected 来源：① 同一问题让不同模型/不同采样温度回答，人工或用更强模型排序；② 用规则改写（正确答案 vs 典型错误答案）；③ 真实用户反馈中同题的优/劣回复。**偏好对必须针对同一 prompt，否则损失没有意义。**

---

## 7. 在 LLaMA-Factory 里怎么配（最小差异对照）

这些方法共用同一份基座和数据，切换时主要改 `stage` 与少量专属字段：

```yaml
### DPO ###
stage: dpo
finetuning_type: lora
pref_beta: 0.1            # β，对应公式里的 KL/间隔强度
pref_loss: sigmoid        # sigmoid=DPO；改成 simpo/ipo/orpo/kto 即切换同族损失

### KTO ###
stage: kto
pref_beta: 0.1

### ORPO / SimPO：很多版本通过 pref_loss 或独立 stage 切换，以本地版本文档为准 ###
stage: orpo               # 或 stage: simpo
```

> 字段名会随版本演进（如 `pref_loss`、`beta`/`pref_beta`），**以你本地版本的 `hparams/finetuning_args.py` 字段为准**——这正是 [`source_walkthrough.md`](source_walkthrough.md) 第 1 节教你的"字段去哪查"。

---

## 8. 一分钟选型决策

```
你有什么数据？
├─ 只有指令-回答 ──────────────► 先做 SFT；想顺带对齐可试 ORPO
├─ 能凑出同题 chosen/rejected
│   ├─ 显存充足、求稳 ─────────► DPO
│   └─ 显存紧张、想省 ref ──────► SimPO
├─ 只有点赞/点踩这类单条标签 ────► KTO
└─ 有可自动判分的环境(数学/代码) ► GRPO/PPO（成本最高，最后再考虑）
```

> 一句话：**先看数据形态排除一批，再看显存与是否要 reference 排除一批，剩下的就是合适的方法**——没有绝对最优，只有和数据、资源、稳定性约束最匹配的那个。