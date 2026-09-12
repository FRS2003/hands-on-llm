# MiniMind From Scratch（从零构建轻量级语言模型）

> 不依赖高层封装，用原生 PyTorch 逐层实现现代轻量级语言模型的核心组件，并完整跑通
> **预训练 → SFT → LoRA → DPO → GRPO** 的训练链路。
>
> 本仓库复现自 [jingyaogong/minimind](https://github.com/jingyaogong/minimind)（MIT License），
> 并在其基础上补充三部分自己的工作：**① 不看参考实现的手写组件；② 分阶段/多方法对比实验；③ 源码与公式推导笔记。**

## 一、项目目标
- 手写 BPE 分词器与 Transformer Decoder：RMSNorm、RoPE、GQA、SwiGLU、Causal Attention；
- 跑通 “数据清洗去重 → 预训练 → SFT → LoRA → DPO → GRPO” 全流程，记录每阶段 loss / 显存峰值 / 吞吐 / 耗时 / 生成效果；
- 用 Triton 编写 RMSNorm 自定义 Kernel、手写分块 Flash Attention 以降低显存；
- 实践 DeepSpeed ZeRO 分片、梯度检查点、混合精度（BF16）与多卡数据并行；
- 依据 Chinchilla 结论拟合 Scaling Law，核算训练资源。

## 二、目录结构
| 目录 | 内容 |
| --- | --- |
| `from_scratch/` | **手写组件**：不看官方实现，自己写的 tokenizer 与模型模块（核心） |
| `reproduced/` | 基于官方仓库跑通的训练脚本与配置（注明来源，非原创） |
| `experiments/` | 对比实验记录：训练日志 CSV、loss 曲线、显存/耗时、生成样例 |
| `notes/` | 源码阅读笔记、RoPE/DPO/GRPO 等公式推导 |

## 三、运行环境
- OS: Windows 11 / Linux；Python 3.10；CUDA 12.x；PyTorch 2.x
- 单卡即可训练 64M 参数模型（约 2–3 小时）

```bash
conda create -n minimind python=3.10 -y && conda activate minimind
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
python -c "import torch; print(torch.cuda.is_available())"
```

## 四、复现路线 Checklist
- [ ] 环境搭建、数据下载与清洗去重、BPE 分词器训练
- [ ] 预训练（Pretrain）
- [ ] 指令微调（SFT，多轮对话仅对 response 计算 loss）
- [ ] LoRA 参数高效微调
- [ ] DPO 偏好对齐
- [ ] GRPO 强化学习对齐
- [ ] Triton Kernel / 分块注意力
- [ ] DeepSpeed ZeRO + 混合精度 + 梯度检查点

## 五、手写组件 Checklist（`from_scratch/`）
- [ ] BPE Tokenizer（merge 规则、编解码）
- [ ] RMSNorm（对比 LayerNorm）
- [ ] RoPE 旋转位置编码（复数/旋转矩阵两种实现）
- [ ] GQA（MHA / MQA / GQA 对比与 KV Cache 显存计算）
- [ ] SwiGLU 前馈网络
- [ ] Causal Self-Attention（含 mask、张量维度标注）
- [ ] Triton 版 RMSNorm Kernel
- [ ] 分块 Online-Softmax（Flash Attention 核心）

## 六、实验结果
详见 [`experiments/training_log.csv`](experiments/training_log.csv) 与各阶段对比表（持续更新）。

| 阶段 | 参数量 | 显存峰值 | 训练耗时 | 最终 loss | 生成效果 |
| --- | --- | --- | --- | --- | --- |
| Pretrain |  |  |  |  |  |
| SFT |  |  |  |  |  |
| LoRA |  |  |  |  |  |
| DPO |  |  |  |  |  |
| GRPO |  |  |  |  |  |

## 七、学习笔记
- [架构组件推导](notes/architecture.md)：RMSNorm / RoPE / GQA / SwiGLU / 残差与 Pre-Norm
- [对齐算法推导](notes/alignment.md)：SFT 损失、DPO 闭式解、GRPO 组内优势、与 PPO 的区别

## 八、参考与致谢
- 原始项目：[jingyaogong/minimind](https://github.com/jingyaogong/minimind)（MIT）
- 论文：RoFormer(RoPE)、GQA、GLU Variants(SwiGLU)、DPO、DeepSeekMath(GRPO)、FlashAttention、Chinchilla

> 说明：`reproduced/` 内为对原项目的学习性复现，著作权归原作者；`from_scratch/`、`experiments/`、`notes/` 为本人独立实现与记录。
