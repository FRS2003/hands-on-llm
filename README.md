# LLaMA-Factory Lab（大模型高效微调与分布式训练实验）

> 源码级复现业界主流 LLM 微调框架 [hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)
>（Apache-2.0，ACL 2024），并围绕**对齐算法、参数高效微调、分布式并行、量化推理**做系统的对照实验与二次开发。
>
> 复现重点不是“会用命令”，而是：读懂 Model Loader / Data Worker / Trainer 源码链路，
> 并用小模型（Qwen2.5-0.5B/1.5B）在可控变量下产出自己的对比数据与选型结论。

## 一、实验内容
1. **对齐算法横向对比**：SFT / PPO / DPO / KTO / ORPO / SimPO —— 公式推导、损失与负样本构造、显存/稳定性/收敛/效果（`experiments/alignment_compare.csv`）。
2. **PEFT 方法对比**：Full / Freeze / LoRA / QLoRA / Adapter / Prefix / P-Tuning / DoRA —— 可训练参数量、显存、精度、耗时、推理延迟（`experiments/peft_compare.csv`）。
3. **分布式训练**：DDP / TP / PP 与 DeepSpeed ZeRO-1/2/3 —— 单卡显存、吞吐、通信占比与折中（`experiments/distributed_compare.csv`）。
4. **量化与推理部署**：FP16 / GPTQ / AWQ + vLLM —— 精度(PPL)、显存、首 Token 延迟、TPOT、吞吐（`experiments/quant_infer.csv`）。
5. **框架二次开发**：自定义多轮/偏好数据集、训练回调、Loss 与指标可视化、断点续训（`custom/`）。

## 二、目录结构
| 目录 | 内容 |
| --- | --- |
| `configs/` | 各实验的 yaml 配置（对齐/PEFT/分布式/量化），只改一个变量做对照 |
| `experiments/` | 四张对比实验记录表 + 原始日志、loss 曲线 |
| `notes/` | 源码走读笔记、对齐算法公式推导 |
| `custom/` | 自定义数据集加载、回调、可视化等二次开发代码 |

## 三、环境
```bash
conda create -n llmfactory python=3.10 -y && conda activate llmfactory
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory && pip install -e ".[torch,metrics]"
# 训练：llamafactory-cli train ../configs/xxx.yaml
```

## 四、实验 Checklist
- [ ] 源码走读：Model Loader / Data Worker / Trainer 主链路
- [ ] SFT / LoRA 跑通基线
- [ ] 六种对齐算法对照（同数据同模型，只改 stage）
- [ ] 8–9 种 PEFT 方法对照
- [ ] DDP + ZeRO-1/2/3 分布式对照（torchrun 多进程）
- [ ] GPTQ/AWQ 量化 + vLLM 部署，测延迟与吞吐
- [ ] 自定义数据集 + 训练回调二次开发

## 五、结果汇总
详见 `experiments/` 下四张表（持续补充真实数据，每个数字附日志/截图）。

## 六、参考与致谢
- 框架：[hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)（Apache-2.0，ACL 2024）
- 论文：DPO、KTO、ORPO、SimPO、LoRA、QLoRA、DoRA、ZeRO、PagedAttention

> `configs/` 基于官方示例改写并标注改动；`experiments/`、`notes/`、`custom/` 为本人实验、笔记与开发。
