# configs · 可直接套用的训练配置模板

每份 yaml 都带逐行中文注释，**复制后改 `model_name_or_path` 与 `dataset` 即可跑**。
字段含义与源码位置见 [`../notes/source_walkthrough.md`](../notes/source_walkthrough.md)，方法选型见 [`../notes`](../notes)。

| 文件 | 用途 | 关键点 |
| --- | --- | --- |
| `sft_lora_qwen0.5.yaml` | **首选**：LoRA 指令微调 | r=8/alpha=16、target=all、lr 2e-4、梯度累积 |
| `sft_qlora_qwen1.5.yaml` | 小卡跑更大模型：4bit 量化 + LoRA | `quantization_bit: 4`、batch=1、累积补回 |
| `sft_full_qwen0.5.yaml` | 全参 SFT 基线（LoRA 的效果上界对照） | finetuning_type=full、lr 1e-5 |
| `dpo_lora_qwen0.5.yaml` | DPO 偏好对齐 | stage=dpo、成对数据、pref_beta、lr 5e-6 |
| `ds_zero2.json` | DeepSpeed ZeRO-2 并行配置 | yaml 里 `deepspeed:` 指向它，torchrun 启动 |

## 对照实验原则（重要）

研究某个变量时，**只改它、其余全部保持一致**，结果才可比：

- 研究 LoRA 秩：复制 `sft_lora`，只改 `lora_rank`（8/16/32/64）；
- 研究微调方法：固定同一份数据与步数，只切换 `finetuning_type`（full/lora）或量化开关；
- 研究对齐算法：固定 SFT 起点，只切换 `stage`（dpo/kto/orpo）与对应数据；
- 研究并行：固定全局 batch，只改 `ds_zero*.json` 的 stage。

## 多卡启动示例

```bash
# 单卡
llamafactory-cli train sft_lora_qwen0.5.yaml
# 两卡 + ZeRO-2（yaml 内加 deepspeed: configs/ds_zero2.json）
torchrun --nproc_per_node 2 -m llamafactory.cli train sft_lora_qwen0.5.yaml
```

> 字段名随 LLaMA-Factory 版本演进，以本地 `src/llamafactory/hparams/*.py` 的 dataclass 为准。