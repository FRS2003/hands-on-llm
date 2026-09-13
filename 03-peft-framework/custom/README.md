# custom · 二次开发示例（拿来即用）

本目录演示 LLaMA-Factory 三个最常用的扩展点，**不需要改框架源码**。源码层原理见 [`../notes/source_walkthrough.md`](../notes/source_walkthrough.md)。

## 1. 注册自己的数据集

| 文件 | 演示什么 |
| --- | --- |
| `dataset_info.example.json` | 三种数据的注册写法：单轮指令(alpaca)、多轮对话(sharegpt)、成对偏好(DPO) |
| `my_instruction.example.json` | 单轮指令-回答样例（SFT/ORPO 用） |
| `my_multiturn.example.json` | 多轮对话 messages 样例（SFT 用） |
| `my_preference.example.json` | chosen/rejected 成对样例（DPO/SimPO 用） |

接入步骤：

1. 把你的 json 数据放进 LLaMA-Factory 的 `data/` 目录；
2. 把 `dataset_info.example.json` 里对应条目合并进 `data/dataset_info.json`；
3. yaml 里写 `dataset: my_instruction`（用注册时的 key）。

> 自检：训练前先让框架加载一条样本，decode 出 `input_ids/labels`，确认只有 assistant 段参与 loss（原理见源码走读第 3 节）。

## 2. 自定义训练回调

`custom_callback.py` 提供两个可直接用的 `TrainerCallback`：

- `LossRecorderCallback`：把平滑 loss、学习率、耗时、显存逐步写 jsonl，方便自己画曲线；
- `EarlyStopOnPlateauCallback`：loss 长期不降时安全早停。

通过 `trainer.add_callback(...)` 或 `Trainer(callbacks=[...])` 接入，不动训练主循环。

## 3. 更深入的扩展（方向指引）

- 改损失：继承对应 stage 的 Trainer、重写 `compute_loss`（如自定义偏好损失）；
- 断点续训：yaml 设 `resume_from_checkpoint: saves/xxx/checkpoint-xxxx`；
- 合并 LoRA 导出独立模型：`llamafactory-cli export`，导出后即可用 vLLM 部署。