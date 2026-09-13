# LLaMA-Factory 源码走读：一份 yaml 是怎么跑起来的

> **所属模块**：第 ③ 模块「工业框架」，配合 [`../configs`](../configs) 的 yaml 模板与 [`alignment.md`](alignment.md)（对齐算法选型）阅读
> **前置知识**：会用 PyTorch 训练、知道 HuggingFace Transformers 的 `from_pretrained` / `Trainer` 基本用法
> **文档定位**：不逐行讲源码，而是给一张**走读地图**——跟着"一条训练命令的生命周期"，看清配置解析、模型加载、数据处理、训练器四条主链路分别在哪些文件、该重点看什么。读完你能自己定位任何一个 yaml 参数最终在哪生效。

---

## 📖 写在前面：为什么要读框架源码

直接抄命令也能把 LoRA 跑起来，但一旦遇到"loss 不下降""显存爆了""多轮对话标签错位""想加个自定义指标"，只会命令就寸步难行。LLaMA-Factory 本质是对 Transformers + PEFT + DeepSpeed 的**一层结构化封装**，读懂它的主链路，你会同时搞懂三件事：

1. 工业训练框架如何把"配置 → 模型 → 数据 → 训练循环"解耦；
2. LoRA / 量化 / DPO 这些能力到底挂在哪一步；
3. 想二次开发（自定义数据集、回调、Trainer）该改哪里。

> 版本说明：LLaMA-Factory 迭代很快，类名/文件可能微调，但**四条主链路的分层长期稳定**。下文以 `src/llamafactory/` 下的结构为线索，建议对照你本地 clone 的版本阅读，路径对不上时按"职责关键词"搜索即可。

---

## 0. 一张总图：命令的生命周期

```
llamafactory-cli train config.yaml
        │
        ▼
[1] 配置解析  hparams/      yaml → 多个 dataclass（Model/Data/Finetune/Eval Arguments）
        │
        ▼
[2] 模型加载  model/        tokenizer → 基座模型 → 量化 → 挂 LoRA adapter
        │
        ▼
[3] 数据处理  data/         dataset_info 注册 → 标准化 → 模板拼接 → label mask → collator
        │
        ▼
[4] 训练器    train/<stage>/  SFT/DPO/PPO Trainer + workflow + callbacks
        │
        ▼
     checkpoint / 日志 / 导出
```

记住这张图，下面逐段拆。

---

## 1. 配置解析层 `hparams/`：yaml 到 dataclass

### 1.1 入口怎么找到这层

`llamafactory-cli train` 最终调用 `train/tuner.py` 里的主函数，第一件事就是 `read_args()`——它位于 `hparams/parser.py`，用 HuggingFace 的 `HfArgumentParser` 把一份扁平 yaml **拆成几组 dataclass**：

| dataclass | 文件 | 管什么（对应 yaml 字段） |
| --- | --- | --- |
| `ModelArguments` | `hparams/model_args.py` | `model_name_or_path`、`quantization_bit`、`adapter_name_or_path` |
| `DataArguments` | `hparams/data_args.py` | `dataset`、`template`、`cutoff_len`、`mask_history` |
| `FinetuningArguments` | `hparams/finetuning_args.py` | `stage`、`finetuning_type`、`lora_rank`、`lora_target`、DPO 的 `beta` |
| `TrainingArguments` | 继承 transformers | `num_train_epochs`、`learning_rate`、`per_device_train_batch_size`、`bf16`、`deepspeed` |

### 1.2 读码重点

- **一个 yaml 字段去哪找**：先判断它属于模型/数据/微调/训练哪一类，再去对应 dataclass 看字段定义、默认值和 `__post_init__` 里的合法性校验（比如 `stage=ppo` 时会强制要求某些字段）。
- **`finetuning_type` 是总开关**：`full / freeze / lora` 在这里被解析，后续模型加载层据此决定挂不挂 adapter。
- 这一层只做"解析与校验"，不碰真正的张量——理解"声明式配置"如何变成"强类型对象"，是读所有工业框架的通用范式。

---

## 2. 模型加载层 `model/`：从基座到可训练模型

### 2.1 加载顺序（关键）

`model/loader.py` 的 `load_model` / `load_tokenizer` 严格按下面顺序走，顺序错了显存和结果都会出问题：

```
load_tokenizer              先建分词器（决定 special tokens、padding 侧）
        │
from_pretrained(基座)       载入原始权重（torch_dtype / device_map）
        │
量化（可选）   model/quantization.py  → BitsAndBytesConfig(4bit) 或 GPTQ/AWQ
        │
patch_model    model/patcher.py      → 给不同模型架构打注意力/梯度检查点补丁
        │
init_adapter   model/adapter.py      → 用 peft.LoraConfig 挂 LoRA、冻结其余参数
```

### 2.2 三个最值得读的点

1. **QLoRA = 量化基座 + LoRA adapter 的组合**：`quantization.py` 先把基座按 4bit 量化载入（NF4 + 双重量化），`adapter.py` 再在冻结的量化层上插 LoRA。读懂这里就明白"为什么 QLoRA 省显存却仍能训练"——被训练的只有 LoRA 小矩阵，且以 fp16/bf16 计算。
2. **`lora_target` 怎么匹配层**：`adapter.py` 会在模型的**所有线性层名字**里找匹配 `q_proj/k_proj/v_proj/o_proj/...` 的模块注入 LoRA。想知道 LoRA 到底加在了哪些层，在这里打断点打印 `model.named_modules()` 最直接。
3. **`patcher.py` 是"兼容性魔法"集中地**：不同模型（LLaMA/Qwen/ChatGLM…）的注意力实现、rope scaling、梯度检查点写法不一，框架在这里统一打补丁。读它能学到"一个框架如何优雅适配几十种架构"。

### 2.3 和本学习库 ② 的呼应

② 模块是**手写** RMSNorm/RoPE/GQA，这里是**看工业框架如何加载并改造现成模型**：一个理解内部公式，一个理解工程装配，两者对照着看收获最大。
---

## 3. 数据处理层 `data/`：最容易出 bug 的一层

这层负责把原始 json 变成模型能吃的 `input_ids / labels`，是多轮对话和偏好学习最容易出错的地方，也最值得细读。

### 3.1 流水线

```
dataset_info.json 注册      data/parser.py     声明数据文件名、格式(alpaca/sharegpt)、列名
        │
原始样本标准化               data/converter.py  统一成内部 {"messages"/"instruction",...} 结构
        │
对话模板拼接                 data/template.py   套 system/user/assistant 模板、加 special token
        │
tokenize + label mask       data/processor.py  编码；prompt 段 label=-100，只留 response
        │
collator 组 batch            data/collator.py   padding；DPO 还要拼 chosen/rejected 成对
```

### 3.2 三个必须看懂的机制

1. **`dataset_info.json` 是注册表**：你的自定义数据必须先在这里登记（文件名、格式类型、列字段），yaml 里的 `dataset: 你的名字` 才能找到它。这是"配置即注册"的设计，自定义数据示例见 [`../custom`](../custom)。
2. **label mask 与 ② 的 SFT 完全对应**：`processor.py` 把用户指令/历史轮次的 label 置为 `-100`（PyTorch 交叉熵默认忽略 -100），只对 assistant 回复算 loss——这正是 ② `alignment.md` 里"只对 response 算 loss"的工业实现。
3. **DPO 的成对 collator**：偏好数据一条样本含 chosen 和 rejected 两段，`PairwiseDataCollator` 会把它们编码成成对张量喂给 DPOTrainer。看懂它就理解了为什么偏好数据格式和 SFT 不一样。

### 3.3 调试技巧

数据层问题不要靠猜：拿一条样本，在 collator 之后打印 `input_ids` 和 `labels`，用 `tokenizer.decode` 还原，肉眼确认"哪些位置 label 不是 -100、拼出来的对话对不对"。这是定位多轮/偏好数据问题最快的方法。

---

## 4. 训练器层 `train/`：stage 如何分派

### 4.1 目录即 stage

`train/` 下按训练阶段分子目录，每个子目录一般有 `workflow.py`（组装流程）和 `trainer.py`（自定义 Trainer）：

| 子目录 | stage | 核心 Trainer | 关键不同 |
| --- | --- | --- | --- |
| `train/sft/` | sft | `CustomSeq2SeqTrainer` | 标准语言建模 / 对 response 算 loss |
| `train/dpo/` | dpo | `CustomDPOTrainer` | 成对前向，算 chosen/rejected 的对数似然差（见 ② DPO 推导） |
| `train/ppo/` | ppo | 价值头 + reward model | 在线 rollout，四模型协作 |
| `train/kto/` | kto | KTOTrainer | 只需"好/坏"二元标签，不需成对 |

### 4.2 自定义 Trainer 都改了什么

它们大多继承 Transformers 的 `Trainer` / TRL 的对应 Trainer，重写的通常是：

- `compute_loss`：换成该算法的损失（DPO/KTO 各有公式）；
- `training_step` / `compute_rewards`：PPO 这类需要在线生成的逻辑；
- 并通过 `callbacks/` 挂日志、loss 平滑、显存清理、断点续训等横切逻辑。

### 4.3 回调机制 `train/callbacks/`

Transformers 的 `TrainerCallback` 在 `on_epoch_begin / on_log / on_evaluate / on_save` 等时机被调用。框架自带日志与可视化回调；你想加"实时画 loss 曲线""指标上报""早停"，都是写一个 callback 传进去，**不用改训练主循环**——示例见 [`../custom`](../custom)。

---

## 5. 二次开发的三个标准切入点

| 需求 | 改哪里 | 是否要动框架源码 |
| --- | --- | --- |
| 加自己的数据集 | 写 json + 在 `dataset_info.json` 注册 | 否（纯外部配置） |
| 加训练回调（日志/早停/画图） | 自定义 `TrainerCallback`，yaml/启动处传入 | 否 |
| 改损失或训练步 | 继承对应 stage 的 Trainer 重写 `compute_loss` | 通常子类化即可 |

这也是 LLaMA-Factory 好上手二次开发的原因：**绝大多数扩展点都被设计成"外部注册 / 子类化"，不需要改框架本体**。

---

## 6. 建议的读码顺序（配合 ② 一起）

1. 先跑通 [`../configs`](../configs) 里的 `sft_lora_qwen0.5.yaml`，有一个能跑的实例；
2. 从 `train/tuner.py` 的主函数入口，顺着第 0 节总图走一遍，**只看调用关系、不抠实现**；
3. 精读 `data/processor.py` 的 label mask（对照 ② SFT）；
4. 精读 `model/adapter.py` 的 LoRA 挂载（对照 ② LoRA 与本模块 PEFT 选型笔记）；
5. 对比读 `train/sft/trainer.py` 与 `train/dpo/trainer.py` 的 `compute_loss`（对照 ② DPO 闭式解）；
6. 最后用 [`../custom`](../custom) 的示例做一次"加数据集 + 加回调"的小改造，读完动手才算真正掌握。

> 一句话：**配置层决定"做什么"，模型层决定"拿什么模型做"，数据层决定"喂什么"，训练器层决定"怎么更新"**——四层解耦，正是这套框架能同时支持几十种模型和算法的根本原因。