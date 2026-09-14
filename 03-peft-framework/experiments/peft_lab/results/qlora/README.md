---
library_name: peft
license: other
base_model: /root/autodl-tmp/models/ms_cache/Qwen/Qwen2___5-0___5B
tags:
- llama-factory
- lora
- generated_from_trainer
model-index:
- name: qlora
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# qlora

This model is a fine-tuned version of [/root/autodl-tmp/models/ms_cache/Qwen/Qwen2___5-0___5B](https://huggingface.co//root/autodl-tmp/models/ms_cache/Qwen/Qwen2___5-0___5B) on the zh_instruct_3k dataset.
It achieves the following results on the evaluation set:
- Loss: 2.0826

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0001
- train_batch_size: 2
- eval_batch_size: 1
- seed: 42
- gradient_accumulation_steps: 8
- total_train_batch_size: 16
- optimizer: Use adamw_torch with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_ratio: 0.05
- num_epochs: 1.0

### Training results

| Training Loss | Epoch | Step | Validation Loss |
|:-------------:|:-----:|:----:|:---------------:|
| 1.7932        | 1.0   | 188  | 2.0826          |


### Framework versions

- PEFT 0.15.2
- Transformers 4.52.4
- Pytorch 2.6.0+cu124
- Datasets 3.6.0
- Tokenizers 0.21.1