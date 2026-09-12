# configs（实验配置）

每个对照实验一份 yaml，命名体现变量，例如：
- `sft_qwen0.5_lora_r8.yaml`
- `dpo_qwen0.5.yaml` / `kto_qwen0.5.yaml` / `orpo_qwen0.5.yaml`
- `ds_zewo2.yaml`（DeepSpeed ZeRO-2）
对照原则：除被研究的变量外，其余超参/数据/步数全部一致。
