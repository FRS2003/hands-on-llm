# from_scratch（手写组件）

本目录放**不看官方实现、自己独立写出**的模块，是体现“从零实现”的核心：
- `tokenizer_bpe.py`：BPE 分词器
- `model.py`：RMSNorm / RoPE / GQA / SwiGLU / Attention / DecoderBlock
- `kernel_rmsnorm.py`：Triton 版 RMSNorm
- `flash_attention_tile.py`：分块 online-softmax

要求：每个文件标注关键张量维度；写完后再与官方实现对照并记录差异。
