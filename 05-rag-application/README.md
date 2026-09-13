# VSD Literature Assistant · 医学文献混合检索 RAG 与问答 Agent

> 📚 **本模块属于 [Hands-on-LLM 动手学大模型](../README.md) 学习库 · [返回总览看完整学习路线](../README.md)**

一个面向**先天性心脏病（VSD / CHD）医学文献**的检索增强问答系统，完整覆盖"数据采集 → 双路索引 → 混合召回 → 融合 → 重排 → 查询改写 → 带引用生成 → ReAct Agent"的 RAG 全链路，前端使用 Streamlit，大模型通过 OpenAI 兼容接口接入（默认 DeepSeek）。

> **新手从这里开始**：先读 [`docs/RAG入门导读.md`](docs/RAG入门导读.md)（[Word 版](docs/RAG入门导读.docx)）建立全局脉络，再用 [`docs/RAG技术详解.md`](docs/RAG技术详解.md)（[Word 版](docs/RAG技术详解.docx)）深入原理，最后用 [`docs/RAG检索自测问答.md`](docs/RAG检索自测问答.md)（[Word 版](docs/RAG检索自测问答.docx)）自测。

## 检索流水线（先建立整体印象）

```text
文献采集与清洗                双路索引                 混合召回与融合            查询改写与重排            生成 / 智能体
extract_pmc            build_dual_index      BM25（关键词, 手写）      HyDE / QE 查询扩展     带 [n] 引用的 LLM 回答
expand_papers    →     rebuild_m3       +    稠密向量(BGE-M3/FAISS) → 分数归一化加权融合 → CrossEncoder 重排 →  ReAct Agent(agent/)
fix_abstracts          （粗索引 A + 细索引 B）   0.6×向量 + 0.4×BM25       粗排 Top20→精排 Top5    Streamlit(app/)
llm_summarize
```

> **融合方式澄清**：本项目双路结果采用**分数归一化后的加权融合（0.6 向量 + 0.4 BM25）**，并**未使用 RRF（倒数排名融合）**；二者对比见 [`docs/RAG入门引导.md`](docs/RAG入门导读.md) 第 4 节与技术详解第 7.2 节。

## 目录说明

| 目录 / 文件 | 内容 |
| --- | --- |
| `docs/` | 三件套教学文档：入门导读、技术详解、检索自测问答（均附 Word） |
| `retrieval/` | 文献采集清洗与双路索引构建的**主线脚本**（建库按文件名顺序执行） |
| `evaluation/` | 可复跑的评测与消融：`benchmark_hybrid.py`（三路总览）、`benchmark_qe.py`、`test_qe_split.py`、`test_hyde.py`、`analyze_misses.py` |
| `evaluation/results/` | **真实历史评测日志留档**（21 题 / 58 题），附口径说明，见该目录 README |
| `agent/` | ReAct 智能体：`agent_core.py`（清单式 ReAct）→ `agent_core_v2.py`（自主规划 / 反思）的两版演进，含工具集与入口 |
| `app/` | Streamlit 前端 `app.py`，API Key 运行时输入 |
| `archive/process-scripts/` | 开发迭代中的一次性 / 重复 / 被替代脚本（32 个），仅留痕，主线无需阅读 |
| `.devcontainer/` | 容器化开发 / 部署配置 |
## 实测结果（均标注口径，原始日志见 `evaluation/results/`）

| 实验 | 题集 / 口径 | 结果 |
| --- | --- | --- |
| 混合检索 Top-5 召回 | 21 题、L1 难度 | Recall 16/21 = **76.2%** |
| 粗排 → 加 Reranker | 20 题原型阶段、Recall@5 | FAISS 粗排约 65% → 粗排 Top20 + CrossEncoder 精排 Top5 达 85% |
| 无 Reranker 权重网格 | 58 题、6 档权重 | 向量 0.9/BM25 0.1 最优 68.4%，区间仅 59.6%–68.4%（权重不敏感） |
| 单轮 RAG vs ReAct Agent | 58 题（L1×26/L2×26/L3×6） | 完全命中 RAG 58/58=100% vs Agent 50/58=86%；平均 Recall 100% vs 94.5%；Agent 平均约 9 步 |

> 这些数字来自不同阶段、不同题集，**不能直接横向比较**，详细口径见 `evaluation/results/README.md`。结论：小规模知识库下数据质量与 Reranker 是主要杠杆，单主题问题单轮 RAG 已足够，跨主题复杂问题才需要 Agent。

## Agent 两版演进（为什么保留两个版本）

- `agent_core.py`（v1）：**清单式 ReAct**，按预设步骤调用检索工具，结构透明、适合理解循环骨架；
- `agent_core_v2.py`（v2）：引入**自主规划与反思**，由模型决定下一步检索 / 阅读 / 终止，更接近真实 Agent，但也更需要 Harness 约束。

对照阅读两版，能直观看到"从写死流程到自主决策"的取舍。Agent 的终止控制、两步答案合成等工程问题，见第 ④ 模块 [`04-agent/docs/问题答疑.md`](../04-agent/docs/问题答疑.md) 与 [`Agent架构自测问答.md`](../04-agent/docs/Agent架构自测问答.md)。

## 快速开始

```bash
pip install -r requirements.txt
cp .env.example .env        # 填入 DEEPSEEK_API_KEY（也可在界面运行时输入）

# 1) 用 retrieval/ 下脚本从自有文献构建 BM25 + 向量索引
#    extract_pmc → expand_papers → fix_abstracts → llm_summarize → build_dual_index
# 2) 跑评测（可先用 benchmark_hybrid.py 看三路对比）
python evaluation/benchmark_hybrid.py
# 3) 启动前端
streamlit run app/app.py
```

## 数据与安全说明

- 仓库**不含任何真实 API Key**：代码统一从环境变量 `DEEPSEEK_API_KEY` 读取，前端也支持运行时输入；
- 私有医学文献语料与 FAISS 索引（`data/`、`*.index`、`*.pkl`）受版权与隐私限制**不随仓库分发**，需按 `retrieval/` 流程用自有文献构建；`.gitignore` 已排除这些产物；
- 想在没有私有语料时理解流程，可直接阅读 `docs/` 与 `evaluation/results/` 的真实日志，不影响学习主线。