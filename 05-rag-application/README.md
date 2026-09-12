# VSD Literature Assistant · 医学文献混合检索 RAG 与问答 Agent

> 📚 **本模块属于 [Hands-on-LLM 动手学大模型](../README.md) 学习库 · 返回总览看完整学习路线**

面向**先天性心脏病（VSD / CHD）医学文献**的智能检索与问答系统：从 PubMed / PMC 采集文献并构建本地知识库，
采用 **BM25 稀疏检索 + 稠密向量检索的双路混合召回、RRF 融合、重排（Rerank）与 HyDE / 查询扩展**，
结合大模型（DeepSeek，OpenAI 兼容接口）生成**带文献引用**的回答；并实现一个 **ReAct Agent**，
通过“思考–调用检索工具–观察”的多轮循环完成复杂文献问题的检索式推理。前端使用 Streamlit。

## 处理流水线
数据采集与清洗（`extract_pmc` / `expand_papers` / `fix_abstracts` / `llm_summarize`）
→ 双路索引构建（`build_dual_index` / `rebuild_*`，BM25 + 向量，FAISS）
→ 混合召回与融合（`merge_hybrid`，RRF）
→ 查询改写（HyDE / 查询扩展）与重排（reranker）
→ 带引文的 LLM 生成
→ ReAct Agent 编排（`agent/`）
→ Streamlit 应用（`app/`）。

## 目录
| 目录 | 内容 |
| --- | --- |
| `app/` | Streamlit 前端：`app.py`（部署版）、`app_v2.py`（迭代版），API Key 运行时输入 |
| `agent/` | ReAct 智能体内核 `agent_core.py` / `agent_core_v2.py`、工具集 `tools.py`、入口 `run_agent.py` 与测试 |
| `retrieval/` | 文献采集清洗、双路索引构建、混合融合、查询扩展与摘要等数据/索引脚本 |
| `evaluation/` | 自建问题集上的召回评测：Recall@K、重排前后对比、HyDE / 查询扩展消融实验 |
| `utils/` | 语言、摘要、GPU 与中间结果检查的小工具 |
| `.devcontainer/` | 容器化开发/部署配置 |

## 方法与评测要点
- **混合检索**：BM25（关键词）与稠密向量（语义）互补，用 RRF 融合缓解单一召回的漏检；
- **重排与查询改写**：对比 reranker、HyDE、查询扩展（QE）对 Recall@20 的增益，保留消融脚本；
- **可复现实验**：`evaluation/` 内每个 `benchmark_*` / `test_*` 对应一次检索策略迭代，可用自有问题集复跑。

## 快速开始
```bash
pip install -r requirements.txt
cp .env.example .env        # 填入 DEEPSEEK_API_KEY（或在界面运行时输入）
# 1) 用 retrieval/ 下脚本从自有文献构建索引到 data/vsd_rag_db/（私有文献数据不随仓库提供）
# 2) 启动应用
streamlit run app/app.py
```

## 说明
- 仓库**不含任何真实 API Key**：代码统一从环境变量 `DEEPSEEK_API_KEY` 读取，前端也支持运行时输入。
- 私有医学文献语料与 FAISS 索引（`data/`、`*.index`、`*.pkl`）受版权与隐私限制不随仓库分发，需按 `retrieval/` 流程自行构建。
