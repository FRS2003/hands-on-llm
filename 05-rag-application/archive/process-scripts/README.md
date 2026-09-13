# 过程脚本归档（开发迭代留痕，主线学习无需阅读）

这里保存项目迭代过程中产生的**一次性 / 被替代 / 互相重复**的脚本，仅用于保留真实开发痕迹与可追溯性。它们不是最终方案，**学习和复现请以模块根目录下的主线脚本为准**，本目录不参与主流程。

## 为什么归档而不是删除

RAG 系统是经过多轮试错才收敛的：多个 `benchmark_*` 是同一评测的早期拷贝、`test_reranker*` 是重排器的反复试验、`rebuild_*` / `build_index_200` 是索引方案的旧版本、`check_*` / `debug_*` 是排障用的临时检查。保留它们可以看到"结论是怎么一步步试出来的"，但放在主线目录会干扰阅读，因此统一收纳于此。

## 目录内容

| 子目录 | 内容 | 主线对应物（请看这些） |
| --- | --- | --- |
| `evaluation/` | 早期 / 重复的评测脚本（15 个） | `../../evaluation/benchmark_hybrid.py`（三路对比）、`benchmark_qe.py`、`test_qe_split.py`、`test_hyde.py`、`analyze_misses.py` |
| `retrieval/` | 旧版索引与一次性数据处理（6 个） | `../../retrieval/build_dual_index.py`（双路索引最终版）、`rebuild_m3.py`、`extract_pmc.py`、`expand_papers.py`、`fix_abstracts.py`、`llm_summarize.py` |
| `utils/` | 排障期的 check/debug 小工具（9 个） | 无需替代，问题定位后即废弃 |
| `app_v2.py` | Streamlit 前端的迭代版本 | `../../app/app.py`（部署版） |

## 主线脚本一览（回到正轨用）

- 建库：`retrieval/` 下 `extract_pmc → expand_papers → fix_abstracts → llm_summarize → build_dual_index`
- 评测：`evaluation/benchmark_hybrid.py` 为总览，其余为单项消融，真实历史结果见 `evaluation/results/`
- 智能体：`agent/agent_core.py`（清单式 ReAct）→ `agent_core_v2.py`（自主规划 / 反思版）的演进