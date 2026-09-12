# MiniCode Agent — 终端 AI 编程智能体

> 📚 **本模块属于 [Hands-on-LLM 动手学大模型](../README.md) 学习库 · 返回总览看完整学习路线**

一个用 **Harness Engineering** 思路实现的终端 AI Coding Agent：在 ReAct（Thought→Action→Observation）循环之上，
构建了 Skill 能力路由、记忆沉淀与按需注入、分层上下文压缩、主从多智能体协作与分层安全审查，
以提升复杂编程任务下的执行准确率、上下文稳定性、推理效率与安全性。模型通过 OpenAI 兼容接口接入（默认 DeepSeek）。

## 架构与模块
| 模块 | 职责 |
| --- | --- |
| `mini_code_core.py` | Agent 内核：ReAct 循环、Harness（Tool Curfew、提交结果、无进展检测、终止条件）、工具注册与调度 |
| `tools.py` | 编程工具集（读写文件、grep、编辑、远程执行等）与远程 SSH 执行配置 |
| `skill_router.py` | Skill 能力体系：原子 Tool / 高层 Skill 分层，意图识别 + 关键词粗召回 + LLM 精排 |
| `memory.py` | 记忆系统：程序性/情景记忆的“执行-反思-提炼-存储-索引-按需复用”闭环 |
| `context_manager.py` | 分层上下文压缩：结构化摘要、大结果外置、占位压缩、超限兜底 |
| `multi_agent.py` | 主从多智能体：Supervisor 规划审批 + Worker 微型循环执行，异步并发与失败三级处理 |
| `security.py` | 分层安全过滤链：注入检测、操作边界、沙箱路径映射、分级确认、输出脱敏 |
| `cli.py` | 交互式命令行入口 |

## 快速开始
```bash
pip install -r requirements.txt
cp .env.example .env   # 填入你的 DEEPSEEK_API_KEY
python cli.py
```

## 目录
- 根目录为核心源码与测试（`test_*.py`）；
- `skills/`：高层 Skill 的提示模板（add-feature / fix-bug / refactor / code-review）；
- `docs/`：开发全流程记录、复刻方案与问题答疑。

## 安全说明
仓库**不包含任何真实 API Key / 服务器密码**，统一从环境变量读取（见 `.env.example`）。

## 测试
```bash
python -m pytest -q   # 或直接 python test_scenarios.py
```
