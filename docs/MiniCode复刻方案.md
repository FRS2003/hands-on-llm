# MiniCode 复刻方案

> 参考: LiuMengxuan04/MiniCode (TypeScript, 912⭐) + JiayuXu0/MiniCode (Go, 30天教程)

## 一、MiniCode 的架构长什么样

### 核心循环

```
用户输入 → System Prompt (含 Skills 索引 + 上下文)
         → LLM 决策 (调工具 or 回答)
         → 工具执行 (读文件/写代码/跑命令/搜代码)
         → 结果注入上下文
         → LLM 继续 or 终止
```

和你的 Agent 项目一模一样——ReAct 循环 + Function Calling + ToolRegistry。区别只是工具集。

### 模块拆解

| 模块 | MiniCode 实现 | 你的 Agent 项目等价物 |
|------|-------------|-------------------|
| **Agent Loop** | model→tool→model 循环，tool_choice="auto" | `HarnessAgent.run()` |
| **Tool System** | 16+ 内置工具，统一 JSON Schema | `ToolRegistry` + `tools.py` |
| **Skill 系统** | SKILL.md 文件，按目录分层，懒加载 | 类似多 Agent 的专用 Prompt，但更轻量 |
| **MCP 集成** | 动态加载外部工具服务器 | 没用过，新技能 |
| **上下文压缩** | auto-compact，token 计数，摘要替换 | Tool Curfew + 截断，更基础 |
| **权限系统** | 文件修改 review，危险命令确认 | 无（研究 Agent 不需要） |
| **Session 持久化** | 多项目 session 保存和恢复 | 无 |

### 工具集

| 工具 | 功能 |
|------|------|
| `read_file` | 读取文件内容 |
| `edit_file` / `patch_file` | 修改文件（unified diff） |
| `write_file` | 创建新文件 |
| `run_command` | 执行 bash 命令 |
| `glob` | 文件名模式匹配 |
| `grep` | 内容搜索 |
| `web_search` | 网页搜索 |
| `ask_user` | 向用户确认/提问 |
| `load_skill` | 懒加载 Skill 的完整 Prompt |

### Skill 路由

```
启动时: 扫描 4 个目录 → 只把 name+description 注入 System Prompt
运行时: Agent 判断需要某 Skill → 调 load_skill → 完整 Prompt 注入
优先级: 项目级 > 用户级 > 兼容 Claude Code 目录
覆盖: 同名 Skill，高优先级覆盖低优先级
```

---

## 二、用你的 Agent 项目复刻——7 天路线

### 不需要从零写——7 个模块里 4 个已经有了

✅ 已有 → 改配置  
🟡 有基础 → 扩展  
❌ 没有 → 新建

### Day 1-2：Agent Loop + Tool 集切换

**目标：** 把学术 Agent 变成 Coding Agent

**做法：**
- 复制 `agent_core_v4.py` → `minicode_core.py`
- System Prompt 从"研究文献"改成"编程助手"
- 工具集从检索/阅读/验证 → 文件操作/命令执行/代码搜索
- tools.py 替换为：`read_file`, `write_file`, `edit_file`, `run_command`, `grep`, `glob`

**验收：** 能说"读一下 app.py，找到所有路由定义" → Agent 调 read_file → 输出路由列表

工作量：✅ 已有基础，2h 改配置

### Day 3-4：Skill 系统

**目标：** 让 Agent 能根据任务类型加载专用 Prompt

**做法：**
- 建 `.mini-code/skills/` 目录放 SKILL.md
- 每个 Skill = 一个 Markdown 模板
- Skill 索引（name + description + triggers）注入 System Prompt
- Agent 需要时调 `load_skill(skill_name)` 获取完整 Prompt

**首批 Skill：**
```
skills/
├── code-review/SKILL.md     # 代码审查专用 Prompt
├── fix-bug/SKILL.md         # Bug 修复专用 Prompt
├── add-feature/SKILL.md     # 功能添加专用 Prompt
└── refactor/SKILL.md        # 重构专用 Prompt
```

**验收：** 说"review 一下 middleware.py" → 自动加载 code-review Skill → 用专用 Prompt 审查

工作量：🟡 Skill 目录结构和 load_skill 工具（4h）

### Day 5：上下文压缩 + 权限系统

**目标：** 长对话不炸，危险操作要确认

**做法：**
- 上下文压缩：工具输出 > 2000 字符 → 截断 + "调用 `expand_tool(n)` 查看完整输出"
- 权限系统：`run_command` 和 `edit_file` 调用前加一层确认

**验收：** 跑一个 grep 返回 100 行 → 只显示前 20 行 + "输出已截断"；改核心文件时弹确认

工作量：🟡 在现有 Tool Curfew 基础上扩展（3h）

### Day 6：TUI 终端界面

**目标：** 终端里有交互感，不是 print 日志

**做法：**
- 用 Python `rich` 或 `textual` 库做简单的终端 UI
- 显示：当前 Step、工具调用、输出预览、进度条
- 参考 MiniCode Go 版的 Bubble Tea 效果

**验收：** 终端里能看到 Agent 一步步执行的动画效果

工作量：❌ 新模块（4h）

### Day 7：完整测试 + 写文档

**目标：** 5 个真实 Coding 任务跑通，写出经验文档

**做法：**
- 设计 5 个测试任务（参考你的 benchmark 思路）
- 记录每个任务的步数、工具调用、Token 消耗
- 写面试用的技术文档

**验收：** 5 个任务全部完成，有量化数据

---

## 三、和参考项目的差异点（面试必答题）

| 维度 | MiniCode (参考) | 你的版本 | 面试可说的 |
|------|---------------|---------|----------|
| 语言 | TypeScript | Python | "基于 Agent 项目的 Python 架构复用" |
| Agent Loop | 自实现 | 复用 v4 Harness | "同一个 Harness 引擎，切换工具集就跨了领域" |
| 权限 | 内置 | 基于 Tool Curfew 扩展 | "权限不是黑盒——从 Agent 项目的 Harness Engineering 演化而来" |
| 记忆 | ? | 预搜索 + gathered_papers 的思路平移到任务记忆 | "跨会话经验累积，不是数据库而是闭环" |
| TUI | Bubble Tea | Rich/Textual | "终端交互更简洁直观" |

---

## 四、面试叙事线

"我做了一个 Coding Agent，参考 Claude Code 和 MiniCode 的架构，但核心引擎复用了之前 Agent 项目的 Harness Engineering 框架。"

"换的不是架构——同一个 Query Loop + Tool Use，只是工具集和 Prompt 换了。这证明了架构的通用性。"

"Skill 系统是我从 Agent 项目的 Query Router 延伸出来的——Router 只是判 RAG vs Agent，Skill 路由判的是 code-review vs fix-bug vs add-feature vs refactor，是多路分类器。"

"上下文压缩和权限系统是从 Tool Curfew 的思路演化来的——同样是代码层兜底，不信任模型会自觉。"
