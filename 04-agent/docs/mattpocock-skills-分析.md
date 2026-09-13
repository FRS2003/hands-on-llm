# Matt Pocock Skills — 功能总结

> 来源: [github.com/mattpocock/skills](https://github.com/mattpocock/skills)  
> Stars: 103k+ | 安装: `npx skills@latest add mattpocock/skills`

## 核心理念

**"Real Engineering, not Vibe Coding."** 把软件工程经典实践（TDD、DDD、XP）封装成 Claude Code 的 Agent Skill，人控制流程，AI 执行步骤。

## 四类 Skill（对应 AI 编程的 4 大失败模式）

### 1. 需求对齐类（Agent 没做你想要的）

| Skill | 功能 |
|-------|------|
| `/grill-me` | AI 反复盘问你的计划，追问所有分支决策直到完全对齐 |
| `/grill-with-docs` | 同上 + 自动维护 CONTEXT.md 和架构决策记录 |
| `/domain-modeling` | 用 DDD 统一语言建模，消除歧义 |

### 2. 规划生成类（把思考变成可执行任务）

| Skill | 功能 |
|-------|------|
| `/to-prd` | 把对话上下文合成为 PRD，发布到 issue tracker |
| `/to-issues` | 按"竖切片"拆成独立 issue，每人能独立拿走一个 |
| `/triage` | Issue 状态机管理（needs-triage → ready → done） |
| `/implement` | 基于 PRD/issue 实现，集成 TDD + 类型检查 + 代码审查 |

### 3. 质量保证类（代码不工作）

| Skill | 功能 |
|-------|------|
| `/tdd` | 严格红→绿→重构循环，先写失败测试再写最小实现 |
| `/diagnosing-bugs` | 系统化排错：复现→最小化→假设→插桩→修复→回归测试 |
| `/review` | 代码审查 |
| `/prototype` | 快速创建可丢弃的原型验证设计 |

### 4. 架构治理类（代码库变淤泥球）

| Skill | 功能 |
|-------|------|
| `/improve-codebase-architecture` | 扫描代码库找"深模块"机会，生成 HTML 报告 |
| `/codebase-design` | 架构设计词汇表和原则（模块/接口/深度/接缝等） |
| `/resolving-merge-conflicts` | 解决合并冲突 |

### 5. 效率工具类

| Skill | 功能 |
|-------|------|
| `/handoff` | 压缩对话让另一个 Agent 无缝接续 |
| `/teach` | 让 AI 教你一个概念 |
| `/writing-great-skills` | 帮你写新的自定义 Skill |
| `/git-guardrails-claude-code` | 拦截危险 git 操作 |

## 对 MiniCode 的设计启示

1. **Agent 开发也可以用 Skill 模式**：把检索 / 阅读 / 验证 / 对比封装成独立 Skill，每个有专用 Prompt，类似多 Agent 但更轻量
2. **TDD 思维可用于 Agent 评测**：先写预期行为（expected_papers），再跑 Agent 验证
3. **Handoff 模式**：Agent 做研究 + Handoff 给另一个 Agent 写报告，拆开上下文
4. **grill-me 思路**：让 Agent 在回答前自我盘问"找够证据了吗？有矛盾吗？"，不依赖 Prompt，用 Skill 强制
