# Learn Claude Code 项目总结

本文档对 `docs/learn-claude-code` 文档目录和 `src/learn-claude-code/agents` 代码目录进行系统性总结。

---

## 一、核心理念：模型即 Agent

### 1.1 Agent 的本质定义

本项目的核心观点：**Agent 是模型，不是框架，不是提示词链，不是拖拽式工作流。**

Agent 是一个神经网络（Transformer、RNN 等），通过数十亿次梯度更新，在行动序列数据上学会了感知环境、推理目标、采取行动。从 DeepMind DQN 玩 Atari（2013）到 OpenAI Five 征服 Dota 2（2019），再到腾讯绝悟统治王者荣耀（2019），每一个里程碑都证明：Agent 永远是模型本身。

### 1.2 Harness 工程师的使命

当我们在"开发 Agent"时，实际上是在**构建 Harness**——为模型提供可操作的环境：

```
Harness = Tools + Knowledge + Observation + Action Interfaces + Permissions

    Tools:          文件读写、Shell、网络、数据库、浏览器
    Knowledge:      产品文档、领域资料、API 规范、风格指南
    Observation:    git diff、错误日志、浏览器状态、传感器数据
    Action:         CLI 命令、API 调用、UI 交互
    Permissions:    沙箱隔离、审批流程、信任边界
```

**模型做决策，Harness 执行。模型做推理，Harness 提供上下文。模型是驾驶者，Harness 是载具。**

### 1.3 Claude Code 的教学价值

Claude Code 是最优雅、最完整的 agent harness 实现：

```
Claude Code = 一个 agent loop
            + 工具 (bash, read, write, edit, glob, grep, browser...)
            + 按需 skill 加载
            + 上下文压缩
            + 子 agent 派生
            + 带依赖图的任务系统
            + 异步邮箱的团队协调
            + worktree 隔离的并行执行
            + 权限治理
```

它展示了当你信任模型、把工程精力集中在 harness 上时会发生什么。

---

## 二、12 递进式课程详解

### 第一阶段：循环基础

#### **s01: Agent Loop（Agent 循环）**

> *"One loop & Bash is all you need"* — 一个工具 + 一个循环 = 一个 Agent

**核心模式**：一个退出条件控制的 while 循环，持续运行直到模型不再调用工具。

```python
def agent_loop(messages):
    while True:
        response = client.messages.create(model=MODEL, messages=messages, tools=TOOLS)
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            return
        # 执行工具，收集结果，追加到 messages
```

**关键洞察**：不到 30 行代码就是整个 Agent。后面 11 个章节都在这个循环上叠加机制——循环本身始终不变。

---

#### **s02: Tool Use（工具使用）**

> *"加一个工具，只加一个 handler"* — 循环不用动，新工具注册进 dispatch map

**核心模式**：使用字典映射工具名到处理函数，实现工具分发。

```python
TOOL_HANDLERS = {
    "bash":       lambda **kw: run_bash(kw["command"]),
    "read_file":  lambda **kw: run_read(kw["path"]),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file":  lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
}
```

**新增组件**：
- `safe_path()` 路径沙箱防止逃逸工作区
- 专用工具替代全部走 shell 的方式
- 加工具 = 加 handler + 加 schema，循环永远不变

---

### 第二阶段：规划与知识

#### **s03: TodoWrite（待办写入）**

> *"没有计划的 agent 走哪算哪"* — 先列步骤再动手，完成率翻倍

**核心机制**：
- `TodoManager` 存储带状态的项目，同一时间只允许一个 `in_progress`
- "同时只能有一个 in_progress" 强制顺序聚焦
- **Nag Reminder**：模型连续 3 轮不调用 `todo` 时注入提醒，制造问责压力

```python
if rounds_since_todo >= 3 and messages:
    last["content"].insert(0, {"type": "text", "text": "<reminder>Update your todos.</reminder>"})
```

---

#### **s04: Subagents（Subagent）**

> *"大任务拆小，每个个小任务干净的上下文"* — Subagent 用独立 messages[]，不污染主对话

**核心机制**：
- 父 Agent 有 `task` 工具派发子任务
- Subagent 以 `messages=[]` 启动，拥有除 `task` 外的所有基础工具（禁止递归生成）
- Subagent 运行自己的循环，只有最终文本返回给父 Agent
- 整个消息历史直接丢弃，父 Agent 只收到摘要文本

**解决的问题**：Agent 工作越久，messages 数组越臃肿。Subagent 可能跑了 30+ 次工具调用，但父 Agent 只需要一个词："pytest"。

---

#### **s05: Skill Loading（Skill 加载）**

> *"用到什么知识，临时加载什么知识"* — 通过 tool_result 注入，不塞 system prompt

**两层加载策略**：
- **第一层（系统提示）**：放 Skill 名称（低成本，约 100 tokens/skill）
- **第二层（tool_result）**：按需放完整内容（约 2000 tokens）

```python
# 每个 Skill 是一个目录，包含 SKILL.md 文件和 YAML frontmatter
skills/
  pdf/SKILL.md       # ---\n name: pdf\n description: Process PDF files\n ---\n ...
  code-review/SKILL.md
```

---

#### **s06: Context Compact（上下文压缩）**

> *"上下文总会满，要有办法腾地方"* — 三层压缩策略，换来无限会话

**三层压缩机制**：

| 层级 | 名称 | 触发条件 | 动作 |
|------|------|----------|------|
| 1 | micro_compact | 每轮静默执行 | 旧 tool result (>3轮) → 占位符 |
| 2 | auto_compact | token > 50000 | 保存 transcript 到磁盘，LLM 摘要 |
| 3 | manual compact | 模型调用 compact 工具 | 同 auto_compact |

**关键洞察**：完整历史通过 transcript 保存在磁盘上。信息没有真正丢失，只是移出了活跃上下文。

---

### 第三阶段：持久化

#### **s07: Task System（任务系统）**

> *"大目标要拆成小任务，排好序，记在磁盘上"* — 文件持久化的任务图

**核心机制**：把扁平清单升级为持久化到磁盘的**任务图（DAG）**。

每个任务是一个 JSON 文件，有状态、前置依赖 (`blockedBy`)：

```
.tasks/
  task_1.json  {"id":1, "status":"completed"}
  task_2.json  {"id":2, "blockedBy":[1], "status":"pending"}
  task_3.json  {"id":3, "blockedBy":[1], "status":"pending"}
  task_4.json  {"id":4, "blockedBy":[2,3], "status":"pending"}
```

**任务图随时回答三个问题**：
- 什么可以做？—— pending + blockedBy 为空
- 什么被卡住？—— 等待前置任务完成
- 什么做完了？—— completed，自动解锁后续任务

---

#### **s08: Background Tasks（后台任务）**

> *"慢操作丢后台，agent 继续想下一步"* — 后台线程跑命令，完成后注入通知

**核心机制**：
- `BackgroundManager` 用线程安全的通知队列追踪任务
- 子进程完成后，结果进入通知队列
- 每次 LLM 调用前排空通知队列

**解决的问题**：`npm install`、`pytest`、`docker build` 等慢命令阻塞式循环下模型只能干等。

---

### 第四阶段：团队协作

#### **s09: Agent Teams（Agent 团队）**

> *"任务太大一个人干不完，要能分给队友"* — 持久化队友 + JSONL 邮箱

**三大要素**：
1. 能跨多轮对话存活的持久 Agent
2. 身份和生命周期管理
3. Agent 之间的通信通道

**通信机制**：JSONL 收件箱（append-only，drain-on-read）

```
.team/
  config.json           <- 团队名册 + 状态
  inbox/
    alice.jsonl         <- append-only
    bob.jsonl
    lead.jsonl
```

---

#### **s10: Team Protocols（团队协议）**

> *"队友之间要有统一的沟通规矩"* — 一个 request-response 模式驱动所有协商

**两大协议**：

1. **关机协议**：领导请求 → 队友批准（收尾退出）或拒绝（继续干）
2. **计划审批协议**：队友提交 → 队友审查 → 审批或拒绝

**共享 FSM**：`pending → approved | rejected`

每个请求一个 `request_id`，响应引用同一 ID 关联。

---

#### **s11: Autonomous Agents（自治 Agent）**

> *"队友自己看看板，有活就认领"* — 不需要领导逐个分配，自组织

**队友生命周期**：

```
spawn -> WORKING -> IDLE -> WORKING -> ... -> SHUTDOWN

IDLE 阶段：
  +---> check inbox --> message? --> WORK
  +---> scan .tasks/ --> unclaimed? --> claim -> WORK
  +---> 60s timeout --> SHUTDOWN
```

**身份重注入**：Context Compact 后 Agent 可能忘了自己是谁，通过在 messages 开头插入身份块解决。

---

#### **s12: Worktree + Task Isolation（Worktree 任务隔离）**

> *"各干各的目录，互不干扰"* — 任务管目标，worktree 管目录，按 ID 绑定

**双平面架构**：

```
控制平面 (.tasks/)              执行平面 (.worktrees/)
task_1.json                     auth-refactor/
  status: in_progress <->       branch: wt/auth-refactor
  worktree: "auth-refactor"     task_id: 1

事件流：events.jsonl（生命周期日志）
```

**解决的问题**：两个 Agent 同时重构不同模块，未提交的改动互相污染，谁也没法干净回滚。

---

## 三、代码实现结构

### 3.1 文件对应关系

| 文档 | 代码文件 | 核心组件 |
|------|----------|----------|
| s01 | `s01_agent_loop.py` | agent_loop, run_bash |
| s02 | `s02_tool_use.py` | TOOL_HANDLERS, safe_path |
| s03 | `s03_todo_write.py` | TodoManager |
| s04 | `s04_subagent.py` | run_subagent |
| s05 | `s05_skill_loading.py` | SkillLoader |
| s06 | `s06_context_compact.py` | microcompact, auto_compact |
| s07 | `s07_task_system.py` | TaskManager |
| s08 | `s08_background_tasks.py` | BackgroundManager |
| s09 | `s09_agent_teams.py` | TeammateManager, MessageBus |
| s10 | `s10_team_protocols.py` | shutdown_requests, plan_requests |
| s11 | `s11_autonomous_agents.py` | idle_poll, scan_unclaimed_tasks |
| s12 | `s12_worktree_task_isolation.py` | WorktreeManager, EventBus |
| 总纲 | `s_full.py` | 全部机制合一 |

### 3.2 s_full.py 总纲架构

`_full.py` 是全部机制的完整参考实现，包含：

- **24 个工具**：bash, read_file, write_file, edit_file, TodoWrite, task, load_skill, compress, background_run, check_background, task_create, task_get, task_update, task_list, spawn_teammate, list_teammates, send_message, read_inbox, broadcast, shutdown_request, plan_approval, idle, claim_task

- **REPL 命令**：`/compact`, `/tasks`, `/team`, `/inbox`

- **执行流程**：
  1. microcompact（每轮）
  2. auto_compact（超阈值）
  3. drain background notifications
  4. check inbox
  5. LLM call
  6. tool execution
  7. nag reminder
  8. manual compress（可选）

---

## 四、学习路径图

```
第一阶段: 循环                       第二阶段: 规划与知识
==================                   ==============================
s01  Agent Loop              [1]     s03  TodoWrite               [5]
     while + stop_reason                  TodoManager + nag 提醒
     |                                    |
     +-> s02  Tool Use            [4]     s04  Subagent             [5]
              dispatch map: name->handler     每个 Subagent 独立 messages[]
                                              |
                                         s05  Skills               [5]
                                              SKILL.md 通过 tool_result 注入
                                              |
                                         s06  Context Compact      [5]
                                              三层 Context Compact

第三阶段: 持久化                     第四阶段: 团队
==================                   =====================
s07  Task System             [8]     s09  Agent Teams             [9]
     文件持久化 CRUD + 依赖图             队友 + JSONL 邮箱
     |                                    |
s08  Background Tasks        [6]     s10  Team Protocols          [12]
     守护线程 + 通知队列                  关机 + 计划审批 FSM
                                          |
                                     s11  Autonomous Agents       [14]
                                          空闲轮询 + 自动认领
                                     |
                                     s12  Worktree Isolation      [16]
                                          Task 协调 + 按需隔离执行通道

                                     [N] = 工具数量
```

---

## 五、关键设计原则

### 5.1 Harness 工程师职责

1. **实现工具**：给 agent 一双手，设计时原子化、可组合、描述清晰
2. **策划知识**：给 agent 领域专长，按需加载，不要前置塞入
3. **管理上下文**：给 agent 干净的记忆，子 agent 隔离，上下文压缩
4. **控制权限**：给 agent 边界，沙箱化文件访问，对破坏性操作要求审批
5. **收集任务过程数据**：Agent 执行的行动序列是训练信号

### 5.2 核心格言速查

| 课程 | 格言 |
|------|------|
| s01 | One loop & Bash is all you need |
| s02 | 加一个工具，只加一个 handler |
| s03 | 没有计划的 agent 走哪算哪 |
| s04 | 大任务拆小，每个个小任务干净的上下文 |
| s05 | 用到什么知识，临时加载什么知识 |
| s06 | 上下文总会满，要有办法腾地方 |
| s07 | 大目标要拆成小任务，排好序，记在磁盘上 |
| s08 | 慢操作丢后台，agent 继续想下一步 |
| s09 | 任务太大一个人干不完，要能分给队友 |
| s10 | 队友之间要有统一的沟通规矩 |
| s11 | 队友自己看看板，有活就认领 |
| s12 | 各干各的目录，互不干扰 |

---

## 六、姊妹教程

- **learn-claude-code**：临时会话型 agent harness（循环、工具、规划、团队、worktree 隔离）
- **claw0**：主动式常驻 harness（心跳、定时任务、IM 通道、记忆、Soul 人格）

---

## 七、快速开始

```sh
git clone https://github.com/shareAI-lab/learn-claude-code
cd learn-claude-code
pip install -r requirements.txt
cp .env.example .env   # 编辑 .env 填入你的 ANTHROPIC_API_KEY

python agents/s01_agent_loop.py       # 从这里开始
python agents/s12_worktree_task_isolation.py  # 完整递进终点
python agents/s_full.py               # 总纲: 全部机制合一
```

---

**模型就是 Agent。代码是 Harness。造好 Harness，Agent 会完成剩下的。**

**Bash is all you need. Real agents are all the universe needs.**