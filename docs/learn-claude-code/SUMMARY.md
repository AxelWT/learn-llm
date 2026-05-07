# Learn Claude Code 项目总结

本文档对 `docs/learn-claude-code` 文档目录和 `src/learn-claude-code/agents` 代码目录进行系统性总结。

---

## 一、核心理念：模型即 Agent

### 1.1 Agent 的本质定义

本项目的核心观点：**Agent 是模型，不是框架，不是提示词链，不是拖拽式工作流。**

Agent 是一个神经网络（Transformer、RNN 等），通过数十亿次梯度更新，在行动序列数据上学会了感知环境、推理目标、采取行动。从
DeepMind DQN 玩 Atari（2013）到 OpenAI Five 征服 Dota 2（2019），再到腾讯绝悟统治王者荣耀（2019），每一个里程碑都证明：Agent
永远是模型本身。

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

### 1.3 Claude Code 介绍

Claude Code 是一种 agent harness 实现：

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

---

## 二、递进式课程详解

### 第一阶段：循环基础

#### **s01: Agent Loop（Agent 循环）**

> *"One loop & Bash is all you need"* — 一个工具 + 一个循环 = 一个 Agent

**核心模式**：一个退出条件控制的 while 循环，持续运行直到模型不再调用工具。

- **模拟流程图**

```
+--------+      +-------+      +---------+
|  User  | ---> |  LLM  | ---> |  Tool   |
| prompt |      |       |      | execute |
+--------+      +---+---+      +----+----+
                    ^                |
                    |   tool_result  |
                    +----------------+
                    (loop until stop_reason != "tool_use")
```

- **代码示例**

```python
def agent_loop(query):
    messages = [{"role": "user", "content": query}]
    while True:
        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=8000,
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            return

        results = []
        for block in response.content:
            if block.type == "tool_use":
                output = run_bash(block.input["command"])
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": output,
                })
        messages.append({"role": "user", "content": results})
```

---

#### **s02: Tool Use（工具使用）**

> *"加一个工具，只加一个 handler"* — 循环不用动，新工具注册进 dispatch map

**核心模式**：使用字典映射工具名到处理函数，实现工具分发。

- **模拟流程图**

```
+--------+      +-------+      +------------------+
|  User  | ---> |  LLM  | ---> | Tool Dispatch    |
| prompt |      |       |      | {                |
+--------+      +---+---+      |   bash: run_bash |
                    ^           |   read: run_read |
                    |           |   write: run_wr  |
                    +-----------+   edit: run_edit |
                    tool_result | }                |
                                +------------------+

The dispatch map is a dict: {tool_name: handler_function}.
One lookup replaces any if/elif chain.
```

```python
def safe_path(p: str) -> Path:
    path = (WORKDIR / p).resolve()
    if not path.is_relative_to(WORKDIR):
        raise ValueError(f"Path escapes workspace: {p}")
    return path


TOOL_HANDLERS = {
    "bash": lambda **kw: run_bash(kw["command"]),
    "read_file": lambda **kw: run_read(kw["path"]),
    "write_file": lambda **kw: run_write(kw["path"], kw["content"]),
    "edit_file": lambda **kw: run_edit(kw["path"], kw["old_text"], kw["new_text"]),
}


def agent_loop(messages: list):
    while True:
        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=8000,
        )
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            return
        results = []
        for block in response.content:
            if block.type == "tool_use":
                handler = TOOL_HANDLERS.get(block.name)
                output = handler(**block.input) if handler else f"Unknown tool: {block.name}"
                print(f"> {block.name}:")
                print(output[:200])
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": output})
        messages.append({"role": "user", "content": results})
```

**新增组件**：

- `safe_path()` 路径沙箱防止逃逸工作区
- 专用工具替代全部走 shell 的方式
- 加工具 = 加 handler + 加 schema，循环永远不变

---

### 第二阶段：规划与知识

#### **s03: TodoWrite（待办写入）**

> *"没有计划的 agent 走哪算哪"* — 先列步骤再动手，完成率翻倍

- **模拟流程图**

```
+--------+      +-------+      +---------+
|  User  | ---> |  LLM  | ---> | Tools   |
| prompt |      |       |      | + todo  |
+--------+      +---+---+      +----+----+
                    ^                |
                    |   tool_result  |
                    +----------------+
                          |
              +-----------+-----------+
              | TodoManager state     |
              | [ ] task A            |
              | [>] task B  <- doing  |
              | [x] task C            |
              +-----------------------+
                          |
              if rounds_since_todo >= 3:
                inject <reminder> into tool_result
```

**核心机制**：

- `TodoManager` 存储带状态的项目，同一时间只允许一个 `in_progress`
- "同时只能有一个 in_progress" 强制顺序聚焦
- **Nag Reminder**：模型连续 3 轮不调用 `todo` 时注入提醒，制造问责压力

```python
# -- TodoManager: structured state the LLM writes to --
class TodoManager:
    def __init__(self):
        self.items = []

    def update(self, items: list) -> str:
        if len(items) > 20:
            raise ValueError("Max 20 todos allowed")
        validated = []
        in_progress_count = 0
        for i, item in enumerate(items):
            text = str(item.get("text", "")).strip()
            status = str(item.get("status", "pending")).lower()
            item_id = str(item.get("id", str(i + 1)))
            if not text:
                raise ValueError(f"Item {item_id}: text required")
            if status not in ("pending", "in_progress", "completed"):
                raise ValueError(f"Item {item_id}: invalid status '{status}'")
            if status == "in_progress":
                in_progress_count += 1
            validated.append({"id": item_id, "text": text, "status": status})
        if in_progress_count > 1:
            raise ValueError("Only one task can be in_progress at a time")
        self.items = validated
        return self.render()

    def render(self) -> str:
        if not self.items:
            return "No todos."
        lines = []
        for item in self.items:
            marker = {"pending": "[ ]", "in_progress": "[>]", "completed": "[x]"}[item["status"]]
            lines.append(f"{marker} #{item['id']}: {item['text']}")
        done = sum(1 for t in self.items if t["status"] == "completed")
        lines.append(f"\n({done}/{len(self.items)} completed)")
        return "\n".join(lines)


# -- Agent loop with nag reminder injection --
def agent_loop(messages: list):
    rounds_since_todo = 0
    while True:
        # Nag reminder is injected below, alongside tool results
        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=8000,
        )
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            return
        results = []
        used_todo = False
        for block in response.content:
            if block.type == "tool_use":
                handler = TOOL_HANDLERS.get(block.name)
                try:
                    output = handler(**block.input) if handler else f"Unknown tool: {block.name}"
                except Exception as e:
                    output = f"Error: {e}"
                print(f"> {block.name}:")
                print(str(output)[:200])
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": str(output)})
                if block.name == "todo":
                    used_todo = True
        rounds_since_todo = 0 if used_todo else rounds_since_todo + 1
        if rounds_since_todo >= 3:
            results.append({"type": "text", "text": "<reminder>Update your todos.</reminder>"})
        messages.append({"role": "user", "content": results})
```

---

#### **s04: Subagent（子 Agent 调度）**

> *"大任务拆小，每个个小任务干净的上下文"* — Subagent 用独立 messages[]，不污染主对话

- **模拟流程图**

```
Parent agent                     Subagent
+------------------+             +------------------+
| messages=[...]   |             | messages=[]      | <-- fresh
|                  |  dispatch   |                  |
| tool: task       | ----------> | while tool_use:  |
|   prompt="..."   |             |   call tools     |
|                  |  summary    |   append results |
|   result = "..." | <---------- | return last text |
+------------------+             +------------------+

Parent context stays clean. Subagent context is discarded.
```

**核心机制**：

- 父 Agent 有 `task` 工具派发子任务
- Subagent 以 `messages=[]` 启动，拥有除 `task` 外的所有基础工具（禁止递归生成）
- Subagent 运行自己的循环，只有最终文本返回给父 Agent
- 整个消息历史直接丢弃，父 Agent 只收到摘要文本

**解决的问题**：Agent 工作越久，messages 数组越臃肿。Subagent 可能跑了 30+ 次工具调用，但父 Agent 只需要一个词："pytest"。

```python
# -- Subagent: fresh context, filtered tools, summary-only return --
def run_subagent(prompt: str) -> str:
    sub_messages = [{"role": "user", "content": prompt}]  # fresh context
    for _ in range(30):  # safety limit
        response = client.messages.create(
            model=MODEL, system=SUBAGENT_SYSTEM, messages=sub_messages,
            tools=CHILD_TOOLS, max_tokens=8000,
        )
        sub_messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            break
        results = []
        for block in response.content:
            if block.type == "tool_use":
                handler = TOOL_HANDLERS.get(block.name)
                output = handler(**block.input) if handler else f"Unknown tool: {block.name}"
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": str(output)[:50000]})
        sub_messages.append({"role": "user", "content": results})
    # Only the final text returns to the parent -- child context is discarded
    return "".join(b.text for b in response.content if hasattr(b, "text")) or "(no summary)"


def agent_loop(messages: list):
    while True:
        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=PARENT_TOOLS, max_tokens=8000,
        )
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            return
        results = []
        for block in response.content:
            if block.type == "tool_use":
                if block.name == "task":
                    desc = block.input.get("description", "subtask")
                    prompt = block.input.get("prompt", "")
                    print(f"> task ({desc}): {prompt[:80]}")
                    output = run_subagent(prompt)
                else:
                    handler = TOOL_HANDLERS.get(block.name)
                    output = handler(**block.input) if handler else f"Unknown tool: {block.name}"
                print(f"  {str(output)[:200]}")
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": str(output)})
        messages.append({"role": "user", "content": results})
```

---

#### **s05: Skill Loading（Skill 加载）**

> *"用到什么知识，临时加载什么知识"* — 通过 tool_result 注入，不塞 system prompt

- **模拟流程图**

```
System prompt (Layer 1 -- always present):
+--------------------------------------+
| You are a coding agent.              |
| Skills available:                    |
|   - git: Git workflow helpers        |  ~100 tokens/skill
|   - test: Testing best practices     |
+--------------------------------------+

When model calls load_skill("git"):
+--------------------------------------+
| tool_result (Layer 2 -- on demand):  |
| <skill name="git">                   |
|   Full git workflow instructions...  |  ~2000 tokens
|   Step 1: ...                        |
| </skill>                             |
+--------------------------------------+
```

**两层加载策略**：

- **第一层（系统提示）**：放 Skill 名称（低成本，约 100 tokens/skill）
- **第二层（tool_result）**：按需放完整内容（约 2000 tokens）

```python
# 每个 Skill 是一个目录，包含 SKILL.md 文件和 YAML frontmatter
skills /
pdf / SKILL.md  # ---\n name: pdf\n description: Process PDF files\n ---\n ...
code - review / SKILL.md


# SkillLoader 递归扫描 `SKILL.md` 文件, 用目录名作为 Skill 标识。
class SkillLoader:
    def __init__(self, skills_dir: Path):
        self.skills = {}
        for f in sorted(skills_dir.rglob("SKILL.md")):
            text = f.read_text()
            meta, body = self._parse_frontmatter(text)
            name = meta.get("name", f.parent.name)
            self.skills[name] = {"meta": meta, "body": body}

    def get_descriptions(self) -> str:
        lines = []
        for name, skill in self.skills.items():
            desc = skill["meta"].get("description", "")
            lines.append(f"  - {name}: {desc}")
        return "\n".join(lines)

    def get_content(self, name: str) -> str:
        skill = self.skills.get(name)
        if not skill:
            return f"Error: Unknown skill '{name}'."
        return f"<skill name=\"{name}\">\n{skill['body']}\n</skill>"
```

---

#### **s06: Context Compact（上下文压缩）**

> *"上下文总会满，要有办法腾地方"* — 三层压缩策略，换来无限会话

- **模拟流程图**

```
Every turn:
+------------------+
| Tool call result |
+------------------+
        |
        v
[Layer 1: micro_compact]        (silent, every turn)
  Replace tool_result > 3 turns old
  with "[Previous: used {tool_name}]"
        |
        v
[Check: tokens > 50000?]
   |               |
   no              yes
   |               |
   v               v
continue    [Layer 2: auto_compact]
              Save transcript to .transcripts/
              LLM summarizes conversation.
              Replace all messages with [summary].
                    |
                    v
            [Layer 3: compact tool]
              Model calls compact explicitly.
              Same summarization as auto_compact.
```

**三层压缩机制**：

| 层级 | 名称             | 触发条件            | 动作                        |
|----|----------------|-----------------|---------------------------|
| 1  | micro_compact  | 每轮静默执行          | 旧 tool result (>3轮) → 占位符 |
| 2  | auto_compact   | token > 50000   | 保存 transcript 到磁盘，LLM 摘要  |
| 3  | manual compact | 模型调用 compact 工具 | 同 auto_compact            |

**关键洞察**：完整历史通过 transcript 保存在磁盘上。信息没有真正丢失，只是移出了活跃上下文。

```python
def agent_loop(messages: list):
    while True:
        micro_compact(messages)  # Layer 1
        if estimate_tokens(messages) > THRESHOLD:
            messages[:] = auto_compact(messages)  # Layer 2
        response = client.messages.create(...)
        # ... tool execution ...
        if manual_compact:
            messages[:] = auto_compact(messages)  # Layer 3
```

---

### 第三阶段：持久化

#### **s07: Task System（任务系统）**

> *"大目标要拆成小任务，排好序，记在磁盘上"* — 文件持久化的任务图

**核心机制**：把扁平清单升级为持久化到磁盘的**任务图（DAG）**。

每个任务是一个 JSON 文件，有状态、前置依赖 (`blockedBy`)：

- **模拟流程图**

```
.tasks/
  task_1.json  {"id":1, "status":"completed"}
  task_2.json  {"id":2, "blockedBy":[1], "status":"pending"}
  task_3.json  {"id":3, "blockedBy":[1], "status":"pending"}
  task_4.json  {"id":4, "blockedBy":[2,3], "status":"pending"}
  
  
任务图 (DAG):
                 +----------+
            +--> | task 2   | --+
            |    | pending  |   |
+----------+     +----------+    +--> +----------+
| task 1   |                          | task 4   |
| completed| --> +----------+    +--> | blocked  |
+----------+     | task 3   | --+     +----------+
                 | pending  |
                 +----------+

顺序:   task 1 必须先完成, 才能开始 2 和 3
并行:   task 2 和 3 可以同时执行
依赖:   task 4 要等 2 和 3 都完成
状态:   pending -> in_progress -> completed
```

**任务图随时回答三个问题**：

- 什么可以做？—— pending + blockedBy 为空
- 什么被卡住？—— 等待前置任务完成
- 什么做完了？—— completed，自动解锁后续任务

```python
# 创建任务
class TaskManager:
    def __init__(self, tasks_dir: Path):
        self.dir = tasks_dir
        self.dir.mkdir(exist_ok=True)
        self._next_id = self._max_id() + 1

    def create(self, subject, description=""):
        task = {"id": self._next_id, "subject": subject,
                "status": "pending", "blockedBy": [],
                "owner": ""}
        self._save(task)
        self._next_id += 1
        return json.dumps(task, indent=2)

def _clear_dependency(self, completed_id):
    for f in self.dir.glob("task_*.json"):
        task = json.loads(f.read_text())
        if completed_id in task.get("blockedBy", []):
            task["blockedBy"].remove(completed_id)
            self._save(task)

# 更新任务状态
def update(self, task_id, status=None,
           add_blocked_by=None, remove_blocked_by=None):
    task = self._load(task_id)
    if status:
        task["status"] = status
        if status == "completed":
            self._clear_dependency(task_id)
    if add_blocked_by:
        task["blockedBy"] = list(set(task["blockedBy"] + add_blocked_by))
    if remove_blocked_by:
        task["blockedBy"] = [x for x in task["blockedBy"] if x not in remove_blocked_by]
    self._save(task)

TOOL_HANDLERS = {
    # ...base tools...
    "task_create": lambda **kw: TASKS.create(kw["subject"]),
    "task_update": lambda **kw: TASKS.update(kw["task_id"], kw.get("status")),
    "task_list":   lambda **kw: TASKS.list_all(),
    "task_get":    lambda **kw: TASKS.get(kw["task_id"]),
}
```

---

#### **s08: Background Tasks（后台任务）**

> *"慢操作丢后台，agent 继续想下一步"* — 后台线程跑命令，完成后注入通知

- **模拟流程图**

```
Main thread                Background thread
+-----------------+        +-----------------+
| agent loop      |        | subprocess runs |
| ...             |        | ...             |
| [LLM call]  <---+------- | enqueue(result) |
|  ^drain queue   |        +-----------------+
+-----------------+

Timeline:
Agent --[spawn A]--[spawn B]--[other work]----
             |          |
             v          v
          [A runs]   [B runs]      (parallel)
             |          |
             +-- results injected before next LLM call --+
```

**核心机制**：

- `BackgroundManager` 用线程安全的通知队列追踪任务
- 子进程完成后，结果进入通知队列
- 每次 LLM 调用前排空通知队列

**解决的问题**：`npm install`、`pytest`、`docker build` 等慢命令阻塞式循环下模型只能干等。

```python
# -- BackgroundManager: threaded execution + notification queue --
class BackgroundManager:
    def __init__(self):
        self.tasks = {}  # task_id -> {status, result, command}
        self._notification_queue = []  # completed task results
        self._lock = threading.Lock()

    def run(self, command: str) -> str:
        """Start a background thread, return task_id immediately."""
        task_id = str(uuid.uuid4())[:8]
        self.tasks[task_id] = {"status": "running", "result": None, "command": command}
        thread = threading.Thread(
            target=self._execute, args=(task_id, command), daemon=True
        )
        thread.start()
        return f"Background task {task_id} started: {command[:80]}"

    def _execute(self, task_id: str, command: str):
        """Thread target: run subprocess, capture output, push to queue."""
        try:
            r = subprocess.run(
                command, shell=True, cwd=WORKDIR,
                capture_output=True, text=True, timeout=300
            )
            output = (r.stdout + r.stderr).strip()[:50000]
            status = "completed"
        except subprocess.TimeoutExpired:
            output = "Error: Timeout (300s)"
            status = "timeout"
        except Exception as e:
            output = f"Error: {e}"
            status = "error"
        self.tasks[task_id]["status"] = status
        self.tasks[task_id]["result"] = output or "(no output)"
        with self._lock:
            self._notification_queue.append({
                "task_id": task_id,
                "status": status,
                "command": command[:80],
                "result": (output or "(no output)")[:500],
            })

    def check(self, task_id: str = None) -> str:
        """Check status of one task or list all."""
        if task_id:
            t = self.tasks.get(task_id)
            if not t:
                return f"Error: Unknown task {task_id}"
            return f"[{t['status']}] {t['command'][:60]}\n{t.get('result') or '(running)'}"
        lines = []
        for tid, t in self.tasks.items():
            lines.append(f"{tid}: [{t['status']}] {t['command'][:60]}")
        return "\n".join(lines) if lines else "No background tasks."

    def drain_notifications(self) -> list:
        """Return and clear all pending completion notifications."""
        with self._lock:
            notifs = list(self._notification_queue)
            self._notification_queue.clear()
        return notifs

TOOL_HANDLERS = {
    "background_run":   lambda **kw: BG.run(kw["command"]),
    "check_background": lambda **kw: BG.check(kw.get("task_id")),
}
    
def agent_loop(messages: list):
    while True:
        # Drain background notifications and inject as system message before LLM call
        notifs = BG.drain_notifications()
        if notifs and messages:
            notif_text = "\n".join(
                f"[bg:{n['task_id']}] {n['status']}: {n['result']}" for n in notifs
            )
            messages.append({"role": "user", "content": f"<background-results>\n{notif_text}\n</background-results>"})
        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages,
            tools=TOOLS, max_tokens=8000,
        )
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            return
        results = []
        for block in response.content:
            if block.type == "tool_use":
                handler = TOOL_HANDLERS.get(block.name)
                try:
                    output = handler(**block.input) if handler else f"Unknown tool: {block.name}"
                except Exception as e:
                    output = f"Error: {e}"
                print(f"> {block.name}:")
                print(str(output)[:200])
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": str(output)})
        messages.append({"role": "user", "content": results})
```

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