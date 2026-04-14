# Learn-LLM

LLM（大语言模型）学习项目，涵盖Transformer模型、微调技术、Agent架构、爬虫技术等多个领域的系统性学习内容。

## 项目概述

本项目是一个系统性学习LLM相关知识的项目，包含理论学习笔记和实践代码。主要分为以下几个模块：

- **LLM课程**：基于HuggingFace官方课程的深入学习笔记
- **Agent课程**：深入理解Claude Code架构和Agent开发原理
- **知识库**：Python技术栈相关知识点汇总
- **爬虫学习**：静态/动态页面爬取技术

## 项目结构

```
learn-llm/
│
├── docs/                           # 文档目录
│   ├── agent-course/               # Agent课程文档
│   ├── knowledge-base/             # 知识库
│   ├── learn-agent/                # Agent学习笔记
│   ├── learn-claude-code/          # Claude Code架构学习（核心）
│   └── llm-course/                 # LLM课程笔记
│
├── src/                            # 源代码目录
│   ├── learn-claude-code/          # Claude Code学习代码
│   ├── learn-scraper/              # 爬虫代码
│   └── llm-course/                 # LLM课程实践代码
│
├── .env.example                    # 环境变量示例
├── .gitignore                      # Git忽略配置
├── LICENSE                         # 许可证
└── README.md                       # 项目说明
```

---

## 一、LLM课程（HuggingFace Course）

基于HuggingFace官方LLM课程的系统学习，涵盖从基础到进阶的完整知识体系。

### 学习内容

| 章节 | 主题 | 关键知识点 |
|------|------|------------|
| 第2章 | Transformer模型 | NLP任务、模型架构、Tokenizer（WordPiece分词） |
| 第3章 | 微调预训练模型 | Trainer API、自定义训练循环、Accelerate加速 |
| 第5章 | Datasets库 | 数据加载、预处理、动态填充、大数据处理 |
| 第7章 | NLP任务实践 | 命名实体识别（NER）、掩码语言模型微调 |

### 核心概念

#### 1. Transformer模型基础
- **Tokenizer API**：文本到数字的转换（预处理 → 模型 → 后处理）
- **分词策略**：
  - 基于单词：无法表示相似词的相似性
  - 基于字符：失去单词含义
  - 基于子词（WordPiece）：结合两者优点，BERT使用此策略

#### 2. 模型微调
- **Trainer API**：高级微调接口，封装训练循环
- **自定义训练循环**：使用PyTorch原生API，支持分布式训练
- **Accelerate库**：支持多GPU/TPU分布式训练

```python
# 微调流程示例
raw_datasets = load_dataset("glue", "mrpc")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

trainer = Trainer(model, training_args, train_dataset, eval_dataset)
trainer.train()
```

#### 3. 数据处理原则
- **惰性加载**：Datasets使用Apache Arrow，不一次性加载全部数据
- **动态填充**：每批填充到该批最长长度，节省内存
- **map操作**：批量处理，batched=True提速10-100倍

#### 4. 命名实体识别（NER）
- **标签对齐**：BERT子词分词后，需要将单词级标签对齐到token级
- **BIO标注**：B-XXX（实体开始）、I-XXX（实体内部）、O（非实体）
- **评估指标**：seqeval按实体级别评估，而非token级别

### 参考文档

- [llm-course-part-1.md](docs/llm-course/llm-course-part-1.md) - Transformer基础与微调
- [llm-course-part-2.md](docs/llm-course/llm-course-part-2.md) - Datasets库与NER任务
- [finetuning-principles.md](docs/llm-course/reference/finetuning-principles.md) - 领域适应与迁移学习
- [datasets-large-data-principles.md](docs/llm-course/reference/datasets-large-data-principles.md) - 大数据处理原则

### 实践代码

```
src/llm-course/
├── 2-transformers-tokenizer.py          # Tokenizer使用
├── 2-transformers-using.py              # Transformer模型使用
├── 3-load-datasets.py                   # 数据集加载
├── 3-preprocess-datasets.py             # 数据预处理
├── 3-trainer-fine-tuning.py             # Trainer API微调
├── 3-manual-fine-tuning.py              # 自定义训练循环
├── 7-train-bert-finetuned-ner.py        # NER任务微调
├── 7-train-bert-finetuned-ner-accelerate.py  # Accelerate加速版
├── 5-embedding-then-faiss-search.py     # FAISS语义搜索
└── 5-fetch-issues-with-comments.py      # GitHub数据采集
```

---

## 二、Agent课程（Claude Code架构学习）

深入理解Claude Code的Agent架构设计，学习如何构建围绕Agent模型的工作环境（Harness）。

### 核心理念

> **Agent 是模型。不是框架。不是提示词链。不是拖拽式工作流。**
>
> Agent 是一个神经网络——Transformer、RNN——经过数十亿次梯度更新，在行动序列数据上学会了感知环境、推理目标、采取行动。

#### Agent vs Harness

| 概念 | 角色 | 责责 |
|------|------|------|
| **Agent** | 决策者 | 模型本身，做推理、做决策 |
| **Harness** | 环境 | 工具、知识、上下文、权限边界 |

```
Harness = Tools + Knowledge + Observation + Action + Permissions

    Tools:          文件读写、Shell、网络、数据库、浏览器
    Knowledge:      产品文档、领域资料、API规范、风格指南
    Observation:    git diff、错误日志、浏览器状态
    Action:         CLI命令、API调用、UI交互
    Permissions:    沙箱隔离、审批流程、信任边界
```

### 12个递进式课程

从简单循环到隔离化的自治执行，每个课程添加一个Harness机制。

| 课程 | 主题 | 格言 | 核心内容 |
|------|------|------|----------|
| s01 | Agent Loop | *One loop & Bash is all you need* | while + stop_reason，30行代码构建Agent |
| s02 | Tool Use | *加一个工具，只加一个handler* | dispatch map，工具注册机制 |
| s03 | TodoWrite | *没有计划的agent走哪算哪* | TodoManager + nag提醒 |
| s04 | Subagent | *大任务拆小，干净上下文* | 子Agent独立messages[] |
| s05 | Skills | *用到什么知识，临时加载什么* | Skill.md通过tool_result注入 |
| s06 | Context Compact | *上下文总会满，要有办法腾地方* | 三层压缩策略 |
| s07 | Task System | *大目标拆成小任务，记在磁盘上* | 文件持久化CRUD + 依赖图 |
| s08 | Background Tasks | *慢操作丢后台* | 守护线程 + 通知队列 |
| s09 | Agent Teams | *任务太大一个人干不完* | 队友 + JSONL邮箱 |
| s10 | Team Protocols | *队友之间要有统一的沟通规矩* | 关机 + 计划审批FSM |
| s11 | Autonomous Agents | *队友自己看看板，有活就认领* | 空闲轮询 + 自动认领 |
| s12 | Worktree Isolation | *各干各的目录，互不干扰* | Task协调 + 按需隔离 |

### Agent Loop核心模式

```python
def agent_loop(messages):
    while True:
        response = client.messages.create(
            model=MODEL, system=SYSTEM,
            messages=messages, tools=TOOLS,
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            return

        results = []
        for block in response.content:
            if block.type == "tool_use":
                output = TOOL_HANDLERS[block.name](**block.input)
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": output,
                })
        messages.append({"role": "user", "content": results})
```

### 学习路径

```
第一阶段: 循环                       第二阶段: 规划与知识
s01  Agent Loop              →      s03  TodoWrite
     |                                |
s02  Tool Use                     s04  Subagent
                                       |
                                  s05  Skills
                                       |
                                  s06  Context Compact

第三阶段: 持久化                     第四阶段: 团队
s07  Task System              →     s09  Agent Teams
     |                                |
s08  Background Tasks            s10  Team Protocols
                                       |
                                  s11  Autonomous Agents
                                       |
                                  s12  Worktree Isolation
```

### 参考文档

- [README.md](docs/learn-claude-code/README.md) - Agent与Harness的核心理念
- [s01-the-agent-loop.md](docs/learn-claude-code/s01-the-agent-loop.md) - Agent循环基础
- [s02-tool-use.md](docs/learn-claude-code/s02-tool-use.md) - 工具使用
- [s03-todo-write.md](docs/learn-claude-code/s03-todo-write.md) - 任务计划
- ... (共12个课程文档)

### 实践代码

```
src/learn-claude-code/agents/
├── s01_agent_loop.py               # Agent循环
├── s02_tool_use.py                 # 工具使用
├── s03_todo_write.py               # 任务计划
├── s04_subagent.py                 # 子Agent
├── s05_skill_loading.py            # Skill加载
├── s06_context_compact.py          # 上下文压缩
├── s07_task_system.py              # 任务系统
├── s08_background_tasks.py         # 后台任务
├── s09_agent_teams.py              # Agent团队
├── s10_team_protocols.py           # 团队协议
├── s11_autonomous_agents.py        # 自治Agent
├── s12_worktree_task_isolation.py  # Worktree隔离
└── s_full.py                       # 全部机制合一
```

---

## 三、知识库

Python技术栈相关知识点汇总，包含实用工具和技术笔记。

### 内容列表

| 文档 | 主题 | 内容 |
|------|------|------|
| [python-syntax-notes.md](docs/knowledge-base/python-syntax-notes.md) | Python语法 | 特殊语法、常用技巧 |
| [pydantic使用指南.md](docs/knowledge-base/pydantic使用指南.md) | Pydantic | 数据验证、模型定义 |
| [asyncio事件循环机制总结.md](docs/knowledge-base/asyncio事件循环机制总结.md) | Asyncio | 异步编程、事件循环 |
| [git-worktree-guide.md](docs/knowledge-base/git-worktree-guide.md) | Git Worktree | 多分支并行工作 |
| [tmux-guide.md](docs/knowledge-base/tmux-guide.md) | Tmux | 终端复用、会话管理 |
| [psutil-系统监控.md](docs/knowledge-base/psutil-系统监控.md) | Psutil | 系统监控、进程管理 |
| [环境变量作用域与存储.md](docs/knowledge-base/环境变量作用域与存储.md) | 环境变量 | 配置管理 |
| [scraper.md](docs/knowledge-base/scraper.md) | 爬虫技术 | 静态/动态/智能爬虫模板 |

---

## 四、爬虫学习

提供三种爬虫模板，支持静态页面、动态页面和智能判断。

### 爬虫类型

| 类型 | 技术 | 适用场景 |
|------|------|----------|
| 静态爬虫 | requests + BeautifulSoup | 服务端渲染页面 |
| 动态爬虫 | Playwright | JavaScript渲染页面、需登录页面 |
| 智能爬虫 | 自动判断 | 未知页面类型，自动选择合适方式 |

### 核心特性

#### 智能爬虫
- **页面类型检测**：自动识别React/Vue/Angular/Next.js等框架
- **登录页面识别**：检测登录特征，提示手动登录
- **置信度评估**：基于多维度判断页面是否动态

#### 动态爬虫
- **手动登录支持**：有头模式下等待用户登录
- **Cookie配置**：支持从浏览器复制Cookie
- **内容提取**：HTML + 纯文本 + 标题

#### 静态爬虫
- **重试机制**：max_retries配置
- **User-Agent轮换**：避免被封
- **请求延迟**：request_delay防止高频请求

### 实践代码

```
src/learn-scraper/
├── static-page-scraper.py          # 静态页面爬虫
├── dynamic-page-scraper.py         # 动态页面爬虫
├── smart-page-scraper.py           # 智能爬虫（自动判断）
```

### 参考文档

- [scraper.md](docs/knowledge-base/scraper.md) - 完整爬虫模板和使用说明

---

## 快速开始

### 环境准备

```bash
# 克隆项目
git clone https://github.com/shareAI-lab/learn-llm
cd learn-llm

# 安装依赖（各模块依赖不同，请查看对应requirements.txt）
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env 填入必要的API密钥
```

### 运行示例

```bash
# LLM课程 - NER任务微调
python src/llm-course/s07_train_bert_finetuned_ner.py

# Agent课程 - Agent循环基础
python src/learn-claude-code/agents/s01_agent_loop.py

# 爬虫 - 智能爬虫
python src/learn-scraper/smart_page_scraper.py
```

---

## 学习建议

### 推荐学习路径

1. **LLM基础** → 先理解Transformer模型和Tokenizer
2. **微调实践** → 使用Trainer API微调模型，理解训练流程
3. **Agent架构** → 学习Claude Code的12个课程，理解Harness工程
4. **工具技能** → 掌握爬虫、Git Worktree等实用工具

### 关键学习点

- **模型是Agent**：Agent的能力来自模型训练，而非代码逻辑
- **Harness是环境**：为Agent提供工具、知识、权限边界
- **领域适应**：专业领域需先进行领域内微调再任务微调
- **上下文管理**：Agent需要干净、高效的消息上下文

---

## 许可证

MIT License

---

## 参考资源

- [HuggingFace LLM Course](https://huggingface.co/learn/llm-course/zh-CN/chapter0/1)
- [Claude Code Architecture](https://github.com/shareAI-lab/learn-claude-code)
- [Transformers Documentation](https://huggingface.co/docs/transformers)