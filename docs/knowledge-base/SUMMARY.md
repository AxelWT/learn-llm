# Python 开发知识库总结

本知识库整理了 Python 开发中常用的核心技术、工具使用指南和最佳实践，涵盖异步编程、数据验证、系统监控、爬虫开发等多个领域。

---

## 目录

- [Python asyncio 事件循环机制](#python-asyncio-事件循环机制)
- [Pydantic 数据验证框架](#pydantic-数据验证框架)
- [Python 经典技巧与用法](#python-经典技巧与用法)
- [Python psutil 系统监控](#python-psutil-系统监控)
- [网络爬虫开发指南](#网络爬虫开发指南)
- [Monkey Patch 技术指南](#monkey-patch-技术指南)
- [Git Worktree 使用指南](#git-worktree-使用指南)
- [tmux 终端复用工具](#tmux-终端复用工具)
- [环境变量管理](#环境变量管理)
- [Python 语法笔记](#python-语法笔记)

---

## Python asyncio 事件循环机制

### 核心概念

`asyncio` 是 Python 的异步编程框架，核心在于区分**协程**与**普通函数**：

| 场景 | 关键字使用 | 说明 |
|:-----|:-----------|:-----|
| 定义异步函数 | `async def func():` | 声明可挂起的函数 |
| 调用异步函数 | `await func()` | 挂起当前协程等待结果 |
| 定义普通函数 | `def func():` | 不涉及 IO 等待的纯逻辑 |
| 顶层启动协程 | `asyncio.run(main())` | 同步入口的唯一方式 |

### async/await 使用原则

1. **async 只是一个"通行证"** - 标记函数可被挂起，不会自动让代码变快
2. **必须有底层异步支持** - 需调用 `asyncio.sleep()`、`aiohttp` 等异步库
3. **不要滥用 async** - CPU密集型用普通函数，IO密集型用异步函数
4. **async 具有传染性** - 调用异步函数的函数也必须是异步的

### 事件循环架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        单线程事件循环架构                         │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                事件循环 (Event Loop)                     │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌───────┐       │   │
│  │  │ 协程A   │  │ 协程B   │  │ 协程C   │  │ ...   │       │   │
│  │  └─────────┘  └─────────┘  └─────────┘  └───────┘       │   │
│  │  ┌─────────────────────────────────────────────────┐    │   │
│  │  │              就绪队列 (Ready Queue)              │    │   │
│  │  └─────────────────────────────────────────────────┘    │   │
│  │  ┌─────────────────────────────────────────────────┐    │   │
│  │  │    选择器 (Selector: epoll/kqueue/select)       │◄──┼───│ 监听系统I/O事件
│  │  └─────────────────────────────────────────────────┘    │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 协程执行流程

1. `await future` → 协程挂起，返回 Future 给事件循环
2. 事件循环执行其他协程
3. 系统调用 `epoll_wait/select` 等待 I/O 或超时
4. I/O 就绪或定时器到期 → `future.set_result()` 触发回调
5. 回调将协程放入就绪队列
6. 事件循环调用 `coro.send(result)` 恢复协程

### 关键要点

- **单线程默认模式** - 一个事件循环处理所有协程
- **I/O 多路复用** - 通过系统调用高效等待多个事件
- **Future 对象** - 协程与事件循环间的"信物"，传递结果
- **适合场景** - I/O 密集型任务，如网络请求、数据库访问

---

## Pydantic 数据验证框架

### 核心功能

Pydantic 是 Python 的数据验证框架，通过类型注解自动进行数据验证和转换。

### 基本使用

```python
from pydantic import BaseModel

class User(BaseModel):
    id: int
    name: str
    email: str
    is_active: bool = True  # 带默认值

# 创建实例（自动类型转换）
user = User(id="123", name="张三", email="test@example.com")
print(user.model_dump())  # 转换为字典
```

### 字段验证 (Field)

```python
from pydantic import BaseModel, Field

class Product(BaseModel):
    # 数值约束
    age: int = Field(ge=0, le=150)
    score: float = Field(gt=0, lt=100)

    # 字符串约束
    username: str = Field(min_length=3, max_length=20, pattern=r"^\w+$")

    # 默认值
    tags: list[str] = Field(default_factory=list)
```

### 模型配置 (ConfigDict)

| 参数 | 说明 |
|------|------|
| `str_strip_whitespace` | 自动去除字符串空白 |
| `validate_assignment` | 赋值时验证 |
| `extra="forbid"` | 禁止额外字段 |
| `frozen=True` | 不可变模型 |
| `from_attributes=True` | 支持从 ORM 对象创建 |

### 验证器

```python
from pydantic import field_validator, model_validator

class User(BaseModel):
    password: str
    confirm_password: str

    @field_validator("password")
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError("密码至少8字符")
        return v

    @model_validator(mode="after")
    def passwords_match(self):
        if self.password != self.confirm_password:
            raise ValueError("两次密码不一致")
        return self
```

### Pydantic-Settings 配置管理

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    database_url: str
    debug: bool = False

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="APP_",
    )
```

**配置优先级**：命令行参数 > 环境变量 > .env 文件 > 默认值

---

## Python 经典技巧与用法

### 1. 装饰器 (Decorator)

修改函数行为而不改变源码，用于日志记录、性能测量、缓存等。

```python
from functools import wraps

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"耗时: {time.time() - start:.2f}s")
        return result
    return wrapper
```

### 2. 上下文管理器 (Context Manager)

资源自动管理，确保正确释放。

```python
from contextlib import contextmanager

@contextmanager
def timer():
    start = time.time()
    yield
    print(f"耗时: {time.time() - start:.2f}s")
```

### 3. 生成器 (Generator)

惰性计算，节省内存。

```python
def fibonacci():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b
```

### 4. 魔法方法 (Magic Methods)

自定义类的行为：

| 方法 | 用途 |
|------|------|
| `__init__` | 构造函数 |
| `__str__` / `__repr__` | 字符串表示 |
| `__eq__` / `__lt__` | 比较运算 |
| `__add__` / `__mul__` | 算术运算 |
| `__len__` | 长度 |
| `__getitem__` | 索引访问 |
| `__call__` | 可调用 |
| `__enter__` / `__exit__` | 上下文管理器 |

### 5. 鸭子类型 (Duck Typing)

"如果它走起来像鸭子，叫起来像鸭子，那它就是鸭子。" - 关注接口而非类型。

### 6. `*args` 和 `**kwargs`

可变参数，灵活的函数签名。

### 7. functools 工具箱

- `@wraps` - 保留函数元信息
- `partial` - 偏函数，固定部分参数
- `@lru_cache` - 自动缓存结果
- `singledispatch` - 根据参数类型选择实现

### 8. 数据类 (Dataclass)

Python 3.7+ 自动生成 `__init__`、`__repr__`、`__eq__`。

### 9. 描述符 (Descriptor)

属性访问控制，`@property` 的底层原理。

### 10. `__slots__`

限制属性、节省内存（约 40-60%）。

---

## Python psutil 系统监控

### 功能概述

psutil 是跨平台系统监控库，提供 CPU、内存、磁盘、网络、进程等信息。

### CPU 监控

```python
psutil.cpu_percent(interval=1)       # 总体使用率
psutil.cpu_count()                   # 逻辑核心数
psutil.getloadavg()                  # 系统负载 (仅Linux/Unix)
```

### 内存监控

```python
mem = psutil.virtual_memory()
print(f"总内存: {mem.total / (1024**3):.1f} GB")
print(f"使用率: {mem.percent}%")
```

### 磁盘监控

```python
disk = psutil.disk_usage('/')
io = psutil.disk_io_counters()
```

### 网络监控

```python
net = psutil.net_io_counters()
# 每个接口的网络 I/O
net_per_nic = psutil.net_io_counters(pernic=True)
```

### 进程管理

```python
p = psutil.Process(pid)
p.cpu_percent()      # CPU使用率
p.memory_info()      # 内存信息
p.terminate()        # 终止进程
```

### 性能优化

使用 `Process.oneshot()` 优化多次属性访问。

---

## 网络爬虫开发指南

### 页面类型检测

通过特征判断页面类型：
- **React**: `data-reactroot`, `__REACT_DEVTOOLS`
- **Vue**: `data-v-[a-f0-9]+`, `__vue__`
- **Angular**: `ng-version`, `ng-app`
- **SPA**: `__NEXT_DATA__`, `bundle.*.js`

### 静态页面爬虫

使用 `requests + BeautifulSoup`：
- 支持重试机制、User-Agent轮换、请求延迟
- 适合纯静态HTML页面

### 动态页面爬虫

使用 `Playwright`：
- 支持无头/有头模式切换
- 支持手动登录（有头模式）
- 支持 cookie 配置

### Cookie 配置格式

```json
[
  {
    "name": "Cookie",
    "value": "name1=value1; name2=value2",
    "domain": "example.com",
    "path": "/"
  }
]
```

### 最佳实践

1. **请求延迟** - 避免被封，设置 1-2 秒间隔
2. **User-Agent轮换** - 模拟真实用户
3. **重试机制** - 处理网络不稳定
4. **登录页面检测** - 自动识别需要登录的页面

---

## Monkey Patch 技术指南

### 什么是 Monkey Patch

运行时动态修改代码的技术，通过替换模块、类或方法的实现来改变行为。

### 原理

Python 中一切皆对象，可以在运行时重新赋值：

```python
original = module.function
module.function = patched_function
```

### 应用场景

1. **监控与调试** - 监控网络请求、API调用
2. **修复第三方库Bug** - 临时修复无法立即升级的问题
3. **单元测试 Mock** - 替换依赖为假数据
4. **扩展类方法** - 给现有类添加新方法
5. **动态功能开关** - 根据配置启用/禁用功能

### 最佳实践

1. **保存原始版本** - 便于恢复
2. **使用 `@wraps`** - 保留元信息
3. **添加标识** - `__monkey_patched__ = True`
4. **控制作用域** - 使用上下文管理器
5. **文档记录** - 说明目标、原因、影响

### 优缺点

| 优点 | 缺点 |
|------|------|
| 零侵入 | 隐蔽性 |
| 全局生效 | 版本兼容风险 |
| 灵活可控 | 调试困难 |

---

## Git Worktree 使用指南

### 什么是 Git Worktree

允许同一个仓库在不同目录同时 checkout 不同分支，实现并行工作而不互相干扰。

### 核心特点

- 共享同一个 `.git` 数据库
- 不同目录 checkout 不同分支的文件
- 磁盘占用小，切换快

### 常用命令

```bash
# 创建 worktree
git worktree add -b feature-a .worktrees/feature-a main

# 查看所有 worktree
git worktree list

# 删除 worktree
git worktree remove .worktrees/feature-a

# 清理残留记录
git worktree prune
```

### 工作流程示例

**并行开发多个功能**：
```bash
git worktree add -b feature-a .worktrees/feature-a main
git worktree add -b feature-b .worktrees/feature-b main
# 分别在不同目录开发，完成后合并
```

**紧急 Hotfix**：
```bash
git worktree add -b hotfix .worktrees/hotfix main
# 不 stash、不切换，直接修复 bug
```

### 核心规则

**同一个分支不能同时在多个目录 checkout**

---

## tmux 终端复用工具

### 什么是 tmux

终端复用器，允许在一个终端窗口里开多个会话和窗格，断开后重新连接。

### 常用快捷键

> 所有快捷键需要先按 `Ctrl + b`（前缀）

**会话管理**：
| 操作 | 快捷键 |
|------|--------|
| 分离会话 | `Ctrl+b d` |
| 列出会话 | `Ctrl+b s` |
| 重命名会话 | `Ctrl+b $` |

**窗口管理**：
| 操作 | 快捷键 |
|------|--------|
| 新建窗口 | `Ctrl+b c` |
| 切换窗口 | `Ctrl+b 0-9` |
| 下一个窗口 | `Ctrl+b n` |

**窗格管理**：
| 操作 | 快捷键 |
|------|--------|
| 垂直分屏 | `Ctrl+b %` |
| 水平分屏 | `Ctrl+b " ` |
| 切换窗格 | `Ctrl+b 方向键` |
| 放大/恢复 | `Ctrl+b z` |

### 配置文件 (~/.tmux.conf)

```bash
# 设置前缀为 Ctrl+a
set -g prefix C-a

# 启用鼠标支持
set -g mouse on

# 窗口从 1 开始
set -g base-index 1
```

### 典型应用场景

- **SSH 远程开发** - 断线后会话继续运行
- **多任务管理** - 一个窗口多个窗格
- **后台运行** - 长时间任务不中断

---

## 环境变量管理

### 作用域层次

| 作用域 | 范围 | 持久性 |
|--------|------|--------|
| 当前 shell | 仅当前终端会话 | 关闭终端即消失 |
| 子进程 | 当前 shell 及启动的程序 | 随当前 shell 结束 |
| 用户级 | 用户所有终端会话 | 持久 |
| 系统级 | 所有用户 | 持久 |

### 存储位置

**环境变量存储在内存中**，不存储在文件里（除非主动写入配置文件）。

### 配置方式

| 方式 | 命令 | 持久性 |
|------|------|--------|
| export | `export VAR="val"` | 临时 |
| shell 配置 | `~/.zshrc` | 持久 |
| .env 文件 | 项目根目录 | 项目持久 |
| 系统配置 | `/etc/environment` | 系统持久 |

### Python 项目推荐

使用 `python-dotenv` 或 `uv` 自动加载 `.env` 文件。

---

## Python 语法笔记

### 字典解包语法 (**)

```python
batch = {'input_ids': tensor, 'attention_mask': tensor}
outputs = model(**batch)  # 等价于 model(input_ids=..., attention_mask=...)
```

| 符号 | 作用 |
|------|------|
| `**dict` | 字典解包为关键字参数 |
| `*list` | 列表解包为位置参数 |

### range 函数剖析

**range 不是生成器**：
- range 是类，实现 `__iter__` 和 `__next__`
- 可重复迭代
- 惰性计算，不占内存

```python
r = range(5)
list(r)  # [0, 1, 2, 3, 4]
list(r)  # [0, 1, 2, 3, 4]  # 再次迭代仍有效

# 生成器只能迭代一次
g = (x for x in range(5))
list(g)  # [0, 1, 2, 3, 4]
list(g)  # []  # 已耗尽
```

### HTML 实体编码判断

```python
import html

def has_html_entities(text):
    unescaped = html.unescape(text)
    return unescaped != text

# 示例
text = "I&#039;m a transformer"
print(has_html_entities(text))  # True
print(html.unescape(text))  # "I'm a transformer"
```

---

## 文件清单

| 文件名 | 主题 | 内容概述 |
|--------|------|----------|
| asyncio-event-loop-guide.md | 异步编程 | asyncio 原理、事件循环机制、协程执行流程 |
| pydantic-guide.md | 数据验证 | Pydantic 基础、验证器、序列化、配置管理 |
| python-classic-techniques.md | 编程技巧 | 装饰器、上下文管理器、生成器、魔法方法等 |
| psutil-system-monitor.md | 系统监控 | CPU、内存、磁盘、网络、进程监控 |
| scraper-guide.md | 爬虫开发 | 静态/动态爬虫模板、页面类型检测 |
| monkey-patch-guide.md | 运行时修改 | Monkey patch 原理、应用场景、最佳实践 |
| git-worktree-guide.md | Git 技巧 | Worktree 创建、管理、并行开发流程 |
| tmux-guide.md | 终端工具 | tmux 会话、窗口、窗格管理 |
| environment-variables-scope-storage.md | 环境变量 | 作用域、存储位置、配置方式 |
| python-syntax-notes.md | 语法笔记 | 字典解包、range函数、HTML实体处理 |

---

*生成时间: 2026-04-24*