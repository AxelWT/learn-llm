# Monkey Patch 完全指南

## 什么是 Monkey Patch？

**Monkey Patch（猴子补丁）** 是一种在**运行时动态修改代码**的技术，通过替换模块、类或方法的实现来改变其行为，而无需修改原始源代码。

### 核心原理

Python 中一切皆对象，包括函数、类、模块。你可以在运行时重新赋值，从而改变其行为：

```python
# 原始函数
def greet():
    return "Hello"

# 运行时替换（这就是 monkey patch）
greet = lambda: "Hi"

print(greet())  # 输出: "Hi"
```

---

## 典型应用场景与示例

### 1. 监控与调试（最常见）

监控网络请求、API 调用等，无需修改原有代码。

```python
import requests
from functools import wraps
import time

# 保存原始方法
original_request = requests.Session.request

# 创建包装函数
@wraps(original_request)
def monitored_request(self, method, url, **kwargs):
    start = time.time()
    print(f"[请求] {method} {url}")

    response = original_request(self, method, url, **kwargs)

    elapsed = time.time() - start
    print(f"[响应] {response.status_code} | 耗时 {elapsed:.2f}s")

    return response

# 应用 monkey patch
requests.Session.request = monitored_request

# 现在所有 requests 调用都会被监控
requests.get("https://api.github.com")  # 自动打印日志
```

### 2. 修复第三方库的 Bug

当第三方库有问题且无法立即升级时，临时修复。

```python
# 假设某个库的函数有 bug
from some_library import buggy_function

# 保存原始版本
_original = buggy_function

def fixed_function(*args, **kwargs):
    # 添加修复逻辑
    if args and args[0] is None:
        args = (0,) + args[1:]  # 修复 None 参数问题
    return _original(*args, **kwargs)

# 替换
some_library.buggy_function = fixed_function
```

### 3. 单元测试 Mock

在测试中替换数据库、网络等依赖为假数据。

```python
# 原始代码
class UserService:
    def get_user(self, user_id):
        # 实际查询数据库
        return Database.query(f"SELECT * FROM users WHERE id = {user_id}")

# 测试时 monkey patch
def test_get_user():
    # 替换数据库查询为假数据
    Database.query = lambda sql: {"id": 1, "name": "Test User"}

    service = UserService()
    user = service.get_user(1)

    assert user["name"] == "Test User"
```

### 4. 扩展类的方法

给现有类添加新方法，无需继承。

```python
# 给 str 类添加一个新方法
def remove_spaces(self):
    return self.replace(" ", "")

str.remove_spaces = remove_spaces

# 现在所有字符串都有这个方法
text = "hello world"
print(text.remove_spaces())  # 输出: "helloworld"
```

### 5. 动态功能开关

根据配置动态启用/禁用功能。

```python
import logging

class ConfigurableLogger:
    def __init__(self):
        self._original_info = logging.info
        self._enabled = True

    def enable(self):
        logging.info = self._original_info
        self._enabled = True

    def disable(self):
        logging.info = lambda *args, **kwargs: None
        self._enabled = False

logger_ctrl = ConfigurableLogger()
logger_ctrl.disable()  # 全局禁用日志
logger_ctrl.enable()   # 恢复日志
```

---

## 优缺点分析

### 优点

| 优点 | 说明 |
|------|------|
| **零侵入** | 无需修改原有代码，适合监控/调试 |
| **全局生效** | 所有调用自动使用新版本 |
| **灵活可控** | 随时可以启用、禁用、恢复 |
| **快速修复** | 可临时修复第三方库问题 |

### 缺点与风险

| 缺点 | 说明 |
|------|------|
| **隐蔽性** | 修改发生在运行时，难以发现和追踪 |
| **版本兼容** | 依赖库升级后，patch 可能失效 |
| **命名冲突** | 多人同时 patch 同一对象可能冲突 |
| **调试困难** | 行为不符合预期时，难以定位问题 |
| **维护成本** | 代码行为与源码不一致，增加理解难度 |

---

## 最佳实践

### 1. 保存原始版本（便于恢复）

```python
original = module.function

# 应用 patch
module.function = patched_function

# 完成后恢复
module.function = original
```

### 2. 使用 `@wraps` 保留元信息

```python
from functools import wraps

@wraps(original_function)
def patched_function(*args, **kwargs):
    return original_function(*args, **kwargs)
```

### 3. 添加清晰的标识

```python
patched_function.__monkey_patched__ = True
patched_function.__patch_reason__ = "监控网络请求"
```

### 4. 控制 patch 作用域

```python
# 方式一：使用上下文管理器
class MonkeyPatchContext:
    def __init__(self, obj, attr, new_func):
        self.obj = obj
        self.attr = attr
        self.new_func = new_func
        self.original = getattr(obj, attr)

    def __enter__(self):
        setattr(self.obj, self.attr, self.new_func)
        return self

    def __exit__(self, *args):
        setattr(self.obj, self.attr, self.original)

# 使用
with MonkeyPatchContext(requests.Session, 'request', monitored_request):
    requests.get("https://example.com")  # 在此范围内生效
# 离开上下文后自动恢复
```

### 5. 文档记录

```python
"""
Monkey Patch 说明：
- 目标: requests.Session.request
- 原因: 监控所有网络请求的流量和耗时
- 影响: 全局生效，所有 requests 调用
- 恢复: 调用 monitor.stop() 恢复原始方法
"""
```

---

## 完整示例：网络监控器

以下是 `monitor_wrapper.py` 中的实现模式：

```python
class NetworkMonitor:
    def __init__(self):
        self._original_request = None  # 保存原始方法
        self._patched = False           # 状态标记

    def start(self):
        """启动监控 - 应用 monkey patch"""
        if self._patched:
            return

        # 1. 保存原始方法
        self._original_request = requests.Session.request

        # 2. 创建包装函数
        @wraps(self._original_request)
        def monitored_request(session_self, method, url, **kwargs):
            # 监控逻辑...
            return self._original_request(session_self, method, url, **kwargs)

        # 3. 应用 patch
        requests.Session.request = monitored_request
        self._patched = True

    def stop(self):
        """停止监控 - 恢复原始方法"""
        if not self._patched:
            return

        # 恢复原始方法
        requests.Session.request = self._original_request
        self._patched = False
```

---

## "Monkey" 名称由来

说法不一，常见的解释：

1. **"游击补丁"演变**：从 "guerrilla patch"（游击补丁）→ "gorilla" → "monkey"
2. **形象比喻**：像猴子一样"到处乱动"，改变原有的结构
3. **玩笑命名**：程序员文化的幽默命名方式

---

## 总结

| 要点 | 建议 |
|------|------|
| **何时使用** | 监控、调试、测试 mock、临时修复 |
| **何时避免** | 生产环境核心逻辑、长期方案 |
| **关键原则** | 保存原始版本、使用 @wraps、清晰标识、及时恢复 |

Monkey patch 是一把"双刃剑"，用得好可以快速解决问题，用不好会造成难以追踪的 bug。理解原理、谨慎使用、遵循最佳实践是关键。