# Python 经典技巧与用法

以下是行业内广泛了解的 Python 经典技巧。

---

## 1. 装饰器（Decorator）

最经典的元编程技巧，修改函数行为而不改变源码。

```python
# 基本用法
@timer
def slow_function():
    time.sleep(1)

# 等价于
slow_function = timer(slow_function)

# 带参数的装饰器
@retry(max_attempts=3)
def unstable_api_call():
    ...

# 常见内置装饰器
@property      # getter
@classmethod   # 类方法
@staticmethod  # 静态方法
```

**典型应用**：日志记录、性能测量、缓存、权限验证、重试机制。

---

## 2. 上下文管理器（Context Manager）

资源自动管理，确保正确释放。

```python
# 基本用法
with open('file.txt') as f:
    data = f.read()
# 文件自动关闭

# 自定义上下文管理器（类方式）
class Timer:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        print(f"耗时: {time.time() - self.start:.2f}s")

with Timer():
    do_something()

# 装饰器写法（更简洁）
from contextlib import contextmanager

@contextmanager
def timer():
    start = time.time()
    yield
    print(f"耗时: {time.time() - start:.2f}s")

with timer():
    do_something()
```

**典型应用**：文件操作、数据库连接、锁管理、临时状态切换。

---

## 3. 生成器（Generator）

惰性计算，节省内存。

```python
# 普通列表 - 占用大量内存
numbers = [i * 2 for i in range(1000000)]

# 生成器表达式 - 惰性计算
numbers = (i * 2 for i in range(1000000))

# yield 函数
def fibonacci():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

for num in fibonacci():
    if num > 100:
        break
    print(num)

# 处理大文件
def read_large_file(filepath):
    with open(filepath) as f:
        for line in f:
            yield line.strip()

# 不一次性加载整个文件
for line in read_large_file('huge_data.txt'):
    process(line)
```

**典型应用**：大数据处理、无限序列、流式数据处理。

---

## 4. 魔法方法（Magic Methods / Dunder Methods）

自定义类的行为，Python 风格的核心。

```python
class Vector:
    def __init__(self, x, y):
        self.x, self.y = x, y

    # 字符串表示
    def __str__(self):
        return f"Vector({self.x}, {self.y})"

    def __repr__(self):
        return f"Vector({self.x}, {self.y})"

    # 运算符重载
    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar):
        return Vector(self.x * scalar, self.y * scalar)

    # 长度、比较
    def __len__(self):
        return int((self.x**2 + self.y**2) ** 0.5)

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y

    # 可调用
    def __call__(self, scale):
        return Vector(self.x * scale, self.y * scale)

    # 索引访问
    def __getitem__(self, index):
        return [self.x, self.y][index]

v1 = Vector(1, 2)
v2 = Vector(3, 4)

print(v1 + v2)      # Vector(4, 6)
print(v1 * 10)      # Vector(10, 20)
print(v1 == v2)     # False
print(len(v1))      # 2
print(v1(5))        # Vector(5, 10)  (可调用)
print(v1[0])        # 1 (索引访问)
```

**常用魔法方法**：

| 方法 | 用途 |
|------|------|
| `__init__` | 构造函数 |
| `__str__` / `__repr__` | 字符串表示 |
| `__eq__` / `__lt__` | 比较运算 |
| `__add__` / `__mul__` | 算术运算 |
| `__len__` | 长度 |
| `__getitem__` / `__setitem__` | 索引访问 |
| `__call__` | 可调用 |
| `__iter__` | 迭代器 |
| `__enter__` / `__exit__` | 上下文管理器 |

---

## 5. 鸭子类型（Duck Typing）

"如果它走起来像鸭子，叫起来像鸭子，那它就是鸭子。"

不检查类型，只检查行为。

```python
# 不关心对象是什么类型，只关心它有什么方法
def process(obj):
    # 只要 obj 有 read() 方法就行
    return obj.read()

# File、StringIO、HTTPResponse 都可以传入
# 无需继承同一个基类

# 实际应用示例
class Dog:
    def speak(self):
        return "汪汪"

class Cat:
    def speak(self):
        return "喵喵"

class Robot:
    def speak(self):
        return "哔哔"

def make_sound(animal):
    # 不检查类型，直接调用
    print(animal.speak())

make_sound(Dog())    # 汪汪
make_sound(Cat())    # 喵喵
make_sound(Robot())  # 唔唔
```

**核心思想**：关注接口而非类型，代码更灵活、更通用。

---

## 6. `*args` 和 `**kwargs`

可变参数，灵活的函数签名。

```python
def func(*args, **kwargs):
    print(f"位置参数: {args}")      # 元组
    print(f"关键字参数: {kwargs}")  # 字典

func(1, 2, 3, name="test", value=100)
# 位置参数: (1, 2, 3)
# 关键字参数: {'name': 'test', 'value': 100}

# 解包传递
args = (1, 2, 3)
kwargs = {'name': 'test', 'value': 100}
func(*args, **kwargs)

# 结合固定参数
def combined(required, *args, default=None, **kwargs):
    print(f"必需参数: {required}")
    print(f"位置参数: {args}")
    print(f"默认参数: {default}")
    print(f"关键字参数: {kwargs}")

combined("必须", 1, 2, default="默认值", extra="额外")
# 必需参数: 必须
# 位置参数: (1, 2)
# 默认参数: 默认值
# 关键字参数: {'extra': '额外'}
```

**典型应用**：包装函数、代理调用、灵活的 API 设计。

---

## 7. 元类（Metaclass）

控制类的创建过程，高级元编程。

```python
# 示例 1：禁止类被继承
class FinalMeta(type):
    def __new__(cls, name, bases, attrs):
        for base in bases:
            if isinstance(base, FinalMeta):
                raise TypeError(f"{base.__name__} 不能被继承")
        return super().__new__(cls, name, bases, attrs)

class Base(metaclass=FinalMeta):
    pass

# 尝试继承会报错
class Child(Base):  # TypeError: Base 不能被继承
    pass


# 示例 2：自动注册所有子类
class RegistryMeta(type):
    registry = {}

    def __new__(cls, name, bases, attrs):
        new_class = super().__new__(cls, name, bases, attrs)
        cls.registry[name] = new_class
        return new_class

class Plugin(metaclass=RegistryMeta):
    pass

class AudioPlugin(Plugin):
    pass

class VideoPlugin(Plugin):
    pass

print(RegistryMeta.registry)
# {'Plugin': <class 'Plugin'>, 'AudioPlugin': <class 'AudioPlugin'>, 'VideoPlugin': <class 'VideoPlugin'>}


# 示例 3：自动添加方法
class AutoMethodMeta(type):
    def __new__(cls, name, bases, attrs):
        # 自动添加一个 greeting 方法
        attrs['greeting'] = lambda self: f"Hello from {self.__class__.__name__}"
        return super().__new__(cls, name, bases, attrs)

class MyClass(metaclass=AutoMethodMeta):
    pass

obj = MyClass()
print(obj.greeting())  # Hello from MyClass
```

**典型应用**：ORM 框架（Django）、插件系统、抽象基类（ABC）、单例模式。

---

## 8. 描述符（Descriptor）

属性访问控制，`@property` 的底层原理。

```python
class ValidatedAttribute:
    """验证属性值的描述符"""

    def __init__(self, min_value=0, max_value=100):
        self.min_value = min_value
        self.max_value = max_value
        self.storage_name = None

    def __set_name__(self, owner, name):
        """设置属性的存储名称"""
        self.storage_name = f"_{name}"

    def __get__(self, obj, objtype=None):
        """获取属性值"""
        if obj is None:
            return self
        return getattr(obj, self.storage_name, None)

    def __set__(self, obj, value):
        """设置属性值（带验证）"""
        if not isinstance(value, (int, float)):
            raise TypeError("值必须是数字")
        if value < self.min_value:
            raise ValueError(f"值不能小于 {self.min_value}")
        if value > self.max_value:
            raise ValueError(f"值不能大于 {self.max_value}")
        setattr(obj, self.storage_name, value)


class Person:
    age = ValidatedAttribute(min_value=0, max_value=150)
    salary = ValidatedAttribute(min_value=0, max_value=1000000)

    def __init__(self, age, salary):
        self.age = age
        self.salary = salary


p = Person(25, 50000)
print(p.age)      # 25

p.age = 30        # 正常
p.age = -1        # ValueError: 值不能小于 0
p.age = 200       # ValueError: 值不能大于 150
p.age = "twenty"  # TypeError: 值必须是数字
```

**描述符三方法**：

| 方法 | 触发时机 |
|------|----------|
| `__get__` | 访问属性时 |
| `__set__` | 设置属性时 |
| `__delete__` | 删除属性时 |

**典型应用**：`@property`、ORM 字段验证、类型检查、惰性计算。

---

## 9. `__slots__`

限制属性、节省内存。

```python
# 普通类 - 动态属性字典，内存开销大
class Normal:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

# slots 类 - 固定属性，内存节省
class Optimized:
    __slots__ = ['x', 'y', 'z']

    def __init__(self, x, y, z):
        self.x, self.y, self.z = x, y, z


# 内存对比（创建 100 万个对象）
import sys

n = Normal(1, 2, 3)
o = Optimized(1, 2, 3)

print(sys.getsizeof(n))   # ~56 bytes (有 __dict__)
print(sys.getsizeof(o))   # ~16 bytes (无 __dict__)

# 限制动态添加属性
o = Optimized(1, 2, 3)
o.new_attr = 100  # AttributeError: 'Optimized' object has no attribute 'new_attr'

# 但类级别的方法仍然可以添加
Optimized.new_method = lambda self: self.x + self.y
print(o.new_method())  # 3
```

**优点**：
- 节省内存（约 40-60%）
- 更快的属性访问
- 防止意外添加属性

**缺点**：
- 无法动态添加实例属性
- 影响 `__dict__` 相关功能
- 继承时需要注意

**典型应用**：大量对象的数据类、性能优化、属性限制。

---

## 10. `functools` 工具箱

实用的函数工具。

```python
from functools import (
    wraps,       # 保留元信息
    partial,     # 偏函数
    lru_cache,   # 缓存
    reduce,      # 归约
    singledispatch,  # 单分派泛型函数
)

# 1. wraps - 保留被包装函数的元信息
def decorator(func):
    @wraps(func)  # 保留 __name__, __doc__ 等
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper

@decorator
def my_func():
    """这是文档"""
    pass

print(my_func.__name__)  # my_func（不是 wrapper）
print(my_func.__doc__)   # 这是文档


# 2. partial - 固定部分参数，创建新函数
def power(base, exp):
    return base ** exp

square = partial(power, exp=2)
cube = partial(power, exp=3)

print(square(5))  # 25
print(cube(5))    # 125


# 3. lru_cache - 自动缓存结果
@lru_cache(maxsize=128)
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# 第一次计算 fib(100) 需要时间
print(fibonacci(100))  # 快速

# 第二次直接返回缓存
print(fibonacci(100))  # 更快

# 查看缓存统计
print(fibonacci.cache_info())
# CacheInfo(hits=1, misses=1, maxsize=128, currsize=1)


# 4. reduce - 归约操作
from functools import reduce

numbers = [1, 2, 3, 4, 5]
result = reduce(lambda x, y: x + y, numbers)
print(result)  # 15

# 带初始值
result = reduce(lambda x, y: x + y, numbers, 10)
print(result)  # 25


# 5. singledispatch - 根据参数类型选择不同实现
@singledispatch
def process(value):
    raise NotImplementedError("不支持此类型")

@process.register(int)
def _(value):
    return f"处理整数: {value}"

@process.register(str)
def _(value):
    return f"处理字符串: {value}"

@process.register(list)
def _(value):
    return f"处理列表，长度: {len(value)}"

print(process(42))       # 处理整数: 42
print(process("hello"))  # 处理字符串: hello
print(process([1,2,3]))  # 处理列表，长度: 3
```

---

## 11. 数据类（Dataclass）

Python 3.7+ 自动生成常用方法。

```python
from dataclasses import dataclass, field

@dataclass
class Product:
    name: str
    price: float
    quantity: int = 1
    tags: list = field(default_factory=list)

    def total(self):
        return self.price * self.quantity


p1 = Product("Apple", 5.0)
p2 = Product("Apple", 5.0)

print(p1 == p2)      # True（自动生成 __eq__）
print(p1)            # Product(name='Apple', price=5.0, quantity=1, tags=[])
print(p1.total())    # 5.0


# 更多选项
@dataclass(frozen=True)  # 不可变
class Point:
    x: int
    y: int

p = Point(1, 2)
p.x = 3  # FrozenInstanceError


@dataclass(order=True)  # 自动生成比较方法
class Score:
    value: int

s1 = Score(10)
s2 = Score(20)
print(s1 < s2)  # True
print(sorted([s2, s1]))  # [Score(value=10), Score(value=20)]
```

**自动生成的方法**：

| 参数 | 生成方法 |
|------|----------|
| 默认 | `__init__`, `__repr__`, `__eq__` |
| `order=True` | `__lt__`, `__le__`, `__gt__`, `__ge__` |
| `frozen=True` | 阻止修改，生成 `__hash__` |

---

## 12. 命名元组（Named Tuple）

轻量级、不可变的数据结构。

```python
from collections import namedtuple

# 定义
Point = namedtuple('Point', ['x', 'y'])
Person = namedtuple('Person', 'name age city')
RGB = namedtuple('RGB', 'red green blue')

# 使用
p = Point(3, 4)
print(p.x, p.y)     # 3 4
print(p[0])         # 3（可索引）
print(p[-1])        # 4

person = Person("Alice", 30, "NYC")
print(person.name)  # Alice
print(person.age)   # 30

# 不可变
person.age = 31  # AttributeError!

# 解包
name, age, city = person

# 转换为字典
print(person._asdict())  # {'name': 'Alice', 'age': 30, 'city': 'NYC'}

# 替换某字段（返回新对象）
new_person = person._replace(age=31)
print(new_person)  # Person(name='Alice', age=31, city='NYC')
```

**优点**：
- 内存高效（类似普通元组）
- 可读性好（字段名）
- 不可变（安全）
- 支持解包和索引

---

## 13. 闭包（Closure）

函数捕获外部变量，实现状态保持。

```python
# 基本闭包
def outer(x):
    def inner(y):
        return x + y  # inner 捕获了 outer 的 x
    return inner

add5 = outer(5)
add10 = outer(10)

print(add5(3))   # 8  (5 + 3)
print(add10(3))  # 13 (10 + 3)


# 计数器闭包
def counter():
    count = 0

    def increment():
        nonlocal count  # 修改外部变量
        count += 1
        return count

    return increment

c = counter()
print(c())  # 1
print(c())  # 2
print(c())  # 3

# 每个 counter() 创建独立的闭包
c2 = counter()
print(c2())  # 1（独立计数）


# 配置闭包
def make_multiplier(factor):
    def multiply(x):
        return x * factor
    return multiply

double = make_multiplier(2)
triple = make_multiplier(3)

print(double(5))   # 10
print(triple(5))   # 15


# 实际应用：带状态的函数工厂
def make_cache():
    cache = {}

    def get_or_compute(key, compute_func):
        if key not in cache:
            cache[key] = compute_func()
        return cache[key]

    def clear():
        cache.clear()

    return get_or_compute, clear

get, clear = make_cache()
```

**典型应用**：回调函数、状态保持、工厂函数、部分应用。

---

## 14. 列表推导式与字典推导式

简洁的数据转换。

```python
# 列表推导式
squares = [x**2 for x in range(10)]
# [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

# 条件过滤
positives = [x for x in numbers if x > 0]

# 条件表达式
labels = ['even' if x % 2 == 0 else 'odd' for x in range(10)]

# 嵌套推导式
matrix = [[i*j for j in range(5)] for i in range(3)]
# [[0, 0, 0, 0, 0], [0, 1, 2, 3, 4], [0, 2, 4, 6, 8]]

# 展平嵌套列表
flat = [x for row in matrix for x in row]


# 字典推导式
scores = {name: score for name, score in data}

# 键值互换
swapped = {v: k for k, v in original.items()}

# 条件过滤
filtered = {k: v for k, v in data.items() if v > 100}


# 集合推导式
unique_lengths = {len(word) for word in words}


# 生成器推导式（惰性）
lazy_squares = (x**2 for x in range(1000000))  # 不立即计算
```

---

## 15. 模块级单例

Python 模块天然是单例。

```python
# singleton.py
class Database:
    def __init__(self):
        self.connection = create_connection()
        self.cache = {}

    def query(self, sql):
        # 查询逻辑
        pass

# 模块级实例 - 只创建一次
db = Database()


# 其他文件导入
from singleton import db

# 无论导入多少次，都是同一个实例
db.query("SELECT * FROM users")


# 另一种写法：类内部实现
class Singleton:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

s1 = Singleton()
s2 = Singleton()
print(s1 is s2)  # True
```

**优点**：简单、线程安全（模块加载时创建）、无需额外代码。

---

## 16. 动态导入

运行时按需加载模块。

```python
import importlib

# 方式 1：import_module（类似 import 语句）
module = importlib.import_module('os.path')
print(module.join('a', 'b'))

# 动态导入不同模块
module_name = 'json'  # 从配置或用户输入获取
module = importlib.import_module(module_name)
module.dumps({'key': 'value'})

# 方式 2：从文件路径导入
import importlib.util

spec = importlib.util.spec_from_file_location(
    'my_module',
    '/path/to/script.py'
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

# 调用模块中的函数
module.some_function()

# 添加到 sys.modules（使其成为正式模块）
import sys
sys.modules['my_module'] = module

# 之后可以正常 import
import my_module


# 方式 3：重新加载已导入的模块（开发时有用）
import my_module
importlib.reload(my_module)  # 重新执行模块代码
```

**典型应用**：插件系统、延迟加载、配置驱动的模块选择、脚本加载器。

---

## 总结对比表

| 技巧 | 用途 | 经典程度 | 学习难度 |
|------|------|----------|----------|
| **装饰器** | 函数增强、日志、缓存 | ⭐⭐⭐⭐⭐ | 中等 |
| **上下文管理器** | 资源管理、自动清理 | ⭐⭐⭐⭐⭐ | 简单 |
| **生成器** | 惰性计算、大数据处理 | ⭐⭐⭐⭐⭐ | 中等 |
| **魔法方法** | 自定义类行为、运算符重载 | ⭐⭐⭐⭐⭐ | 中等 |
| **鸭子类型** | 动态类型核心思想 | ⭐⭐⭐⭐⭐ | 简单 |
| **`*args/kwargs`** | 灵活参数、包装函数 | ⭐⭐⭐⭐ | 简单 |
| **元类** | 高级元编程、类创建控制 | ⭐⭐⭐ | 困难 |
| **描述符** | 属性访问控制、验证 | ⭐⭐⭐ | 困难 |
| **`__slots__`** | 内存优化、属性限制 | ⭐⭐⭐ | 简单 |
| **functools** | 函数工具、缓存、偏函数 | ⭐⭐⭐⭐ | 简单 |
| **数据类** | 数据容器、自动方法生成 | ⭐⭐⭐⭐ | 简单 |
| **命名元组** | 轻量数据结构 | ⭐⭐⭐⭐ | 简单 |
| **闭包** | 状态保持、函数工厂 | ⭐⭐⭐⭐ | 中等 |
| **推导式** | 简洁数据转换 | ⭐⭐⭐⭐⭐ | 简单 |
| **模块单例** | 单例模式实现 | ⭐⭐⭐ | 简单 |
| **动态导入** | 按需加载模块 | ⭐⭐⭐ | 中等 |

---

## 学习建议

### 初学者优先掌握
1. 装饰器
2. 上下文管理器
3. 生成器
4. 推导式
5. `*args/kwargs`

### 进阶掌握
1. 魔法方法
2. 数据类
3. functools 工具
4. 闭包

### 高级掌握
1. 元类
2. 描述符
3. `__slots__`
4. 动态导入

---