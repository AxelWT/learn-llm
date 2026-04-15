# Pydantic 完整使用指南

## 目录

1. [Pydantic 基础](#1-pydantic-基础)
2. [Pydantic 字段验证](#2-pydantic-字段验证)
3. [Pydantic 模型配置](#3-pydantic-模型配置)
4. [Pydantic 验证器](#4-pydantic-验证器)
5. [Pydantic 序列化](#5-pydantic-序列化)
6. [Pydantic-Settings 配置管理](#6-pydantic-settings-配置管理)
7. [常用类型](#7-常用类型)
8. [实战示例](#8-实战示例)

---

## 1. Pydantic 基础

### 1.1 安装

```bash
pip install pydantic
pip install pydantic-settings  # 配置管理
pip install "pydantic[email]"  # 邮箱验证支持
```

### 1.2 基本模型

```python
from pydantic import BaseModel

class User(BaseModel):
    id: int
    name: str
    email: str
    is_active: bool = True  # 带默认值

# 创建实例
user = User(id=1, name="张三", email="zhangsan@example.com")

# 访问属性
print(user.id)        # 1
print(user.name)      # 张三
print(user.is_active) # True

# 转换为字典
print(user.model_dump())
# {'id': 1, 'name': '张三', 'email': 'zhangsan@example.com', 'is_active': True}

# 转换为 JSON
print(user.model_dump_json())
# '{"id":1,"name":"张三","email":"zhangsan@example.com","is_active":true}'
```

### 1.3 自动类型转换

```python
from pydantic import BaseModel

class Item(BaseModel):
    id: int
    price: float
    tags: list[str]

# Pydantic 会自动转换类型
item = Item(id="123", price="99.9", tags='["food", "drink"]')
print(item.id)     # 123 (int)
print(item.price)  # 99.9 (float)
print(item.tags)   # ['food', 'drink'] (list)
```

### 1.4 从字典/JSON 创建

```python
from pydantic import BaseModel

class User(BaseModel):
    id: int
    name: str

# 从字典创建
data = {"id": 1, "name": "张三"}
user = User(**data)
user = User.model_validate(data)  # 推荐方式

# 从 JSON 创建
json_data = '{"id": 1, "name": "张三"}'
user = User.model_validate_json(json_data)
```

---

## 2. Pydantic 字段验证

### 2.1 Field 函数

```python
from pydantic import BaseModel, Field

class Product(BaseModel):
    # 基本用法
    name: str = Field(description="产品名称")

    # 默认值
    quantity: int = Field(default=0)

    # 默认工厂函数
    tags: list[str] = Field(default_factory=list)

    # 别名（输入/输出时的键名映射）
    price: float = Field(alias="product_price")

    # 数值约束
    age: int = Field(ge=0, le=150)      # 0 <= age <= 150
    score: float = Field(gt=0, lt=100)   # 0 < score < 100

    # 字符串约束
    username: str = Field(min_length=3, max_length=20)
    code: str = Field(pattern=r"^[A-Z]{3}\d{3}$")  # 正则验证

    # 示例值（用于文档）
    email: str = Field(examples=["user@example.com"])

    # 必填字段（无默认值）
    required_field: str

    # 可选字段
    optional_field: str | None = None
```

### 2.2 常用约束参数

| 参数 | 适用类型 | 说明 |
|------|----------|------|
| `default` | 所有 | 默认值 |
| `default_factory` | 所有 | 默认值工厂函数 |
| `alias` | 所有 | 字段别名 |
| `description` | 所有 | 字段描述 |
| `examples` | 所有 | 示例值 |
| `gt`, `ge` | 数值 | 大于/大于等于 |
| `lt`, `le` | 数值 | 小于/小于等于 |
| `multiple_of` | 数值 | 必须是某数的倍数 |
| `min_length`, `max_length` | 字符串/集合 | 长度限制 |
| `pattern` | 字符串 | 正则表达式 |
| `allow_inf_nan` | 浮点数 | 允许 inf/nan |

### 2.3 字段示例

```python
from pydantic import BaseModel, Field

class User(BaseModel):
    # 用户名：3-20字符，只允许字母数字下划线
    username: str = Field(
        min_length=3,
        max_length=20,
        pattern=r"^\w+$",
        description="用户名",
        examples=["zhang_san"]
    )

    # 年龄：0-150
    age: int = Field(
        ge=0,
        le=150,
        description="用户年龄"
    )

    # 余额：非负数，保留两位小数
    balance: float = Field(
        ge=0,
        description="账户余额",
        examples=[100.50]
    )

    # 标签：默认空列表
    tags: list[str] = Field(
        default_factory=list,
        max_length=10,
        description="用户标签"
    )
```

---

## 3. Pydantic 模型配置

### 3.1 model_config

```python
from pydantic import BaseModel, ConfigDict

class User(BaseModel):
    model_config = ConfigDict(
        str_strip_whitespace=True,      # 自动去除字符串空白
        str_min_length=1,               # 字符串最小长度
        str_max_length=1000,            # 字符串最大长度
        validate_assignment=True,       # 赋值时验证
        validate_default=True,          # 验证默认值
        extra="forbid",                 # 禁止额外字段
        extra="ignore",                 # 忽略额外字段
        extra="allow",                  # 允许额外字段
        populate_by_name=True,          # 允许通过字段名或别名填充
        from_attributes=True,           # 允许从 ORM 对象创建
        frozen=True,                    # 不可变模型
        use_enum_values=True,           # 使用枚举值而非枚举对象
    )

    name: str
    age: int
```

### 3.2 配置参数详解

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `str_strip_whitespace` | False | 自动去除字符串首尾空白 |
| `validate_assignment` | False | 属性赋值时是否验证 |
| `validate_default` | False | 是否验证默认值 |
| `extra` | "ignore" | 处理额外字段：ignore/forbid/allow |
| `populate_by_name` | False | 允许通过字段名填充（即使有别名） |
| `from_attributes` | False | 允许从对象属性创建（ORM支持） |
| `frozen` | False | 模型是否不可变 |
| `use_enum_values` | False | 使用枚举值而非枚举对象 |

### 3.3 配置示例

```python
from pydantic import BaseModel, ConfigDict, Field, ValidationError

# 示例1：禁止额外字段
class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str

try:
    StrictModel(name="test", extra="data")  # 报错
except ValidationError as e:
    print(e)

# 示例2：赋值时验证
class ValidateOnAssign(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    age: int = Field(ge=0)

user = ValidateOnAssign(age=10)
user.age = -1  # ValidationError: 年龄不能为负

# 示例3：不可变模型
class FrozenModel(BaseModel):
    model_config = ConfigDict(frozen=True)
    value: str

obj = FrozenModel(value="test")
obj.value = "new"  # ValidationError: 模型不可变

# 示例4：从 ORM 对象创建
class ORMUser:
    def __init__(self, name: str, email: str):
        self.name = name
        self.email = email

class UserModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    name: str
    email: str

orm_obj = ORMUser("张三", "test@example.com")
user = UserModel.model_validate(orm_obj)
```

---

## 4. Pydantic 验证器

### 4.1 field_validator 字段验证器

```python
from pydantic import BaseModel, field_validator

class User(BaseModel):
    name: str
    age: int
    email: str

    @field_validator("name")
    @classmethod
    def name_must_not_contain_space(cls, v: str) -> str:
        if " " in v:
            raise ValueError("名字不能包含空格")
        return v

    @field_validator("age")
    @classmethod
    def age_must_be_positive(cls, v: int) -> int:
        if v < 0:
            raise ValueError("年龄必须为正数")
        return v

    @field_validator("email")
    @classmethod
    def email_must_be_valid(cls, v: str) -> str:
        if "@" not in v:
            raise ValueError("邮箱格式不正确")
        return v.lower()  # 转换后返回
```

### 4.2 model_validator 模型验证器

```python
from pydantic import BaseModel, model_validator

class User(BaseModel):
    password: str
    confirm_password: str

    @model_validator(mode="after")
    def passwords_match(self) -> "User":
        if self.password != self.confirm_password:
            raise ValueError("两次密码不一致")
        return self

class DateRange(BaseModel):
    start_date: str
    end_date: str

    @model_validator(mode="after")
    def validate_dates(self) -> "DateRange":
        if self.start_date > self.end_date:
            raise ValueError("开始日期不能晚于结束日期")
        return self
```

### 4.3 验证器模式

```python
from pydantic import BaseModel, model_validator

class Model(BaseModel):
    value1: int
    value2: int

    # mode="before": 在类型转换之前执行
    @model_validator(mode="before")
    @classmethod
    def validate_before(cls, data: dict) -> dict:
        # data 是原始输入字典
        return data

    # mode="after": 在所有字段验证完成后执行
    @model_validator(mode="after")
    def validate_after(self) -> "Model":
        # self 是已验证的模型实例
        return self

    # mode="wrap": 包装整个验证过程
    @model_validator(mode="wrap")
    @classmethod
    def validate_wrap(cls, data, handler):
        # 前置处理
        result = handler(data)  # 执行默认验证
        # 后置处理
        return result
```

### 4.4 field_validator 多字段验证

```python
from pydantic import BaseModel, field_validator

class Product(BaseModel):
    name: str
    category: str

    # 验证多个字段
    @field_validator("name", "category")
    @classmethod
    def capitalize_fields(cls, v: str) -> str:
        return v.capitalize()

    # 验证所有字符串字段
    @field_validator("*")
    @classmethod
    def validate_all_string_fields(cls, v):
        if isinstance(v, str):
            return v.strip()
        return v
```

---

## 5. Pydantic 序列化

### 5.1 导出为字典

```python
from pydantic import BaseModel, Field

class User(BaseModel):
    id: int
    name: str
    password: str = Field(exclude=True)  # 导出时排除
    email: str = Field(alias="user_email")

user = User(id=1, name="张三", password="secret", user_email="test@example.com")

# 基本导出
print(user.model_dump())
# {'id': 1, 'name': '张三', 'email': 'test@example.com'}

# 包含排除字段
print(user.model_dump(exclude={"name"}))
# {'id': 1, 'email': 'test@example.com'}

# 只包含特定字段
print(user.model_dump(include={"id", "name"}))
# {'id': 1, 'name': '张三'}

# 使用别名
print(user.model_dump(by_alias=True))
# {'id': 1, 'name': '张三', 'user_email': 'test@example.com'}

# 排除未设置的字段
print(user.model_dump(exclude_unset=True))

# 排除默认值
print(user.model_dump(exclude_defaults=True))
```

### 5.2 导出为 JSON

```python
from pydantic import BaseModel

class User(BaseModel):
    id: int
    name: str

user = User(id=1, name="张三")

# 基本导出
json_str = user.model_dump_json()
# '{"id":1,"name":"张三"}'

# 格式化
json_str = user.model_dump_json(indent=2)
# {
#   "id": 1,
#   "name": "张三"
# }

# 其他参数与 model_dump 相同
json_str = user.model_dump_json(
    exclude={"name"},
    by_alias=True,
    indent=2
)
```

### 5.3 自定义序列化

```python
from pydantic import BaseModel, field_serializer
from datetime import datetime

class Event(BaseModel):
    name: str
    created_at: datetime

    @field_serializer("created_at")
    def serialize_datetime(self, dt: datetime, _info) -> str:
        return dt.strftime("%Y-%m-%d %H:%M:%S")

    @field_serializer("name", when_used="json")
    def serialize_name(self, name: str) -> str:
        return name.upper()

# when_used 可选值：
# - "always": 总是使用自定义序列化
# - "json": 仅在 JSON 序列化时使用
# - "json-unless-none": JSON 序列化时使用，但 None 值保持不变
```

### 5.4 自定义反序列化

```python
from pydantic import BaseModel, field_validator
from datetime import datetime

class Event(BaseModel):
    created_at: datetime

    @field_validator("created_at", mode="before")
    @classmethod
    def parse_datetime(cls, v):
        if isinstance(v, str):
            return datetime.strptime(v, "%Y-%m-%d %H:%M:%S")
        return v

event = Event(created_at="2024-01-15 10:30:00")
```

---

## 6. Pydantic-Settings 配置管理

### 6.1 基本使用

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    app_name: str = "MyApp"
    debug: bool = False
    database_url: str

# 从环境变量读取
# export DATABASE_URL="postgresql://localhost/db"
settings = Settings()
print(settings.database_url)
```

### 6.2 .env 文件支持

```python
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",              # .env 文件路径
        env_file_encoding="utf-8",    # 文件编码
        case_sensitive=False,         # 是否区分大小写
        extra="ignore",               # 忽略额外变量
    )

    database_url: str
    secret_key: str = Field(alias="APP_SECRET_KEY")
    debug: bool = False
```

### 6.3 配置来源优先级

优先级从高到低：

1. 命令行参数（如果启用）
2. 环境变量
3. .env 文件
4. 默认值

```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        cli_parse_args=True,  # 启用命令行参数
        env_file=".env",
    )

    port: int = 8000
    host: str = "localhost"

# 可以通过命令行覆盖
# python app.py --port 9000 --host 0.0.0.0
```

### 6.4 环境变量命名

```python
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    # 直接使用字段名作为环境变量名
    database_url: str

    # 使用别名
    db_name: str = Field(alias="DATABASE_NAME")

    # 嵌套变量（使用 __ 分隔）
    # 环境变量: REDIS__HOST=localhost
    # 环境变量: REDIS__PORT=6379
    class Redis(BaseSettings):
        host: str = "localhost"
        port: int = 6379

    redis: Redis = Redis()
```

### 6.5 .env 文件示例

```bash
# .env 文件

# 基本变量
DATABASE_URL=postgresql://user:pass@localhost/db
DEBUG=true

# 列表（JSON 格式）
ALLOWED_HOSTS=["localhost", "127.0.0.1"]

# 别名字段
DATABASE_NAME=mydb

# 嵌套配置
REDIS__HOST=localhost
REDIS__PORT=6379
```

### 6.6 SettingsConfigDict 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `env_file` | None | .env 文件路径 |
| `env_file_encoding` | "utf-8" | 文件编码 |
| `case_sensitive` | False | 是否区分大小写 |
| `cli_parse_args` | False | 是否解析命令行参数 |
| `cli_prefix` | "" | 命令行参数前缀 |
| `env_prefix` | "" | 环境变量前缀 |
| `env_nested_delimiter` | None | 嵌套变量分隔符 |
| `extra` | "ignore" | 处理额外字段 |

### 6.7 环境变量前缀

```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="APP_",  # 所有环境变量需要 APP_ 前缀
    )

    # 对应环境变量: APP_NAME
    name: str = "MyApp"

    # 对应环境变量: APP_DEBUG
    debug: bool = False

# export APP_NAME="Production App"
# export APP_DEBUG=true
```

### 6.8 嵌套配置

```python
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

class DatabaseConfig(BaseModel):
    host: str = "localhost"
    port: int = 5432
    name: str = "app_db"

class RedisConfig(BaseModel):
    host: str = "localhost"
    port: int = 6379

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_nested_delimiter="__",  # 使用 __ 分隔嵌套
    )

    database: DatabaseConfig = DatabaseConfig()
    redis: RedisConfig = RedisConfig()

# 环境变量:
# DATABASE__HOST=localhost
# DATABASE__PORT=5432
# DATABASE__NAME=mydb
# REDIS__HOST=127.0.0.1
# REDIS__PORT=6380
```

---

## 7. 常用类型

### 7.1 基本类型

```python
from pydantic import BaseModel

class Model(BaseModel):
    a: int              # 整数
    b: float            # 浮点数
    c: str              # 字符串
    d: bool             # 布尔值
    e: bytes            # 字节
```

### 7.2 特殊字符串类型

```python
from pydantic import BaseModel, SecretStr, SecretBytes

class Model(BaseModel):
    # 敏感字符串（打印时自动隐藏）
    password: SecretStr

    # 敏感字节
    api_key: SecretBytes

m = Model(password="secret123", api_key=b"key123")

print(m.password)              # SecretStr('**********')
print(m.password.get_secret_value())  # secret123

# JSON 导出时自动隐藏
print(m.model_dump_json())  # {"password":"**********","api_key":"********"}
```

### 7.3 日期时间类型

```python
from datetime import datetime, date, time, timedelta
from pydantic import BaseModel

class Model(BaseModel):
    dt: datetime        # 日期时间
    d: date             # 日期
    t: time             # 时间
    td: timedelta       # 时间间隔

# 自动解析 ISO 格式
m = Model(
    dt="2024-01-15T10:30:00",
    d="2024-01-15",
    t="10:30:00",
    td="1 day"
)
```

### 7.4 集合类型

```python
from typing import List, Set, FrozenSet, Dict, Tuple
from pydantic import BaseModel

class Model(BaseModel):
    # 列表
    items: list[str]
    items2: List[str]

    # 集合（自动去重）
    unique_items: set[int]

    # 字典
    mapping: dict[str, int]

    # 元组
    point: tuple[float, float]

    # 固定长度元组
    record: tuple[str, int, float]
```

### 7.5 可选类型

```python
from typing import Optional
from pydantic import BaseModel

class Model(BaseModel):
    # Python 3.10+ 语法
    a: str | None = None

    # 传统语法
    b: Optional[str] = None
```

### 7.6 联合类型

```python
from typing import Union
from pydantic import BaseModel

class Model(BaseModel):
    # Python 3.10+ 语法
    value: int | str

    # 传统语法
    value2: Union[int, str]

# 自动判断类型
m = Model(value=123)      # int
m = Model(value="hello")  # str
```

### 7.7 枚举类型

```python
from enum import Enum, IntEnum
from pydantic import BaseModel, ConfigDict

class Color(str, Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"

class Priority(IntEnum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3

class Model(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    color: Color
    priority: Priority

m = Model(color="red", priority=2)
print(m.color)      # Color.RED 或 "red"（取决于 use_enum_values）
print(m.priority)   # Priority.MEDIUM 或 2
```

### 7.8 自定义类型

```python
from pydantic import BaseModel, GetCoreSchemaHandler
from pydantic_core import core_schema

class PhoneNumber:
    def __init__(self, value: str):
        self.value = value

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler: GetCoreSchemaSchemaHandler):
        return core_schema.with_info_plain_validator_function(
            cls.validate
        )

    @classmethod
    def validate(cls, value, _info):
        if not isinstance(value, str):
            raise TypeError("必须是字符串")
        if not value.startswith("+"):
            raise ValueError("电话号码必须以+开头")
        return cls(value)

class User(BaseModel):
    phone: PhoneNumber

user = User(phone="+8613800138000")
```

### 7.9 严格模式

```python
from pydantic import BaseModel, ConfigDict

class StrictModel(BaseModel):
    model_config = ConfigDict(strict=True)

    # strict=True 时不进行自动类型转换
    age: int

# 严格模式下，字符串不会自动转换为整数
StrictModel(age=25)      # OK
StrictModel(age="25")    # ValidationError in strict mode
```

---

## 8. 实战示例

### 8.1 Web API 请求/响应模型

```python
from pydantic import BaseModel, Field, EmailStr
from datetime import datetime
from typing import Optional

# 请求模型
class UserCreateRequest(BaseModel):
    username: str = Field(min_length=3, max_length=20, pattern=r"^\w+$")
    email: EmailStr
    password: str = Field(min_length=8, max_length=100)
    age: Optional[int] = Field(None, ge=0, le=150)

# 响应模型
class UserResponse(BaseModel):
    id: int
    username: str
    email: str
    created_at: datetime

    class Config:
        from_attributes = True  # 支持从 ORM 对象创建

# 使用示例
def create_user(request: UserCreateRequest) -> UserResponse:
    # 验证通过，处理业务逻辑
    user = save_to_database(request)
    return UserResponse.model_validate(user)
```

### 8.2 配置管理示例

```python
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

# 项目根目录
BASE_DIR = Path(__file__).parent.parent

class DatabaseSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="DB_")

    host: str = "localhost"
    port: int = 5432
    name: str = "app"
    user: str = "postgres"
    password: SecretStr

    @property
    def url(self) -> str:
        return f"postgresql://{self.user}:{self.password.get_secret_value()}@{self.host}:{self.port}/{self.name}"

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        case_sensitive=False,
    )

    # 应用配置
    app_name: str = "MyApp"
    debug: bool = False

    # 安全配置
    secret_key: SecretStr
    allowed_hosts: list[str] = ["localhost"]

    # 嵌套配置
    database: DatabaseSettings = DatabaseSettings()

# 使用
settings = Settings()
print(settings.database.url)
```

### 8.3 数据验证管道

```python
from pydantic import BaseModel, field_validator, model_validator
from typing import Optional
import re

class UserRegistration(BaseModel):
    username: str
    email: str
    password: str
    confirm_password: str
    phone: Optional[str] = None

    @field_validator("username")
    @classmethod
    def validate_username(cls, v: str) -> str:
        if not re.match(r"^[a-zA-Z0-9_]+$", v):
            raise ValueError("用户名只能包含字母、数字和下划线")
        if len(v) < 3:
            raise ValueError("用户名至少3个字符")
        return v.lower()

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        if "@" not in v:
            raise ValueError("无效的邮箱地址")
        return v.lower()

    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("密码至少8个字符")
        if not re.search(r"[A-Z]", v):
            raise ValueError("密码必须包含大写字母")
        if not re.search(r"[a-z]", v):
            raise ValueError("密码必须包含小写字母")
        if not re.search(r"\d", v):
            raise ValueError("密码必须包含数字")
        return v

    @field_validator("phone")
    @classmethod
    def validate_phone(cls, v: Optional[str]) -> Optional[str]:
        if v and not re.match(r"^\+?\d{10,15}$", v):
            raise ValueError("无效的电话号码")
        return v

    @model_validator(mode="after")
    def passwords_match(self) -> "UserRegistration":
        if self.password != self.confirm_password:
            raise ValueError("两次密码输入不一致")
        return self

# 使用
try:
    user = UserRegistration(
        username="TestUser",
        email="test@example.com",
        password="Password123",
        confirm_password="Password123"
    )
except ValueError as e:
    print(f"验证失败: {e}")
```

### 8.4 动态模型

```python
from pydantic import create_model, Field

# 动态创建模型
DynamicModel = create_model(
    "DynamicModel",
    name=(str, Field(description="名称")),
    age=(int, Field(default=0, ge=0)),
    email=(str | None, None),
)

# 使用动态模型
instance = DynamicModel(name="张三", age=25)
print(instance.model_dump())

# 从字典创建动态模型
fields = {
    "field1": (str, ...),  # ... 表示必填
    "field2": (int, Field(default=10)),
}
Model = create_model("CustomModel", **fields)
```

### 8.5 与 FastAPI 集成

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, EmailStr

app = FastAPI()

class UserCreate(BaseModel):
    username: str = Field(min_length=3, max_length=20)
    email: EmailStr
    password: str = Field(min_length=8)

class UserResponse(BaseModel):
    id: int
    username: str
    email: str

@app.post("/users", response_model=UserResponse)
async def create_user(user: UserCreate):
    # Pydantic 自动验证请求体
    # 创建用户逻辑
    return {"id": 1, "username": user.username, "email": user.email}

@app.get("/users/{user_id}", response_model=UserResponse)
async def get_user(user_id: int):
    # response_model 自动过滤敏感字段
    return {"id": user_id, "username": "test", "email": "test@example.com"}
```

---

## 常见问题

### Q1: 如何跳过验证？

```python
# 使用 model_construct 跳过验证
user = User.model_construct(id=1, name="test")
```

### Q2: 如何获取字段信息？

```python
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

# 获取所有字段
print(User.model_fields)

# 获取单个字段信息
print(User.model_fields["name"])
```

### Q3: 如何处理未知字段？

```python
from pydantic import BaseModel, ConfigDict

class Model(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str

m = Model(name="test", unknown="value")
print(m.name)      # test
print(m.unknown)   # value (额外字段)
```

### Q4: 如何实现深拷贝？

```python
user_copy = user.model_copy(deep=True)
```

### Q5: 如何处理循环引用？

```python
from typing import Optional
from pydantic import BaseModel

class Node(BaseModel):
    value: int
    next: Optional["Node"] = None  # 使用字符串注解

# 或在文件开头添加:
from __future__ import annotations
```

---

## 参考资料

- [Pydantic 官方文档](https://docs.pydantic.dev/)
- [Pydantic-Settings 文档](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)
- [Pydantic GitHub](https://github.com/pydantic/pydantic)