# 环境变量作用域与存储

## 环境变量的作用域

| 作用域 | 范围 | 持久性 |
|--------|------|--------|
| **当前 shell** | 仅当前终端会话 | 关闭终端即消失 |
| **子进程** | 当前 shell 及其启动的程序 | 随当前 shell 结束而消失 |
| **用户级** | 用户所有终端会话 | 持久（写入配置文件） |
| **系统级** | 所有用户 | 持久（写入系统配置） |

## 存储位置

环境变量 **存储在内存中**，不存储在文件里（除非你主动写入配置文件）。

### 不同作用域的配置方式

**1. 当前 shell（临时）**
```bash
export MY_VAR="value"      # 仅当前会话有效
```

**2. 用户级持久化**
```bash
# 写入 shell 配置文件
echo 'export PIPEGATE_JWT_SECRET="your-secret"' >> ~/.zshrc
source ~/.zshrc           # 立即生效
```

| Shell | 配置文件 |
|-------|----------|
| zsh | `~/.zshrc` |
| bash | `~/.bashrc` 或 `~/.bash_profile` |

**3. 项目级（推荐）**
```bash
# 创建 .env 文件（不会被 git 跟踪）
echo 'PIPEGATE_JWT_SECRET="your-secret"' > .env
```

Python 项目通常用 `python-dotenv` 或 `uv` 自动加载 `.env`。

**4. 系统级**
```bash
# 写入系统配置（需要 root）
echo 'PIPEGATE_JWT_SECRET="value"' | sudo tee /etc/environment
```

## 查看环境变量

```bash
echo $PIPEGATE_JWT_SECRET    # 查看单个变量
env                          # 列出所有环境变量
printenv                     # 同上
```

## 总结

| 方式 | 命令 | 作用域 | 持久性 |
|------|------|--------|--------|
| `export` | `export VAR="val"` | 当前 shell + 子进程 | 临时 |
| shell 配置文件 | `~/.zshrc` | 用户所有会话 | 持久 |
| `.env` 文件 | 项目根目录 | 项目内 | 持久 |
| `/etc/environment` | 系统配置 | 全局 | 持久 |