# tmux 使用指南

> **版本**: tmux 3.6a  
> **安装位置**: `/opt/homebrew/bin/tmux`  
> **生成时间**: 2026-02-07

---

## 📖 tmux 是什么？

**tmux** 是一个终端复用器（Terminal Multiplexer），让你能够：

- 在一个终端窗口里开多个会话和窗格
- 断开会话后重新连接（SSH 掉线也不怕）
- 后台运行长时间任务

### 为什么使用 tmux？

- **远程开发**: SSH 掉线后，会话和任务继续运行
- **多任务管理**: 一个窗口，多个窗格，同时运行多个命令
- **持久化工作**: 关闭终端不丢失工作状态
- **高效操作**: 快捷键快速切换和管理

---

## 🚀 快速开始

### 基本操作

```bash
# 创建新会话（命名）
tmux new -s mysession

# 列出所有会话
tmux ls

# 附加到指定会话
tmux attach -t mysession
tmux a -t mysession  # 简写

# 杀掉会话
tmux kill-session -t mysession

# 杀掉所有会话
tmux kill-server
```

### 最常用的流程

```bash
# 1. 启动 tmux
tmux

# 2. 分屏（按 Ctrl+b % 垂直，Ctrl+b " 水平）

# 3. 在不同窗格中运行命令

# 4. 分离（按 Ctrl+b d）- 继续在后台运行

# 5. 重新连接
tmux attach
```

---

## ⌨️ 快捷键参考表

> **注意**: 所有快捷键需要先按 `Ctrl + b`（称为"前缀"），然后按对应的键

### 会话管理

| 操作 | 命令/快捷键 |
|------|------------|
| 新建会话 | `tmux new -s mysession` |
| 列出会话 | `tmux ls` |
| 附加会话 | `tmux attach -t mysession` |
| 简写附加 | `tmux a -t mysession` |
| 分离会话 | `Ctrl+b d` |
| 列出并切换 | `Ctrl+b s` |
| 重命名会话 | `Ctrl+b $` |

### 窗口管理

| 操作 | 快捷键 |
|------|--------|
| 新建窗口 | `Ctrl+b c` |
| 切换窗口(数字) | `Ctrl+b 0-9` |
| 下一个窗口 | `Ctrl+b n` |
| 上一个窗口 | `Ctrl+b p` |
| 列出所有窗口 | `Ctrl+b w` |
| 重命名窗口 | `Ctrl+b ,` |
| 关闭窗口 | `Ctrl+b &` |

### 窗格管理（分屏）

| 操作 | 快捷键 |
|------|--------|
| 垂直分屏 | `Ctrl+b %` |
| 水平分屏 | `Ctrl+b "` |
| 切换窗格 | `Ctrl+b 方向键` |
| 依次切换窗格 | `Ctrl+b o` |
| 关闭窗格 | `Ctrl+b x` |
| 放大/恢复窗格 | `Ctrl+b z` |
| 交换窗格(向左) | `Ctrl+b {` |
| 交换窗格(向右) | `Ctrl+b }` |
| 显示窗格编号 | `Ctrl+b q` |

### 查看历史/复制

| 操作 | 快捷键 |
|------|--------|
| 复制模式 | `Ctrl+b [` |
| 粘贴 | `Ctrl+b ]` |
| 显示所有快捷键 | `Ctrl+b ?` |
| 命令行模式 | `Ctrl+b :` |
| 显示时钟 | `Ctrl+b t` |

> 💡 **提示**: 在复制模式中：
> - 使用方向键或 Page Up/Down 滚动
> - 按空格开始复制，按回车结束复制
> - 按 `q` 退出复制模式

---

## 💡 实用示例

### 场景 1: SSH 远程开发

```bash
# 1. 创建会话
tmux new -s remote-dev

# 2. 连接到远程服务器
ssh user@server

# 3. 垂直分屏（Ctrl+b %）
# 左边编辑代码，右边运行测试

# 4. 如果 SSH 断开，重新连接
tmux attach -t remote-dev
```

### 场景 2: 开发环境分离

```bash
# 创建不同的会话用于不同项目
tmux new -s project1
tmux new -s project2

# 切换项目
tmux a -t project1
tmux a -t project2
```

### 场景 3: 后台运行长时间任务

```bash
# 1. 启动会话
tmux new -s backup

# 2. 运行长时间任务
rsync -avz /data /backup/

# 3. 分离（Ctrl+b d）- 任务继续运行

# 4. 稍后检查进度
tmux a -t backup
```

### 场景 4: 四分屏监控

```bash
# 创建新会话
tmux new -s monitor

# 按 Ctrl+b " 水平分屏
# 在上面窗格按 Ctrl+b % 垂直分屏
# 在下面窗格按 Ctrl+b % 垂直分屏

# 现在你有 4 个窗格，每个可以运行不同的监控命令：
# - 左上: top
# - 右上: htop
# - 左下: tail -f /var/log/syslog
# - 右下: watch -n 1 'df -h'
```

---

## 🔧 高级技巧

### 配置文件 (`~/.tmux.conf`)

创建或编辑 `~/.tmux.conf` 文件：

```bash
# 设置默认 shell
set -g default-shell /bin/zsh

# 设置前缀为 Ctrl+a（更顺手，避免与 Vim 冲突）
set -g prefix C-a
unbind C-b
bind C-a send-prefix

# 启用鼠标支持（可以用鼠标点击切换窗格）
set -g mouse on

# 设置窗格和窗口从 1 开始（而不是 0）
set -g base-index 1
setw -g pane-base-index 1

# 设置窗口标题
setw -g automatic-rename on
set -g set-titles on

# 更新窗口标题格式
set -g set-titles-string '#T'

# 重新加载配置文件（无需重启 tmux）
bind r source-file ~/.tmux.conf \; display "Config reloaded!"
```

### 批量操作所有窗格

```bash
# 1. 同步输入（在所有窗格中执行相同命令）
Ctrl+b :
:setw synchronize-panes on

# 2. 在一个窗格输入命令，所有窗格同步执行

# 3. 关闭同步
:setw synchronize-panes off
```

**使用场景**: 同时在多台服务器上执行相同命令

### 保存和恢复会话（插件）

使用 **tmux-resurrect** 插件可以保存会话状态：

```bash
# 1. 安装插件管理器
git clone https://github.com/tmux-plugins/tpm ~/.tmux/plugins/tpm

# 2. 在 ~/.tmux.conf 添加：
set -g @plugin 'tmux-plugins/tpm'
set -g @plugin 'tmux-plugins/tmux-resurrect'
set -g @plugin 'tmux-plugins/tmux-continuum'

# 3. 安装插件
~/.tmux/plugins/tpm/bin/install_plugins

# 4. 保存会话
Ctrl+b Ctrl+s

# 5. 恢复会话
Ctrl+b Ctrl+r

# 6. 自动保存（tmux-continuum）
set -g @continuum-restore 'on'
```

---

## ❓ 常见问题

### Q: 如何退出 tmux？
**A**: 输入 `exit` 关闭所有窗口，或按 `Ctrl+d`

### Q: tmux 里的复制模式怎么用？
**A**:
1. 按 `Ctrl+b [` 进入复制模式
2. 使用方向键或 Page Up/Down 滚动
3. 按空格开始复制标记
4. 移动到结束位置
5. 按回车结束复制
6. 按 `Ctrl+b ]` 粘贴

### Q: 如何查看历史输出？
**A**: 按 `Ctrl+b [` 进入复制模式，向上滚动查看

### Q: 如何在 tmux 里滚动？
**A**: 按 `Ctrl+b [` 进入复制模式，然后用方向键或 Page Up/Down 滚动

### Q: 窗格太多乱了怎么办？
**A**: 按 `Ctrl+b q` 显示窗格编号，然后按对应的数字快速切换

### Q: 如何在 tmux 中使用复制粘贴？
**A**:
- **macOS**: 使用系统剪贴板（需配置）
  ```bash
  # 在 ~/.tmux.conf 中添加：
  set -g default-command "reattach-to-user-namespace -l zsh"
  bind-key -T copy-mode-vi y send-keys -X copy-pipe-and-cancel "reattach-to-user-namespace pbcopy"
  ```
- **Linux**: 直接使用 tmux 的复制粘贴

### Q: 如何调整窗格大小？
**A**:
- `Ctrl+b Ctrl+方向键`: 调整窗格大小
- 需要先按 `Ctrl+b`，然后按住 `Ctrl` 再按方向键

---

## 🤖 OpenClaw tmux 技能

作为 OpenClaw，我可以通过 tmux 技能帮你：

- 创建并控制 tmux 会话
- 在后台运行交互式 CLI（如 Codex, Claude Code）
- 抓取窗格输出
- 发送按键和命令

### 示例：后台运行 Claude Code

```bash
# 创建专用 socket
SOCKET="${TMPDIR}/claude.sock"

# 创建新会话（后台运行）
tmux -S "$SOCKET" new -d -s claude -n shell

# 发送命令到会话
tmux -S "$SOCKET" send-keys -t claude:0.0 "cd ~/project && claude 'Fix the bug'" Enter

# 抓取窗格输出
tmux -S "$SOCKET" capture-pane -p -t claude:0.0 -S -100

# 列出会话
tmux -S "$SOCKET" list-sessions

# 附加到会话
tmux -S "$SOCKET" attach -t claude
```

### 示例：并行运行多个编码代理

```bash
SOCKET="${TMPDIR}/codex-army.sock"

# 创建多个会话
for i in 1 2 3; do
  tmux -S "$SOCKET" new -d -s "agent-$i"
done

# 在不同目录启动多个 Codex
tmux -S "$SOCKET" send-keys -t agent-1 "cd /tmp/project1 && codex --yolo 'Fix bug X'" Enter
tmux -S "$SOCKET" send-keys -t agent-2 "cd /tmp/project2 && codex --yolo 'Fix bug Y'" Enter
tmux -S "$SOCKET" send-keys -t agent-3 "cd /tmp/project3 && codex --yolo 'Add feature Z'" Enter

# 检查进度
for sess in agent-1 agent-2 agent-3; do
  tmux -S "$SOCKET" capture-pane -p -t "$sess" -S -3 | head -20
done
```

---

## 📚 学习资源

- **官方文档**: https://github.com/tmux/tmux/wiki
- **快速参考**: https://tmuxcheatsheet.com/
- **Awesome tmux**: https://github.com/rothgar/awesome-tmux

---

**🎯 快速记忆口诀**：

- **Ctrl+b c** - Create（创建窗口）
- **Ctrl+b d** - Detach（分离会话）
- **Ctrl+b %** - % 符号，垂直分屏
- **Ctrl+b "** - " 引号，水平分屏
- **Ctrl+b n** - Next（下一个窗口）
- **Ctrl+b p** - Previous（上一个窗口）
- **Ctrl+b ?** - 问号，显示帮助

---

*生成时间: 2026-02-07 | 版本: tmux 3.6a | 💨 Mr.Air*
