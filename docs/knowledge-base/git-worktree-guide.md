# Git Worktree 使用指南

## 什么是 Git Worktree？

Git Worktree 允许同一个仓库在不同目录同时 checkout 不同分支，实现并行工作而不互相干扰。

**核心特点：**
- 共享同一个 `.git` 数据库（提交历史、分支信息共享）
- 不同目录 checkout 不同分支的文件
- 磁盘占用小，切换快

---

## 常用命令

### 创建 Worktree

```bash
# 创建新分支并 checkout 到指定目录
git worktree add -b <新分支名> <目录路径> <基础分支>

# 示例：基于 main 创建 feature-a 分支到 .worktrees/feature-a 目录
git worktree add -b feature-a .worktrees/feature-a main

# 已有分支 checkout 到新目录
git worktree add <目录路径> <已有分支名>
```

### 查看所有 Worktree

```bash
git worktree list
```

输出示例：
```
/main-repo          main
/worktrees/feature-a  feature-a
/worktrees/hotfix     hotfix
```

### 删除 Worktree

```bash
# 正常删除（需要分支干净，无未提交改动）
git worktree remove <目录路径>

# 强制删除（即使有未提交改动）
git worktree remove --force <目录路径>

# 示例
git worktree remove .worktrees/feature-a
```

### 清理残留记录

```bash
# 清理已删除目录但还残留的 worktree 记录
git worktree prune
```

### 移动 Worktree

```bash
git worktree move <源路径> <目标路径>
```

---

## 核心规则

**同一个分支不能同时在多个目录 checkout**

如果 `feature-a` 分支已经在 `.worktrees/feature-a` 目录被 checkout：
- 不能在主目录执行 `git checkout feature-a`
- 想在 `feature-a` 分支工作，必须进入 `.worktrees/feature-a` 目录

---

## 常见工作流程

### 1. 并行开发多个功能

```bash
# 创建 worktree
git worktree add -b feature-a .worktrees/feature-a main
git worktree add -b feature-b .worktrees/feature-b main

# 开发功能 A
cd .worktrees/feature-a
# 写代码、测试、提交...

# 开发功能 B（独立目录，互不干扰）
cd .worktrees/feature-b
# 写代码、测试、提交...

# 合并到主分支
cd ../..
git merge feature-a
git merge feature-b

# 清理
git worktree remove .worktrees/feature-a
git worktree remove .worktrees/feature-b
```

### 2. 紧急 Hotfix（不打断当前工作）

```bash
# 正在开发 feature，突然要修 bug
# 不 stash、不切换，直接创建 worktree
git worktree add -b hotfix-123 .worktrees/hotfix main

cd .worktrees/hotfix
# 修复 bug、提交、测试...

cd ../..
git merge hotfix-123
git worktree remove .worktrees/hotfix

# 继续开发 feature（完全没被打断）
```

### 3. PR Review / 对比分支

```bash
# 创建 worktree 看 PR 代码
git worktree add .worktrees/pr-review pr-branch

# 用 IDE 打开对比
code .worktrees/pr-review   # PR 版本
code .                      # 当前版本

# 看完删除
git worktree remove .worktrees/pr-review
```

### 4. 长期维护多个版本

```bash
# 同时维护多个发布版本
git worktree add .worktrees/v1.x v1.0
git worktree add .worktrees/v2.x v2.0
git worktree add .worktrees/dev develop
```

---

## Worktree 内的提交如何合并到主分支

```bash
# 1. 在 worktree 目录提交改动
cd .worktrees/feature-a
git add .
git commit -m "完成功能 A"

# 2. 回到主目录合并
cd ../..
git merge feature-a

# 或者只合并某个提交
git cherry-pick <commit-hash>
```

---

## 目录结构示例

```
main-repo/                  # 主目录（main 分支）
├── .git/                   # Git 数据库（共享）
├── src/
├── README.md
│
.worktrees/
├── feature-a/              # feature-a 分支
│   ├── .git                # 指向主目录 .git 的链接
│   ├── src/
│   ├── README.md
│
├── hotfix/                 # hotfix 分支
│   ├── .git
│   ├── src/
│   └── README.md
```

---

## 最佳实践

1. **命名有规律**：`.worktrees/<branch-name>` 或 `.worktrees/<task-name>`
2. **用完就删**：Worktree 是临时的，完成任务就 `git worktree remove`
3. **定期清理**：`git worktree prune` 清理残留记录
4. **IDE 多窗口**：VSCode 等可以打开多个目录，各 worktree 独立窗口

---

## 对比：Worktree vs Clone

| 方式 | 仓库数据 | 工作目录 | 磁盘占用 | 提交互通 |
|------|----------|----------|----------|----------|
| `git clone` | 完全独立 | 完全独立 | 大 | 不互通 |
| `git worktree` | 共享 `.git` | 各目录独立 | 小 | 互通 |

---

## 一句话总结

**Git Worktree = 不切换分支就能同时在多个分支工作**

- 开新功能不用 stash 当前改动
- 紧急修 bug 不打断正在进行的工作
- 多人协作各干各的目录，互不干扰