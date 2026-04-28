- 当前最热门的两个 AI agent 应用分别是：Claude code 和 OpenClaw，
- Claude code 可以自己编写代码自己纠错或者帮你纠正你代码中的问题，
- OpenClaw 是一个个人 AI 助手可以操控你的电脑干任何事，并且可以接入 IM 软件例如 telegram，discord，QQ 等进行通信

### 安装教程
#### Claude code + Open claw 安装教程
- 使用指南: https://help.aliyun.com/zh/model-studio/coding-plan
- 阿里云平台体验地址：https://bailian.console.aliyun.com/cn-beijing
- 智谱平台接入指南：https://docs.bigmodel.cn/cn/coding-plan/tool/claude
- GLM Coding 开发者社区文档：https://zhipu-ai.feishu.cn/wiki/TrlMwahsfihLrKkZsy0cpuTenCz

#### Claude code + Open claw 使用技巧 针对 Mac 电脑
- 在 ~/.claude/settings.json文件配置 Claude code使用什么模型等各种全局配置，~/.claude.json为 agent 自己自动配置的一些项目配置会不断更新，一般不用你维护
- 在 ~/.openclaw/openclaw.json中配置模型/插件/通信组件等各种全局配置，你只用管这个文件就可以了
- 记录一些 Claude code 使用的命令，/plugin 插件管理，/config各种配置模型等管理，/model glm-5切换到 glm-5模型，/compact 手动压缩上下文，一般是自动的不用管，/clear清除历史对话上下文
- 记录一些 OpenClaw 使用的命令，终端输入 openclaw config设置，终端输入 openclaw gateway restart重启网关因为时不时不稳定，openclaw doctor --fix自动修复问题，openclaw tui终端交互

#### Skills使用技巧
- /frontend-design 设计前端页面很好，避免了 AI 风格的蓝紫色前端 UI 页面
- /brainstorming   (superpowers)为AI编程助手设计的技能系统，旨在让AI遵循专业工程实践，变得更规范、可靠、可预测


#### 实战使用Claude code原则
- 一开始一般是用/init 命令为整个项目生成一个 CLAUDE.md文件
- CLAUDE.md文件不应过长，可以用@docs的方式指向想让 Agent 阅读的文档，而不是把整个细则都放到该文件