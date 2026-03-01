- 当前最热门的两个 AI agent 应用分别是：Claude code 和 Openclaw，
- Claude code 可以自己编写代码自己纠错或者帮你纠正你代码中的问题，
- Openclaw 是一个个人 AI 助手可以操控你的电脑干任何事，并且可以接入 IM 软件例如 telegram，discord，QQ 等进行通信

### 安装教程
#### Claude code + Openclaw 安装教程
- 使用指南: https://help.aliyun.com/zh/model-studio/coding-plan
- 阿里云平台体验地址：https://bailian.console.aliyun.com/cn-beijing
- 智谱平台接入指南：https://docs.bigmodel.cn/cn/coding-plan/tool/claude

#### Claude code + Openclaw 使用技巧 针对 Mac 电脑
- 在 ~/.claude/settings.json文件配置 Claude code使用什么模型等各种全局配置，~/.claude.json为 agent 自己自动配置的一些项目配置会不断更新，一般不用你维护
- 在 ~/.openclaw/openclaw.json中配置模型/插件/通信组件等各种全局配置，你只用管这个文件就可以了
- 记录一些 Claude code 使用的命令，/plugin 插件管理，/config各种配置模型等管理，/model glm-5切换到 glm-5模型，/compact 手动压缩上下文，一般是自动的不用管，/clear清除历史对话上下文
- 记录一些 Openclaw 使用的命令，终端输入 openclaw gateway restart重启网关因为时不时不稳定，openclaw doctor --fix自动修复问题，openclaw tui终端交互
