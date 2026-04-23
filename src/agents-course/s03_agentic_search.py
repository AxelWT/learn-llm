"""
RAG Agent 演示脚本
使用 LangGraph 构建一个具有多种工具的智能代理

工具包括：
1. DuckDuckGo 搜索工具 - 用于网络搜索
2. 天气信息工具 - 用于获取天气信息（虚拟数据）
3. HuggingFace Hub 统计工具 - 用于获取模型下载统计
"""

print("=" * 60)
print("步骤 1: 初始化 DuckDuckGo 搜索工具")
print("=" * 60)

# 导入 DuckDuckGo 搜索工具
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_openai import ChatOpenAI
from langchain_core.tools import Tool

# 创建搜索工具实例（原始工具）
duckduckgo_search = DuckDuckGoSearchRun()


# 创建包装函数，添加调用时的打印功能
def search_with_logging(query: str) -> str:
    """
    DuckDuckGo 搜索工具的包装函数，添加调用日志

    参数:
        query: 搜索查询字符串

    返回:
        搜索结果字符串
    """
    print("\n" + "-" * 50)
    print("🔧 [工具调用] DuckDuckGo 搜索工具")
    print(f"   📥 请求参数: {query}")

    # 执行实际搜索
    result = duckduckgo_search.invoke(query)

    print(f"   📤 返回结果: {result[:200]}..." if len(result) > 200 else f"   📤 返回结果: {result}")
    print("-" * 50)

    return result


# 将包装函数封装为 LangChain 工具
search_tool = Tool(
    name="duckduckgo_search",
    func=search_with_logging,
    description="A search tool using DuckDuckGo to find information on the web."
)

# 测试搜索工具
print("\n>>> 测试搜索工具：查询法国现任总统")
results = search_tool.invoke("Who's the current President of France?")
print(f"搜索结果: {results}")

print("\n" + "=" * 60)
print("步骤 2: 创建天气信息工具（虚拟数据）")
print("=" * 60)

# 导入工具相关模块
from langchain_core.tools import Tool
import random


def get_weather_info(location: str) -> str:
    """
    获取指定地点的天气信息（虚拟数据）

    参数:
        location: 地点名称

    返回:
        天气信息字符串，包含天气状况和温度
    """
    # 打印工具调用信息
    print("\n" + "-" * 50)
    print("🔧 [工具调用] 天气信息工具")
    print(f"   📥 请求参数: location='{location}'")

    # 虚拟天气数据：包含不同天气状况和对应温度
    weather_conditions = [
        {"condition": "Rainy", "temp_c": 15},  # 雨天，15摄氏度
        {"condition": "Clear", "temp_c": 25},  # 晴天，25摄氏度
        {"condition": "Windy", "temp_c": 20}  # 大风，20摄氏度
    ]
    # 随机选择一种天气状况
    data = random.choice(weather_conditions)
    result = f"Weather in {location}: {data['condition']}, {data['temp_c']}°C"

    # 打印返回结果
    print(f"   📤 返回结果: {result}")
    print("-" * 50)

    return result


# 将函数封装为 LangChain 工具
# Tool 类用于将普通函数转换为可被 Agent 调用的工具
weather_info_tool = Tool(
    name="get_weather_info",  # 工具名称
    func=get_weather_info,  # 工具函数
    description="Fetches dummy weather information for a given location."  # 工具描述
)

print(">>> 天气信息工具已创建")
print(f"    工具名称: {weather_info_tool.name}")
print(f"    工具描述: {weather_info_tool.description}")

print("\n" + "=" * 60)
print("步骤 3: 创建 HuggingFace Hub 统计工具")
print("=" * 60)

# 导入 HuggingFace Hub 相关模块
from langchain_core.tools import Tool
from huggingface_hub import list_models


def get_hub_stats(author: str) -> str:
    """
    获取指定作者在 HuggingFace Hub 上下载量最高的模型

    参数:
        author: HuggingFace 上的作者/组织名称

    返回:
        模型统计信息字符串
    """
    # 打印工具调用信息
    print("\n" + "-" * 50)
    print("🔧 [工具调用] HuggingFace Hub 统计工具")
    print(f"   📥 请求参数: author='{author}'")

    try:
        # 列出指定作者的模型，按下载次数降序排序，取第一个
        models = list(list_models(author=author, sort="downloads", direction=-1, limit=1))

        if models:
            model = models[0]
            result = f"The most downloaded model by {author} is {model.id} with {model.downloads:,} downloads."
        else:
            result = f"No models found for author {author}."
    except Exception as e:
        result = f"Error fetching models for {author}: {str(e)}"

    # 打印返回结果
    print(f"   📤 返回结果: {result}")
    print("-" * 50)

    return result


# 将函数封装为 LangChain 工具
hub_stats_tool = Tool(
    name="get_hub_stats",  # 工具名称
    func=get_hub_stats,  # 工具函数
    description="Fetches the most downloaded model from a specific author on the Hugging Face Hub."  # 工具描述
)

print(">>> HuggingFace Hub 统计工具已创建")
print(f"    工具名称: {hub_stats_tool.name}")

# 测试工具：获取 Facebook 下载量最高的模型
print("\n>>> 测试工具：获取 Facebook 下载量最高的模型")
print(hub_stats_tool.invoke("facebook"))

print("\n" + "=" * 60)
print("步骤 4: 配置大语言模型并绑定工具")
print("=" * 60)

# 导入 LangGraph 相关模块
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition

# 配置大语言模型（使用阿里云千问）
# ChatOpenAI 类可以通过 OpenAI 兼容接口连接到各种模型服务
llm = ChatOpenAI(
    model="qwen-plus",  # 使用千问 plus 模型
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 阿里云 DashScope API 地址
    api_key="sk-25c26bd3ffc34dfdbe44e9b6c147bffa"  # API Key（请替换为你自己的）
)

print(">>> 大语言模型已配置")
print(f"    模型名称: qwen-plus")
print(f"    API 地址: https://dashscope.aliyuncs.com/compatible-mode/v1")

# 将所有工具绑定到模型
# bind_tools 方法让模型知道有哪些工具可用，以及如何调用它们
tools = [search_tool, weather_info_tool, hub_stats_tool]
chat_with_tools = llm.bind_tools(tools)  # parallel_tool_calls=False 可以加上该参数对比试试

print("\n>>> 工具已绑定到模型")
print(f"    可用工具: {[tool.name for tool in tools]}")

print("\n" + "=" * 60)
print("步骤 5: 定义 Agent 状态和流程图")
print("=" * 60)


# 定义 Agent 状态类型
# AgentState 是一个 TypedDict，用于存储 Agent 运行时的状态
# messages 字件使用 add_messages reducer，可以自动追加消息而不是替换
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def assistant(state: AgentState):
    """
    Agent 的主要处理函数

    参数:
        state: 当前 Agent 状态，包含消息历史

    返回:
        更新后的状态，包含模型的新回复
    """
    # 调用绑定了工具的模型处理消息
    print(f"==========assistant request:{state}==============")
    res = {
        "messages": [chat_with_tools.invoke(state["messages"])],
    }
    print(f"==========assistant response:{res}==============")
    return res


# 使用 StateGraph 构建 Agent 流程图
# StateGraph 是 LangGraph 的核心类，用于定义有状态的流程图
builder = StateGraph(AgentState)

print(">>> Agent 状态类型已定义")
print(">>> 流程图构建器已创建")

print("\n" + "=" * 60)
print("步骤 6: 构建流程图节点和边")
print("=" * 60)

# 定义节点：节点是流程图中的处理单元
# assistant 节点：调用大语言模型处理消息
# tools 节点：执行工具调用（ToolNode 是预置的工具执行节点）
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

print(">>> 流程图节点已添加")
print("    - assistant: 大语言模型处理节点")
print("    - tools: 工具执行节点")

# 定义边：边决定了控制流如何移动
# START -> assistant: 从开始节点进入 assistant
builder.add_edge(START, "assistant")

# 条件边：根据消息内容决定下一步
# tools_condition 会检查最后一条消息是否包含工具调用请求
# 如果需要调用工具，路由到 tools 节点
# 否则，直接返回结果（结束流程）
builder.add_conditional_edges(
    "assistant",
    tools_condition,  # 预置的条件函数，判断是否需要调用工具
)

# tools -> assistant: 工具执行完毕后返回 assistant 继续处理
builder.add_edge("tools", "assistant")

print("\n>>> 流程图边已定义")
print("    - START -> assistant")
print("    - assistant -> tools（条件边：需要工具时）")
print("    - tools -> assistant（工具执行后返回）")

# 编译流程图，生成可执行的 Agent
alfred = builder.compile()

print("\n>>> Agent 流程图已编译完成")
print("    Agent 名称: Alfred")

print("\n" + "=" * 60)
print("步骤 7: 执行 Agent 查询")
print("=" * 60)

# 创建用户消息
# 询问 Facebook 是谁以及他们最受欢迎的模型
messages = [HumanMessage(content="Who is Facebook and what's their most popular model?")]

print(">>> 用户问题: Who is Facebook and what's their most popular model?")
print("\n>>> Agent 正在处理...")

# 执行 Agent
# Agent 会自动判断是否需要调用工具，并执行相应的工具调用
response = alfred.invoke({"messages": messages})

print("\n>>> Agent 处理完成")

# 输出最终结果
print("\n" + "=" * 60)
print("最终结果")
print("=" * 60)
print("🎩 Alfred's Response:")
print(response['messages'][-1].content)

print("\n" + "=" * 60)
print("脚本执行完毕")
print("=" * 60)
