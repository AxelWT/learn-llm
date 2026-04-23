"""
RAG (Retrieval Augmented Generation) 示例脚本

本脚本演示如何使用 LangGraph 构建一个带检索功能的 AI Agent，
该 Agent 可以查询嘉宾信息数据库并根据查询结果回答用户问题。

主要组件：
1. 数据加载：从 HuggingFace 数据集加载嘉宾信息
2. 检索器：使用 BM25 算法进行文档检索
3. Agent：使用 LangGraph 构建带工具调用的对话流程
"""

import datasets
import json
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langgraph.graph.state import CompiledStateGraph

# ==================== 数据加载 ====================

# 从 HuggingFace Hub 加载嘉宾数据集
# 该数据集包含晚宴嘉宾的姓名、关系、描述和邮箱信息
guest_dataset = datasets.load_dataset("agents-course/unit3-invitees", split="train")

# 将数据集转换为 LangChain Document 对象
# Document 是 LangChain 中用于表示文本片段的标准格式
# page_content: 文档的主要内容
# metadata: 元数据，可用于后续检索和过滤
docs = [
    Document(
        page_content="\n".join([
            f"Name: {guest['name']}",
            f"Relation: {guest['relation']}",
            f"Description: {guest['description']}",
            f"Email: {guest['email']}"
        ]),
        metadata={"name": guest["name"]}
    )
    for guest in guest_dataset
]

# 美化输出文档列表
docs_dict = [doc.model_dump() for doc in docs]
print(json.dumps(docs_dict, indent=2, ensure_ascii=False))

# ==================== 检索器设置 ====================

from langchain_community.retrievers import BM25Retriever
from langchain_core.tools import Tool

# BM25 是一种基于词频和文档频率的检索算法
# 相比简单的关键词匹配，BM25 能更好地处理文档长度和词频权重
# from_documents 方法会自动根据文档内容构建索引
bm25_retriever = BM25Retriever.from_documents(docs)


def extract_text(query: str) -> str:
    """
    检索工具函数：根据查询词检索嘉宾信息

    Args:
        query: 查询字符串，可以是嘉宾姓名或关系

    Returns:
        返回最多3个匹配嘉宾的详细信息，若无匹配则返回提示信息
    """
    print(f"===========tool extract_text request:{query}============")
    results = bm25_retriever.invoke(query)
    print(f"===========tool extract_text res:{results}============")
    if results:
        # 返回前3个结果的 page_content，用换行分隔
        return "\n\n".join([doc.page_content for doc in results[:3]])
    else:
        return "No matching guest information found."


# 将检索函数封装为 LangChain Tool
# name: 工具名称，模型调用时会使用此名称
# func: 工具的实际执行函数
# description: 工具描述，模型根据此描述判断是否需要调用该工具
guest_info_tool = Tool(
    name="guest_info_retriever",
    func=extract_text,
    description="Retrieves detailed information about gala guests based on their name or relation."
)

# ==================== Agent 配置 ====================

from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition

# 配置大语言模型（使用阿里云千问）
llm = ChatOpenAI(
    model="qwen-plus",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key="sk-..."  # 替换为你的 DashScope API Key
)

# 将工具绑定到模型
# bind_tools 让模型知道有哪些工具可用，以及如何调用它们
# parallel_tool_calls=False: 不允许并行调用多个工具，确保工具按顺序执行
tools = [guest_info_tool]
chat_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)


# ==================== StateGraph 定义 ====================

# AgentState 定义 Agent 的状态结构
# messages: 存储对话历史，使用 add_messages reducer 自动追加新消息
# TypedDict 提供类型提示，Annotated 用于添加 reducer 函数
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


def assistant(state: AgentState):
    """
    Agent 的主节点函数：调用模型处理当前对话

    Args:
        state: 当前 Agent 状态，包含历史消息

    Returns:
        返回模型响应，会自动追加到 messages 中
    """
    print(f"==========assistant request:{state}==============")
    res = {
        "messages": [chat_with_tools.invoke(state["messages"])],
    }
    print(f"==========assistant response:{res}==============")
    return res


def build_graph() -> CompiledStateGraph:
    """
    构建 LangGraph 流程图

    流程：
    START -> assistant -> (判断是否需要工具)
                        -> 需要工具: tools -> assistant (循环)
                        -> 不需要工具: 直接结束

    Returns:
        CompiledStateGraph: 编译后的可执行图
    """
    # 创建状态图构建器，指定状态类型
    builder = StateGraph(AgentState)

    # 定义节点：图中的处理单元
    # assistant: 主对话节点，调用模型
    # tools: 工具执行节点，执行检索等工具
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))

    # 定义边：节点之间的连接关系
    # START -> assistant: 入口边，从起点到主节点
    builder.add_edge(START, "assistant")

    # 条件边：根据模型输出决定下一步
    # tools_condition 是 LangGraph 提供的预设条件函数
    # 如果模型输出包含 tool_calls，路由到 tools 节点
    # 否则结束流程
    builder.add_conditional_edges(
        "assistant",
        tools_condition,
    )

    # tools -> assistant: 工具执行完成后返回主节点继续处理
    builder.add_edge("tools", "assistant")

    # compile(): 编译图，使其可执行
    return builder.compile()


# ==================== 运行 Agent ====================

# 构建并编译 Agent
alfred = build_graph()

# 创建用户消息，查询特定嘉宾信息
messages = [HumanMessage(content="Tell me about our guest named 'Lady Ada Lovelace'.")]

# invoke(): 执行 Agent，传入初始状态
# 返回包含完整对话历史的字典
response = alfred.invoke({"messages": messages})

# 输出最终回复（取最后一条消息的内容）
print("=" * 60)
print("🎩 Alfred's Response:")
print(response['messages'][-1].content)


"""
该脚本没有系统提示词，模型怎么知道要调用注册的工具呢？

⏺ 好问题。模型是通过bind_tools方法得知工具信息的：

chat_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)

bind_tools的工作原理：

调用invoke时，工具信息会以API的tools参数发送给模型：

{
    "model": "qwen-plus",
    "messages": [...],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "guest_info_retriever",
                "description": "Retrieves detailed information about gala guests based on their name or relation.",
                "parameters": {}
            }
        }
    ]
}

模型根据name和description判断是否需要调用工具。

验证方式：可以查看实际发送的请求参数：

# 查看 bind_tools 后的模型配置                                                                                                                                                     
print(chat_with_tools.kwargs)

或者在assistant函数中打印完整调用信息：

response = chat_with_tools.invoke(state["messages"])
print(
    response.tool_calls)  # 如果模型决定调用工具，这里会有内容                                                                                                                   

所以系统提示词不是必须的，bind_tools已经把工具信息注入到API请求中，模型自动知道有哪些工具可用。但添加系统提示词可以更好地引导模型何时、如何使用工具。
"""