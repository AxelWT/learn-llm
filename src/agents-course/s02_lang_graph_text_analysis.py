import base64
import json
from typing import TypedDict, Optional, Annotated

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.constants import START
from langgraph.graph import add_messages, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition


class AgentState(TypedDict):
    # 提供的文件
    input_file: Optional[str]  # Contains file path (PDF/PNG)
    messages: Annotated[list[AnyMessage], add_messages]


vision_llm = ChatOpenAI(
    model="qwen-vl-max",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key="sk-..."  # 替换为你的 DashScope API Key
)


def extract_text(img_path: str) -> str:
    """
    Extract text from an image file using a multimodal model.

    Master Wayne often leaves notes with his training regimen or meal plans.
    This allows me to properly analyze the contents.
    """
    all_text = ""
    try:
        # 读取图像并编码为 base64
        with open(img_path, "rb") as image_file:
            image_bytes = image_file.read()

        image_base64 = base64.b64encode(image_bytes).decode("utf-8")

        # 准备包含 base64 图像数据的提示
        message = [
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": (
                            "Extract all the text from this image. "
                            "Return only the extracted text, no explanations."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"
                        },
                    },
                ]
            )
        ]

        # 调用具有视觉功能的模型
        response = vision_llm.invoke(message)

        # 附加提取的文本
        all_text += response.content + "\n\n"
        res = all_text.strip()
        print(f"========tool extract_text input:{img_path} output:{res}=========")
        return res
    except Exception as e:
        # 管家应优雅地处理错误
        error_msg = f"Error extracting text: {str(e)}"
        print(error_msg)
        print(f"========tool extract_text input:{img_path} error:{error_msg}=========")
        return ""


def divide(a: int, b: int) -> float:
    """Divide a and b - for Master Wayne's occasional calculations."""
    res = a / b
    print(f"========tool divide input:{a, b} output:{res}=========")
    return res


# 为管家配备工具
tools = [
    divide,
    extract_text
]

llm = ChatOpenAI(
    model="qwen-plus",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    api_key="sk-..."  # 替换为你的 DashScope API Key
)
llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)


def assistant(state: AgentState):
    # 系统消息
    textual_description_of_tool = """
extract_text(img_path: str) -> str:
    Extract text from an image file using a multimodal model.

    Args:
        img_path: A local image file path (strings).

    Returns:
        A single string containing the concatenated text extracted from each image.
divide(a: int, b: int) -> float:
    Divide a and b
"""
    image = state["input_file"]
    sys_msg = SystemMessage(
        content=f"You are an helpful butler named Alfred that serves Mr. Wayne and Batman. You can analyse documents and run computations with provided tools:\n{textual_description_of_tool} \n You have access to some optional images. Currently the loaded image is: {image}")

    response = llm_with_tools.invoke([sys_msg] + state["messages"])
    if response.content:
        print(f"===========assistant response.content 【{response.content}】============")
    elif response.tool_calls:
        print(f"===========assistant response.tool_calls 【{response.tool_calls}】============")
    else:
        print(f"===========assistant response 【{response}】============")

    return {
        "messages": [response],
        "input_file": state["input_file"]
    }


def build_graph() -> CompiledStateGraph:
    ## The graph
    builder = StateGraph(AgentState)

    # 定义节点：这些节点完成工作
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))

    # 定义边：这些决定了控制流如何移动
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges(
        "assistant",
        # 如果最新消息需要工具，则路由至工具
        # 否则，请直接回复
        tools_condition,
    )
    builder.add_edge("tools", "assistant")
    act_graph = builder.compile()

    # 将图保存到文件
    with open("graph.png", "wb") as f:
        f.write(act_graph.get_graph(xray=True).draw_mermaid_png())
    print("Graph saved to graph.png")
    return act_graph


react_graph = build_graph()

# messages = [HumanMessage(content="Divide 6790 by 5")]
# result = react_graph.invoke({"messages": messages, "input_file": None})

messages = [HumanMessage(content="According to the note provided by Mr. Wayne in the provided images. What's the list of items I should buy for the dinner menu?")]
result = react_graph.invoke({"messages": messages, "input_file": "./img.png"})

# 美化输出 messages
messages_json = [msg.model_dump() for msg in result["messages"]]
print(json.dumps(messages_json, indent=2, ensure_ascii=False))
print("=" * 60)
print(result["messages"][-1].content)
