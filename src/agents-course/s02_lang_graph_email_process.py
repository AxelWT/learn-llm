import os
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END, START
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage

# ============trace============
# 增加 trace 监控，在 langfuse 官网页面上查看，可以注释掉不用它
import os

# Get keys for your project from the project settings page: https://cloud.langfuse.com
os.environ["LANGFUSE_PUBLIC_KEY"] = "pk-lf-..."
os.environ["LANGFUSE_SECRET_KEY"] = "sk-lf-..."
os.environ["LANGFUSE_HOST"] = "https://us.cloud.langfuse.com"  # 🇺🇸 US region

from langfuse.langchain import CallbackHandler

# 为 LangGraph/Langchain 初始化 Langfuse CallbackHandler（跟踪）
langfuse_handler = CallbackHandler()
# ============trace============

class EmailState(TypedDict):
    # 正在处理的电子邮件
    email: Dict[str, Any]  # 包含主题、发件人、正文等。

    # 分析与决策
    is_spam: Optional[bool]
    spam_reason: Optional[str]
    email_category: Optional[str]

    # 响应生成
    draft_response: Optional[str]

    # 处理元数据
    messages: List[Dict[str, Any]]  # 跟踪与 LLM 的对话以进行分析


# Initialize our LLM (local ollama)
model = ChatOllama(model="qwen3.5:9b-q4_K_M", temperature=0)


def read_email(state: EmailState):
    """Alfred reads and logs the incoming email"""
    email = state["email"]

    # 在这里我们可能会做一些初步的预处理
    print(f"Alfred is processing an email from {email['sender']} with subject: {email['subject']}")

    # 这里不需要更改状态
    return {}


def classify_email(state: EmailState):
    """Alfred uses an LLM to determine if the email is spam or legitimate"""
    email = state["email"]

    # Prepare prompt for LLM (require JSON format output)
    prompt = f"""
            Analyze the following email and determine if it is spam.

            Email Information:
            Sender: {email['sender']}
            Subject: {email['subject']}
            Body: {email['body']}

            Please output the result strictly in the following JSON format (do not output anything else):
{{"is_spam": true/false, "spam_reason": "reason if spam, otherwise null", "email_category": "inquiry/complaint/thank_you/request/information/general"}}
"""

    # Call the LLM
    messages = [HumanMessage(content=prompt)]
    response = model.invoke(messages)

    # 打印模型输出
    print(f"\n[LLM 分析结果]:\n{response.content}\n")

    # 解析 JSON 响应
    import json
    import re

    response_text = response.content.strip()

    # 尝试从响应中提取 JSON
    try:
        # 尝试直接解析
        result = json.loads(response_text)
    except json.JSONDecodeError:
        # 尝试从文本中提取 JSON 部分
        json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
        else:
            # 默认值
            result = {"is_spam": False, "spam_reason": None, "email_category": "general"}

    is_spam = result.get("is_spam", False)
    spam_reason = result.get("spam_reason")
    email_category = result.get("email_category", "general")

    # 更新消息以进行追踪
    new_messages = state.get("messages", []) + [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response.content}
    ]

    # 返回状态更新
    return {
        "is_spam": is_spam,
        "spam_reason": spam_reason,
        "email_category": email_category,
        "messages": new_messages
    }


def handle_spam(state: EmailState):
    """Alfred discards spam email with a note"""
    reason = state.get("spam_reason") or "No specific reason provided"
    print(f"Alfred has marked the email as spam. Reason: {reason}")
    print("The email has been moved to the spam folder.")

    # 我们已处理完这封电子邮件
    return {}


def draft_response(state: EmailState):
    """Alfred drafts a preliminary response for legitimate emails"""
    email = state["email"]
    category = state["email_category"] or "general"

    # 为 LLM 准备提示词
    prompt = f"""
    As Alfred the butler, draft a polite preliminary response to this email.

    Email:
    From: {email['sender']}
    Subject: {email['subject']}
    Body: {email['body']}

    This email has been categorized as: {category}

    Draft a brief, professional response that Mr. Hugg can review and personalize before sending.
    """

    # Call the LLM
    messages = [HumanMessage(content=prompt)]
    response = model.invoke(messages)

    # 打印模型输出
    print(f"\n[LLM 草稿响应]:\n{response.content}\n")

    # 更新消息以进行追踪
    new_messages = state.get("messages", []) + [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response.content}
    ]

    # 返回状态更新
    return {
        "draft_response": response.content,
        "messages": new_messages
    }


def notify_mr_hugg(state: EmailState):
    """Alfred notifies Mr. Hugg about the email and presents the draft response"""
    email = state["email"]

    print("\n" + "=" * 50)
    print(f"Sir, you've received an email from {email['sender']}.")
    print(f"Subject: {email['subject']}")
    print(f"Category: {state['email_category']}")
    print("\nI've prepared a draft response for your review:")
    print("-" * 50)
    print(state["draft_response"])
    print("=" * 50 + "\n")

    # 我们已处理完这封电子邮件
    return {}


def route_email(state: EmailState) -> str:
    """Determine the next step based on spam classification"""
    if state["is_spam"]:
        return "spam"
    else:
        return "legitimate"


# 创建 graph
email_graph = StateGraph(EmailState)

# 添加 nodes
email_graph.add_node("read_email", read_email)
email_graph.add_node("classify_email", classify_email)
email_graph.add_node("handle_spam", handle_spam)
email_graph.add_node("draft_response", draft_response)
email_graph.add_node("notify_mr_hugg", notify_mr_hugg)

# 添加 edges - 定义流程
email_graph.add_edge(START, "read_email")
email_graph.add_edge("read_email", "classify_email")

# 从 classify_email 添加条件分支
email_graph.add_conditional_edges(
    "classify_email",
    route_email,
    {
        "spam": "handle_spam",
        "legitimate": "draft_response"
    }
)

# 添加最后的 edges
email_graph.add_edge("handle_spam", END)
email_graph.add_edge("draft_response", "notify_mr_hugg")
email_graph.add_edge("notify_mr_hugg", END)

# 编译 graph
compiled_graph = email_graph.compile()

# 保存 graph 图片
png_data = compiled_graph.get_graph().draw_mermaid_png()
with open("graph.png", "wb") as f:
    f.write(png_data)
print("Graph saved to graph.png")

# 合法电子邮件示例
legitimate_email = {
    "sender": "john.smith@example.com",
    "subject": "Question about your services",
    "body": "Dear Mr. Hugg, I was referred to you by a colleague and I'm interested in learning more about your consulting services. Could we schedule a call next week? Best regards, John Smith"
}

# 垃圾邮件示例
spam_email = {
    "sender": "winner@lottery-intl.com",
    "subject": "YOU HAVE WON $5,000,000!!!",
    "body": "CONGRATULATIONS! You have been selected as the winner of our international lottery! To claim your $5,000,000 prize, please send us your bank details and a processing fee of $100."
}

# 处理合法电子邮件
print("\nProcessing legitimate email...")
# legitimate_result = compiled_graph.invoke({
#     "email": legitimate_email,
#     "is_spam": None,
#     "spam_reason": None,
#     "email_category": None,
#     "draft_response": None,
#     "messages": []
# })

# 处理合法电子邮件 langfuse 进行日志链路 trace
legitimate_result = compiled_graph.invoke(
    input={"email": legitimate_email,
           "is_spam": None,
           "spam_reason": None,
           "email_category": None,
           "draft_response": None,
           "messages": []
           },
    config={"callbacks": [langfuse_handler]}
)

# 处理垃圾邮件
print("\nProcessing spam email...")
spam_result = compiled_graph.invoke({
    "email": spam_email,
    "is_spam": None,
    "spam_reason": None,
    "email_category": None,
    "draft_response": None,
    "messages": []
})
