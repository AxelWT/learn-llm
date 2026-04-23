import asyncio
import json
import time
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END, START
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langgraph.graph.state import CompiledStateGraph


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
    # id标识
    email_id: int


# Initialize our LLM (local ollama)
model = ChatOllama(model="qwen3.5:9b-q4_K_M", temperature=0)


async def read_email(state: EmailState):
    """Alfred reads and logs the incoming email"""
    email = state["email"]
    email_id = state["email_id"]

    # 在这里我们可能会做一些初步的预处理
    print(f"Alfred is processing an email (id:{email_id}) from {email['sender']} with subject: {email['subject']}")

    # 这里不需要更改状态
    return {}


async def classify_email(state: EmailState):
    """
    Alfred uses an LLM to determine if the email is spam or legitimate
    """
    email = state["email"]
    email_id = state["email_id"]

    # 为 LLM 准备提示词（要求 JSON 格式输出）
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
    response = await model.ainvoke(messages)

    # 打印模型输出
    print(f"\n[LLM 分析结果][id:{email_id}]:\n{response.content}\n")

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
    res = {
        "is_spam": is_spam,
        "spam_reason": spam_reason,
        "email_category": email_category,
        "messages": new_messages
    }
    print(f"classify_email execute ========={json.dumps(res, indent=4, ensure_ascii=False)}===============")
    return res


async def handle_spam(state: EmailState):
    """Alfred discards spam email with a note"""
    reason = state.get("spam_reason") or "未提供具体原因"
    print(f"Alfred has marked the email as spam.(id:{state['email_id']}) Reason: {reason}")
    print("The email has been moved to the spam folder.")

    # 我们已处理完这封电子邮件
    return {}


async def draft_response(state: EmailState):
    """Alfred drafts a preliminary response for legitimate emails"""
    email = state["email"]
    email_id = state["email_id"]
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
    response = await model.ainvoke(messages)

    # 打印模型输出
    print(f"\n[LLM 草稿响应][id:{email_id}]:\n{response.content}\n")

    # 更新消息以进行追踪
    new_messages = state.get("messages", []) + [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response.content}
    ]

    # 返回状态更新
    res = {
        "draft_response": response.content,
        "messages": new_messages
    }
    print(f"draft_response execute ========={json.dumps(res, indent=4, ensure_ascii=False)}===============")
    return res


async def notify_mr_hugg(state: EmailState):
    """Alfred notifies Mr. Hugg about the email and presents the draft response"""
    email = state["email"]
    email_id = state["email_id"]

    print("\n" + "=" * 50)
    print(f"email_id: {email_id}")
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


def build_graph():
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
    return compiled_graph


async def process_single_email(p_graph: CompiledStateGraph, email_data: dict, email_id: int):
    """处理单个邮件，返回任务 ID 和结果"""
    start = time.time()
    result = await p_graph.ainvoke({
        "email": email_data,
        "is_spam": None,
        "spam_reason": None,
        "email_category": None,
        "draft_response": None,
        "messages": [],
        "email_id": email_id,
    })
    elapsed = time.time() - start
    print(f"[Task {email_id}] 完成，耗时: {elapsed:.2f}s")
    return email_id, elapsed, result


async def concurrent_process(emails: list[dict]):
    """测试并发能力"""
    print(f"\n{'=' * 60}")
    print(f"并发测试: 同时处理 {len(emails)} 个邮件")
    print(f"{'=' * 60}\n")

    graph = build_graph()

    tasks = []
    for i, email_data in enumerate(emails):
        tasks.append(asyncio.create_task(process_single_email(graph, email_data, i)))

    # 并发执行
    overall_start = time.time()
    results = await asyncio.gather(*tasks)
    overall_elapsed = time.time() - overall_start

    # 统计结果
    print(f"\n{'=' * 60}")
    print(f"测试完成!")
    print(f"总耗时: {overall_elapsed:.2f}s")
    print(f"平均单任务耗时: {sum(r[1] for r in results) / len(results):.2f}s")
    print(f"并发效率: {sum(r[1] for r in results) / overall_elapsed:.2f}x")
    print(f"{'=' * 60}\n")


# 测试邮件数据
test_emails = [
    {"sender": "john.smith@example.com", "subject": "Question about services",
     "body": "I'm interested in your consulting services."},
    {"sender": "winner@lottery.com", "subject": "YOU WON $5M!!!", "body": "Send $100 to claim your prize."},
    {"sender": "support@company.com", "subject": "Thank you", "body": "Thanks for your help yesterday."},
    {"sender": "client@business.com", "subject": "Meeting request", "body": "Can we meet next Tuesday?"},
    {"sender": "spam@fake.com", "subject": "Free money!", "body": "Click here for free money now."},
]

if __name__ == "__main__":
    asyncio.run(concurrent_process(test_emails))
