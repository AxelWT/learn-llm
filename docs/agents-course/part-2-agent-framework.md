# 智能体框架

已经可以确定的一些功能的需求：

    一个驱动系统的大语言模型引擎。
    智能体可以访问的工具列表。
    用于从大语言模型输出中提取工具调用的解析器。
    与解析器同步的系统提示。
    一个记忆系统。
    错误日志和重试机制以控制大语言模型的错误。

我们将探讨这些主题在各种框架中如何解决，包括 smolagents、LlamaIndex 和 LangGraph。

---

## smolAgents框架

- to-do

---

## llamaIndex框架

- to-do

---

## langGraph框架

- **LangGraph 如何工作？**

其核心在于，LangGraph 使用有向图结构来定义应用程序的流程：

    节点 表示独立的处理步骤（如调用 LLM、使用工具或做出决策）
    边 定义步骤之间可能的转换
    状态 由用户定义和维护，并在执行期间在节点间传递。当决定下一个目标节点时，我们查看的就是当前状态

注意：langChain 使用有向无环图（DAG）方式编排工作流，langGraph支持有环图来持续驱动 AI 进行改进

- **核心要素**
  **状态**、**节点**、**边**、**状态图**

- **实现**

参考 src/agents-course/s02_lang_graph_email_process.py 实现了辨别垃圾邮件的一个决策分支

参考 src/agents-course/s02_lang_graph_text_analysis.py 实现了 agent-tool 循环 ReAct

测试输出参考如下（测试中的输入图片可以让 AI 生成一张，目的是为了测试图片 OCR 模型的运行效果）：

```bash 
Graph saved to graph.png
===========assistant response.content 【I'll help you extract the dinner menu items from Mr. Wayne's note. Let me analyze the image first.

】============
========tool extract_text input:img.png output:本周推荐食谱：烤西冷牛排配配菜

✓ 草饲本地西冷牛排
- 250g，纹理丰富

✓ 有机菠菜
- 一扎，新鲜洗净

✓ 皮克略辣椒
- 一罐，烤制甜味

✓ 土豆（用于烤制金黄香草土豆）
- 500g，切块，配迷迭香

✓ 鱼油（2 克）- 每日补充=========
===========assistant response.content 【Based on Mr. Wayne's note in the image, here's the list of items you should buy for the dinner menu:

- Grass-fed local sirloin steak (250g)
- Organic spinach (one bunch, fresh and washed)
- Piquillo peppers (one jar, for roasting)
- Potatoes (500g, to be cut and roasted with rosemary)
- Fish oil (2g daily supplement)

Note: The fish oil appears to be a daily supplement rather than a cooking ingredient, but it's included in the note as part of the weekly recommendation.

Would you like me to help with anything else regarding this dinner menu, Master Wayne?】============
[
  {
    "content": "According to the note provided by Mr. Wayne in the provided images. What's the list of items I should buy for the dinner menu?",
    "additional_kwargs": {},
    "response_metadata": {},
    "type": "human",
    "name": null,
    "id": "7c95ee83-6b0d-4f18-9af3-8ef367758ac0"
  },
  {
    "content": "I'll help you extract the dinner menu items from Mr. Wayne's note. Let me analyze the image first.\n\n",
    "additional_kwargs": {
      "refusal": null
    },
    "response_metadata": {
      "token_usage": {
        "completion_tokens": 44,
        "prompt_tokens": 393,
        "total_tokens": 437,
        "completion_tokens_details": null,
        "prompt_tokens_details": {
          "audio_tokens": null,
          "cached_tokens": 0
        }
      },
      "model_provider": "openai",
      "model_name": "qwen-plus",
      "system_fingerprint": null,
      "id": "chatcmpl-c29da592-a289-917d-b378-4e406319c4d4",
      "finish_reason": "tool_calls",
      "logprobs": null
    },
    "type": "ai",
    "name": null,
    "id": "lc_run--019db96f-0449-7a33-ac5b-387085ec9f61-0",
    "tool_calls": [
      {
        "name": "extract_text",
        "args": {
          "img_path": "img.png"
        },
        "id": "call_ddf30b4a7cef4f2ca43d61",
        "type": "tool_call"
      }
    ],
    "invalid_tool_calls": [],
    "usage_metadata": {
      "input_tokens": 393,
      "output_tokens": 44,
      "total_tokens": 437,
      "input_token_details": {
        "cache_read": 0
      },
      "output_token_details": {}
    }
  },
  {
    "content": "本周推荐食谱：烤西冷牛排配配菜\n\n✓ 草饲本地西冷牛排\n- 250g，纹理丰富\n\n✓ 有机菠菜\n- 一扎，新鲜洗净\n\n✓ 皮克略辣椒\n- 一罐，烤制甜味\n\n✓ 土豆（用于烤制金黄香草土豆）\n- 500g，切块，配迷迭香\n\n✓ 鱼油（2 克）- 每日补充",
    "additional_kwargs": {},
    "response_metadata": {},
    "type": "tool",
    "name": "extract_text",
    "id": "d06ca068-90b5-4ae4-bd23-cc9472553c89",
    "tool_call_id": "call_ddf30b4a7cef4f2ca43d61",
    "artifact": null,
    "status": "success"
  },
  {
    "content": "Based on Mr. Wayne's note in the image, here's the list of items you should buy for the dinner menu:\n\n- Grass-fed local sirloin steak (250g)\n- Organic spinach (one bunch, fresh and washed)\n- Piquillo peppers (one jar, for roasting)\n- Potatoes (500g, to be cut and roasted with rosemary)\n- Fish oil (2g daily supplement)\n\nNote: The fish oil appears to be a daily supplement rather than a cooking ingredient, but it's included in the note as part of the weekly recommendation.\n\nWould you like me to help with anything else regarding this dinner menu, Master Wayne?",
    "additional_kwargs": {
      "refusal": null
    },
    "response_metadata": {
      "token_usage": {
        "completion_tokens": 138,
        "prompt_tokens": 562,
        "total_tokens": 700,
        "completion_tokens_details": null,
        "prompt_tokens_details": {
          "audio_tokens": null,
          "cached_tokens": 0
        }
      },
      "model_provider": "openai",
      "model_name": "qwen-plus",
      "system_fingerprint": null,
      "id": "chatcmpl-a737d7c0-1c76-9d6c-8359-2a8ce68484fa",
      "finish_reason": "stop",
      "logprobs": null
    },
    "type": "ai",
    "name": null,
    "id": "lc_run--019db96f-298b-7411-8642-6ca6de610562-0",
    "tool_calls": [],
    "invalid_tool_calls": [],
    "usage_metadata": {
      "input_tokens": 562,
      "output_tokens": 138,
      "total_tokens": 700,
      "input_token_details": {
        "cache_read": 0
      },
      "output_token_details": {}
    }
  }
]
============================================================
Based on Mr. Wayne's note in the image, here's the list of items you should buy for the dinner menu:

- Grass-fed local sirloin steak (250g)
- Organic spinach (one bunch, fresh and washed)
- Piquillo peppers (one jar, for roasting)
- Potatoes (500g, to be cut and roasted with rosemary)
- Fish oil (2g daily supplement)

Note: The fish oil appears to be a daily supplement rather than a cooking ingredient, but it's included in the note as part of the weekly recommendation.

Would you like me to help with anything else regarding this dinner menu, Master Wayne?

Process finished with exit code 0
```

---