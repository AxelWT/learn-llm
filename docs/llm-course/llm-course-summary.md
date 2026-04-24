# HuggingFace LLM 课程详细总结

## 课程概述

本课程是 HuggingFace 官方 LLM 学习课程的中文学习笔记和实践代码，涵盖了从 Transformer 基础到大语言模型训练的完整知识体系。

---

## 一、课程结构总览

### 文档部分 (docs/llm-course/)
- **llm-course-part-1.md**: 课程第一部分核心笔记
- **llm-course-part-2.md**: 课程第二部分核心笔记
- **llm-course-part-3.md**: Gradio 应用构建笔记
- **llm-course-part-2-*.md**: 各类 NLP 任务详解（NER、翻译、MLM、CLM、摘要、问答、分词器）
- **reference/*.md**: 深入原理参考文档

### 代码部分 (src/llm-course/)
- **s02_*.py**: Transformers 基础使用
- **s03_*.py**: 数据集加载与微调
- **s04_*.py**: 预训练模型快速使用
- **s05_*.py**: 数据集处理与语义搜索
- **s06_*.py**: 分词器训练
- **s07_*.py**: 各类任务微调实践

---

## 二、核心知识点详解

### 1. Transformer 模型基础

#### 什么是 NLP？
NLP（自然语言处理）是语言学和机器学习的交叉领域，专注于理解与人类语言相关的一切。NLP 任务的目标不仅是理解单个单词，更要理解这些单词的上下文。

#### 常见 NLP 任务类型
| 任务类型 | 描述 | 示例 |
|---------|------|------|
| 文本分类 | 对整个句子进行分类 | 情感分析、垃圾邮件检测 |
| 词级分类 | 对句子中每个词进行分类 | 命名实体识别(NER)、词性标注 |
| 文本生成 | 生成文本内容 | GPT 系列模型 |
| 问答抽取 | 从文本中提取答案 | SQuAD 任务 |
| 序列转换 | 从输入文本生成新句子 | 翻译、摘要 |

#### 模型架构分类
```
仅编码器（BERT）:
输入 → [编码器] → 理解表示
       双向↑
       输出是向量表示，不是文本

仅解码器（GPT）:
输入 → [解码器] → 生成文本
       单向↑
       每次只能看前面已生成的内容

编码器-解码器（T5）:
输入 → [编码器] → [解码器] → 输出文本
       双向↑        单向↑
       先理解输入    再生成输出
```

---

### 2. Transformers 库使用

#### Pipeline 快速使用
Pipeline 是 HuggingFace 提供的高层 API，用于快速加载和使用预训练模型：

```python
from transformers import pipeline

# 文本生成
generator = pipeline("text-generation")
result = generator("My favorite programming language is")

# 文本填空
fill_mask = pipeline("fill-mask", model="camembert-base")
result = fill_mask("Le camembert est <mask> :)")
```

#### 分词器（Tokenizer）核心功能
分词器是 Pipeline 的核心组件，负责：
1. **预处理**：将文本转换为模型能理解的数字（Tensor）
2. **后处理**：将数字转换回文本

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
sequence = "Using a Transformer network is simple"

# 分词 → 转ID → 解码
tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)
decoded = tokenizer.decode(ids)
```

#### 分词算法对比
| 算法 | 代表模型 | 特点 |
|------|---------|------|
| **BPE** | GPT-2, RoBERTa | 字节级，频率最高的相邻Token合并 |
| **WordPiece** | BERT | 使用 `##` 前缀，最大化语言模型似然度 |
| **Unigram** | T5, mBART | 从大词表逐步剔除，概率模型 |

---

### 3. 微调预训练模型

#### 微调流程概述
```
原始数据 → 分词处理 → 数据集预处理 → 动态填充 → 模型训练 → 评估 → 保存
```

#### Trainer API vs 手动训练循环

**使用 Trainer API（推荐新手）**:
```python
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments("test-trainer", eval_strategy="epoch")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
trainer.train()
```

**手动训练循环（推荐进阶）**:
```python
from torch.optim import AdamW
from transformers import get_scheduler

optimizer = AdamW(model.parameters(), lr=5e-5)
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_training_steps=num_training_steps)

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
```

#### 关键组件详解
```
┌────────────────────┬──────────────────────────────────────┐
│        组件         │                 作用                  │
├────────────────────┼──────────────────────────────────────┤
│ AdamW              │ 优化器，带权重衰减的Adam               │
├────────────────────┼──────────────────────────────────────┤
│ lr_scheduler       │ 学习率调度器，让学习率随训练逐渐降低    │
├────────────────────┼──────────────────────────────────────┤
│ DataCollator       │ 数据整理器，负责填充和批处理           │
└────────────────────┴──────────────────────────────────────┘
```

---

### 4. Datasets 库核心原理

#### Apache Arrow 架构
Datasets 库使用 Apache Arrow 列式存储格式，实现高效的大数据处理：

```
┌─────────────────────────────────────────────────────────────┐
│                    Hugging Face Datasets                      │
├─────────────────────────────────────────────────────────────┤
│                      Apache Arrow                             │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│   │  Column 1   │  │  Column 2   │  │  Column 3   │  ...     │
│   └─────────────┘  └─────────────┘  └─────────────┘          │
├─────────────────────────────────────────────────────────────┤
│                 Memory Mapping (内存映射)                      │
│                      ↓                                        │
│                 Disk Storage (磁盘存储)                        │
└─────────────────────────────────────────────────────────────┘
```

#### 关键技术原理

| 技术 | 作用 | 优势 |
|------|------|------|
| **Memory Mapping** | 磁盘数据映射到内存 | GB级数据集仅用MB级RAM |
| **Zero-Copy** | 无拷贝读取 | 高频数据转换效率高 |
| **Columnar** | 列式存储 | 只访问需要的列 |
| **Streaming** | 流式加载 | 超大数据集处理 |
| **Lazy Processing** | 惰性执行 | 按需执行处理管道 |

#### 数据集操作示例
```python
from datasets import load_dataset

# 加载本地数据集
dataset = load_dataset("csv", data_files="my_file.csv")

# 数据处理
dataset = dataset.map(process_function, batched=True)
dataset = dataset.filter(filter_function)
dataset = dataset.rename_column("old_name", "new_name")

# 保存数据集
dataset.save_to_disk("my_dataset")
```

---

### 5. 主要 NLP 任务详解

#### 任务类型对比
| 任务类型 | 架构 | Labels来源 | 参考答案本质 |
|---------|------|-----------|-------------|
| **标记分类(NER)** | Encoder | 外部标注 | 类别索引 |
| **掩码建模(MLM)** | Encoder | 原始输入 | 被遮盖位置的原词 |
| **翻译/摘要** | Encoder-Decoder | 目标文本 | 目标语言Token序列 |
| **因果语言模型(CLM)** | Decoder-only | 原始输入 | 序列中的下一个词 |
| **问答系统** | Encoder | 字符位置 | Token位置索引 |

#### 命名实体识别(NER)

核心挑战：将单词级标签对齐到Token级

```python
def align_labels_with_tokens(labels, word_ids):
    """
    将单词级别的标签对齐到 token 级别
    - 特殊 token 标为 -100（不参与损失计算）
    - 每个单词的第一个 token 使用原标签
    - 同一单词后续 token：B-XXX 改为 I-XXX
    """
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            new_labels.append(-100)
        else:
            label = labels[word_id]
            if label % 2 == 1:  # B-XXX 改为 I-XXX
                label += 1
            new_labels.append(label)
    return new_labels
```

#### 掩码语言模型(MLM)

自监督学习，用于领域适配：

```
数据处理流程:
1. 分词 → 2. 分块 → 3. 动态遮盖(15%)

遮盖策略(80-10-10规则):
- 80% 替换为 [MASK]
- 10% 替换为随机词
- 10% 保持原样

参考答案：被遮盖前的原始Token ID
```

#### 翻译任务（Seq2Seq）

核心：处理双语分词和序列生成

```python
# 双分词器处理
tokenizer(example["source"], example["target"])

# DataCollatorForSeq2Seq 自动创建 decoder_input_ids
# 将 labels 向右移动一位，添加起始符
```

#### 因果语言模型(CLM)

从零训练GPT类模型：

```
数据处理流程:
1. 流式加载 → 2. 分词 → 3. 串联 → 4. 分块 → 5. labels = input_ids.copy()

训练逻辑:
用当前词预测下一个词
输入: [A, B, C] → 预测: [B, C, D]
```

#### 问答系统（Extractive QA）

核心：滑动窗口处理长文本

```python
# 滑动窗口切分长文本
tokenizer(
    question, context,
    truncation="only_second",
    return_overflowing_tokens=True,
    stride=128
)

# 输出：start_logits, end_logits
# 参考答案：答案的起始和结束Token索引
```

---

### 6. 分词器训练

#### 为什么要训练新分词器？
当预训练模型分词器在处理特定领域文本（如中文、法律文档、Python代码）时效率低，需要重新训练。

#### 训练流程
```python
from transformers import AutoTokenizer

# 加载旧分词器作为模板
old_tokenizer = AutoTokenizer.from_pretrained("gpt2")

# 训练新分词器
new_tokenizer = old_tokenizer.train_new_from_iterator(
    training_corpus,  # 语料库迭代器
    vocab_size=52000  # 新词表大小
)

# 保存
new_tokenizer.save_pretrained("my-tokenizer")
new_tokenizer.push_to_hub("my-tokenizer")
```

#### 快速分词器的特殊能力
- **offset_mapping**: 记录每个Token在原文本中的字符位置
- **word_ids()**: 获取Token对应的单词索引
- **滑动窗口**: 处理超长文本时切分重叠块

---

### 7. 自回归模型详解

#### 核心定义
**自回归(Autoregressive)** = 用前面的词预测后面的词，一个接一个生成。

#### 数学表达
$$P(w_1, w_2, ..., w_n) = P(w_1) \times P(w_2|w_1) \times P(w_3|w_1,w_2) \times ...$$

#### 与掩码模型对比
| 类型 | 代表模型 | 预测方式 | 适用场景 |
|------|----------|----------|----------|
| **自回归** | GPT 系列 | 从左到右，看前面预测后面 | 文本生成、对话 |
| **掩码模型** | BERT 系列 | 遮住中间词，看前后预测中间 | 文本理解、分类 |

---

### 8. 微调原理详解

#### 两阶段微调流程
```
第一步：领域微调（学会"行话"）
- 用无标签专业文本训练
- 得到"法律版BERT"或"医学版BERT"

第二步：任务微调（学会"干活"）
- 加上任务头(Task Head)
- 用有标签数据训练
- 得到最终应用模型
```

#### 任务头(Task Head)
任务头是模型末端的功能层：

| 任务头类型 | 输出 | 用途 |
|-----------|------|------|
| Classification Head | 类别概率 | 情感分析 |
| NER Head | 每个Token的标签 | 实体识别 |
| QA Head | 起始和结束位置 | 问答抽取 |
| LM Head | 词表概率分布 | 文本生成 |

---

### 9. 词表库详解

#### 什么是词表库？
词表库是模型能理解的最小语义单元集合，建立人类语言与计算机数字之间的翻译桥梁。

#### 词表内容
- **基础字符**: 字母、数字、标点
- **子词(Subwords)**: 如 `learn` + `##ing`
- **特殊标记**: `[CLS]`, `[SEP]`, `[PAD]`, `[UNK]`

#### 主流词表对比
| 词表名称 | 代表模型 | 词表大小 | 算法 | 对中文友好度 |
|---------|---------|---------|------|-------------|
| Llama 3 | Llama系列 | ~128k | BPE | 极高 |
| o200k_base | GPT-4o | ~200k | BPE | 极高 |
| Qwen2.5 | Qwen系列 | ~151k | BPE | 顶级 |
| Bert-Chinese | BERT | ~21k | WordPiece | 中等 |

---

### 10. Data Collator 详解

#### 什么是数据整理器？
数据整理器负责将长短不一的句子整理成固定批次，包括填充(Padding)、对齐等操作。

#### 常用类型
| 类型 | 适用任务 | 核心功能 |
|------|---------|---------|
| DataCollatorWithPadding | 分类任务 | 动态填充 |
| DataCollatorForTokenClassification | NER任务 | 填充input_ids和labels |
| DataCollatorForSeq2Seq | 翻译/摘要 | 创建decoder_input_ids |
| DataCollatorForLanguageModeling | MLM/CLM | 动态遮盖或偏移 |

#### 标签向右移动(Shift Right)
用于Seq2Seq任务，防止模型"抄答案"：

```
| 时间步 | 解码器输入 | 期望输出 | 解释 |
|:---|:---|:---|:---|
| 第1步 | <BOS> | 苹 | 预测第一个字 |
| 第2步 | <BOS> 苹 | 果 | 预测下一个字 |
| 第3步 | <BOS> 苹果 | <EOS> | 完成预测 |
```

---

### 11. Gradio 应用构建

#### 基础用法
```python
import gradio as gr
from transformers import pipeline

model = pipeline("text-generation")

def predict(prompt):
    return model(prompt)[0]["generated_text"]

gr.Interface(fn=predict, inputs="text", outputs="text").launch()
```

#### 高级功能
- 多输入组件：Audio、Dropdown、Slider等
- 聊天界面：Chatbot组件
- 图像识别：Image + Label组件
- 模型解释：interpretation功能

---

## 三、代码实践总结

### 数据集处理脚本 (s05_*.py)
- GitHub Issues 抓取与处理
- FAISS 语义搜索实现
- 数据清洗与转换

### 分词器训练 (s06_*.py)
- CodeSearchNet Python代码分词器训练
- 对比新旧分词器效果

### 微调脚本 (s07_*.py)
- NER任务微调（BERT + CoNLL-2003）
- 翻译任务微调（MarianMT + KDE4）
- 情感分类微调（DistilBERT + IMDB）
- 问答任务微调（BERT + SQuAD）
- 分布式训练支持（Accelerate库）

---

## 四、学习要点总结

### 核心流程
```
1. 理解任务需求 → 选择合适架构
2. 准备数据集 → 分词处理 → 标签对齐
3. 选择预训练模型 → 配置任务头
4. 微调训练 → 评估优化
5. 部署应用 → Gradio/HuggingFace Hub
```

### 关键决策点
| 场景 | 建议 |
|------|------|
| 通用任务 | 直接用预训练模型 |
| 专业领域 | 先领域适配，再任务微调 |
| 大数据集 | 使用Streaming + Memory Mapping |
| 分布式训练 | 使用Accelerate库 |

### 常见问题解决
- **稀有Token问题**：领域适配微调
- **长文本处理**：滑动窗口 + stride
- **子词对齐**：使用word_ids() + -100屏蔽
- **内存不足**：Streaming模式 + batched处理

---

## 五、参考资源

- [HuggingFace LLM 课程中文版](https://huggingface.co/learn/llm-course/zh-CN/chapter0/1)
- [Transformers 文档](https://huggingface.co/docs/transformers)
- [Datasets 文档](https://huggingface.co/docs/datasets)
- [Tokenizers 文档](https://huggingface.co/docs/tokenizers)
- [Accelerate 文档](https://huggingface.co/docs/accelerate)

---

*本总结基于 HuggingFace 官方 LLM 课程学习笔记整理，涵盖理论原理与实践代码。*