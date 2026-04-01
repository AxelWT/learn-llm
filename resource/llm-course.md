- huggingface上著名的 llm学习课程，中文版链接：https://huggingface.co/learn/llm-course/zh-CN/chapter0/1

### 阅读笔记

#### 总结

#### 1.Transformer 模型
- 什么是自然语言处理？NLP 是语言学和机器学习交叉领域，专注于理解与人类语言相关的一切，NLP 任务的目标不仅是单独理解单个单词，而且是能够理解这些单词的上下文
- 常见的 NLP 任务，对整个句子进行分类，对句子中每个词进行分类，生成文本内容，从文本中提取答案，从输入文本生成新句子

#### 2.使用 Transformers
- transformers 库的目标是提供一个统一的 API 接口，通过它可以加载/训练和保存任何 transformer 模型
- tokenizer API 是 pipeline()函数的重要组成部分，负责第一步和最后一步的处理，将文本转换到神经网络的输入以及在需要时将其转换回文本
- transformers 库中 pipeline 处理的三个步骤：使用 tokenizer 进行预处理，通过模型传递输入，后处理；模型无法直接处理原始文本，因此该管道的第一步是将文本转换为模型能够理解的数字，tensor （张量）
- tokenizer, 基于单词的 tokenization无法表示相似词的相似性，基于字符的 tokenization 失去了单词的含义，基于子词的 tokenization 结合了前面二者的优点
- 使用 (注意 tensor 必须是矩形，一维向量（张量）长度一致)
```Python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
sequence = "Using a Transformer network is simple"
tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)
decoded_string = tokenizer.decode(ids)
print(tokens)
print(ids)
print(decoded_string)

# 结果
['Using', 'a', 'Trans', '##former', 'network', 'is', 'simple']
[7993, 170, 13809, 23763, 2443, 1110, 3014]
Using a Transformer network is simple
```
- 从 tokenizer 到模型
```Python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]

# tokenizer()函数内部自动做了填充（保证不同句子分词后的 token 列表长度一致）
# tokenizer()函数内部自动做了截断（保证输入不会超过模型可以接受的最大 token 数量）
# tokenizer()函数内部自动做了张量转换，先把句子分词，再把分词映射为数字，再把数字转换为 pytorch tensor，因为模型只接受张量 tensor 输入
tokens = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
output = model(**tokens)
print(output)

#结果
SequenceClassifierOutput(loss=None, logits=tensor([[-1.5607,  1.6123],
        [-3.6183,  3.9137]], grad_fn=<AddmmBackward0>), hidden_states=None, attentions=None)
```

#### 3.微调一个预训练模型
- 如何使用自己的数据集微调预训练模型呢？从模型中心（hub）加载大型数据集；使用高级 Trainer API 微调一个模型；自定义训练过程；利用 Accelerate 库在所有分布式设备上轻松运行自定义训练过程；
- 

#### 4.分享你的模型和标记器

#### 5.DATASETS 库

#### 6.TOKENIZERS 库

#### 7.主要的 NLP 任务

#### 8.如何寻求帮助

#### 9.构建并分享你的模型

