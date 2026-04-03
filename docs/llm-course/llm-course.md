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

1. 从模型中心加载数据集
```Python
from datasets import load_dataset
from transformers import AutoTokenizer

raw_datasets = load_dataset("glue", "mrpc")
print(raw_datasets)

print(raw_datasets["train"][15])
print(raw_datasets["validation"][87])

#结果
DatasetDict({
    train: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx'],
        num_rows: 3668
    })
    validation: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx'],
        num_rows: 408
    })
    test: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx'],
        num_rows: 1725
    })
})
{'sentence1': 'Rudder was most recently senior vice president for the Developer & Platform Evangelism Business .', 'sentence2': 'Senior Vice President Eric Rudder , formerly head of the Developer and Platform Evangelism unit , will lead the new entity .', 'label': 0, 'idx': 16}
{'sentence1': 'However , EPA officials would not confirm the 20 percent figure .', 'sentence2': 'Only in the past few weeks have officials settled on the 20 percent figure .', 'label': 0, 'idx': 812}

```

2. 预处理数据集 + 动态填充
- 预处理数据集的作用是，将数据集提前处理好，生成数字（模型只能读懂数字）保存到内存中（增量文件关联到原数据集文件）
- 动态填充，预处理之后的数据集一批一批加载到内存的同时进行动态填充，以此来保证每批数据的长度统一（模型的要求），同时可以只填充该批的最长长度，比较省空间


```Python
from datasets import load_dataset
from transformers import AutoTokenizer

# 加载的映射（索引），数据不进内存
raw_datasets = load_dataset("glue", "mrpc")
print(raw_datasets)

checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# 输入句子数据进行分词，生成模型可以读懂的数字数据
ids1 = tokenizer("This is the first sentence.", "This is the second one.")
print(ids1)
# 把数字转换回分词文字
t1 = tokenizer.convert_ids_to_tokens(ids1["input_ids"])
# 多句话都在同一个 input_ids列表中了，token_type_ids的值用来区分每一句话（有些模型有token_type_ids，取决于模型预训练时是否用过）
print(t1)


# 预处理训练数据集

# 首先raw_datasets = load_dataset("glue", "mrpc")方式下载的数据集会保存在本地磁盘，以 Apache Arrow 格式
# 以这种方式将磁盘中的数据集所有数据加载到内存中，可能内存会不够用
# tokenized_dataset = tokenizer(raw_datasets["train"]["sentence1"], raw_datasets["train"]["sentence2"], padding=True, truncation=True)

def tokenize_function(example):
    # 之前直接 tokenizer(raw_datasets["train"]["sentence1"]) 报错，是因为 Arrow 类型不被识别。
    # 而 .map() 方法内部会自动处理类型转换，把 Arrow 数据转成 Python 原生类型再传给你的函数，所以 example["sentence1"] 实际上已经是 list[str]，tokenizer 可以正常处理。
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


# 将数据集分成多个 batch（默认每批 1000条），对每个 batch 调用tokenize_function，将返回的 tokenization 结果合并到原始数据集中得到新的数据集
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
print(tokenized_datasets)
print(tokenized_datasets.cache_files)

# 动态填充
# 因为每个句子长度不同，需要填充到统一长度，该工具会在一个 batch 内找到最长的那条，把其他短的补 0（padding token）到相同长度，这样比预先固定长度更省内存
from transformers import DataCollatorWithPadding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 取预处理后的数据集的前 8 行数据，并去除idx,sentence1,sentence2键
samples = tokenized_datasets["train"][:8]
samples = {k: v for k, v in samples.items() if k not in ["idx", "sentence1", "sentence2"]}
# 结果如下[50, 59, 47, 67, 59, 50, 62, 32]，长度不统一
print([len(x) for x in samples["input_ids"]])
# 进行动态填充，长度统一了{'input_ids': torch.Size([8, 67]), 'token_type_ids': torch.Size([8, 67]), 'attention_mask': torch.Size([8, 67]), 'labels': torch.Size([8])}
batch2 = data_collator(samples)
print({k: v.shape for k, v in batch2.items()})


```

3. 使用 Trainer API 或者 Keras 微调一个模型
- transformers 库提供了 Trainer 类，可以帮助你在数据集上微调任何预训练模型
```Python
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

# 加载数据集
raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
# 加载指定模型的分词器
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


# 数据集预处理
def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
# 数据batch动态填充工具
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

from transformers import TrainingArguments

# 专门用来设置训练参数的工具类，创建一个训练参数配置对象，参数“test-trainer”是输出目录的名字，也即模型保存的位置
# training_args = TrainingArguments("test-trainer", num_train_epochs=1, learning_rate=5e-4, per_device_train_batch_size=4)
from transformers import AutoModelForSequenceClassification

# num_labels=2
# model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

from transformers import Trainer

# 训练
# trainer = Trainer(model,
#                   training_args,
#                   train_dataset=tokenized_datasets["train"],
#                   eval_dataset=tokenized_datasets["validation"],
#                   data_collator=data_collator,
#                   processing_class=tokenizer)
# trainer.train()

# 评估

# 使用数据集中的验证集对模型进行评估
# predictions = trainer.predict(tokenized_datasets["validation"])
# print(predictions.predictions.shape, predictions.label_ids.shape)

# 将模型输出的 logits 逻辑值浮点数转化为可以与标签进行比较的预测值，返回数组中最大值所在的索引，`axis=-1`沿最后一个维度（即类别维度）寻找最大值
import numpy as np

# preds = np.argmax(predictions.predictions, axis=-1)
# 将上一步生成的预测值和真实标签值进行对比计算准确率和 F1
import evaluate


# metric = evaluate.load("glue", "mrpc")
# metric.compute(predictions=preds, references=predictions.label_ids)

# 总结上述评估步骤，给出总的评估配置
def compute_metrics(eval_preds):
    metric = evaluate.load("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


training_args = TrainingArguments("test-trainer", eval_strategy="epoch")
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

trainer = Trainer(model,
                  training_args,
                  train_dataset=tokenized_datasets["train"],
                  eval_dataset=tokenized_datasets["validation"],
                  data_collator=data_collator,
                  processing_class=tokenizer,
                  compute_metrics=compute_metrics, )
trainer.train()

```

4. 一个完整的训练过程
- 在不使用transformers库 Trainer 类的情况下实现一样的训练步骤和效果

```Python
# todo
```

#### 4.分享你的模型和标记器

#### 5.DATASETS 库

#### 6.TOKENIZERS 库

#### 7.主要的 NLP 任务

#### 8.如何寻求帮助

#### 9.构建并分享你的模型

