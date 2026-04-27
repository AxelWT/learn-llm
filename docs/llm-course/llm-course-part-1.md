# 模型微调

## 目录

- [1.Transformer 模型](#1transformer-模型)
- [2.使用 Transformers](#2使用-transformers)
- [3.微调一个预训练模型](#3微调一个预训练模型)
  - [从模型中心加载数据集](#1-从模型中心加载数据集)
  - [预处理数据集 + 动态填充](#2-预处理数据集--动态填充)
  - [使用 Trainer API 微调一个模型](#3-使用-trainer-api-微调一个模型)
  - [一个完整的训练过程](#4-一个完整的训练过程)
- [4.使用Accelerate 加速你的训练循环](#4使用accelerate-加速你的训练循环)
- [5.分享你的模型和标记器](#5分享你的模型和标记器)
- [6.参考文档](#6参考文档)


## 1.Transformer 模型
- 什么是自然语言处理？NLP 是语言学和机器学习交叉领域，专注于理解与人类语言相关的一切，NLP 任务的目标不仅是单独理解单个单词，而且是能够理解这些单词的上下文
- 常见的 NLP 任务，对整个句子进行分类，对句子中每个词进行分类，生成文本内容，从文本中提取答案，从输入文本生成新句子

---

## 2.使用 Transformers
- transformers 库的目标是提供一个统一的 API 接口，通过它可以加载/训练和保存任何 transformer 模型
- tokenizer API 是 pipeline()函数的重要组成部分，负责第一步和最后一步的处理，将文本转换到神经网络的输入以及在需要时将其转换回文本
- transformers 库中 pipeline 处理的三个步骤：使用 tokenizer 进行预处理，通过模型传递输入，后处理；模型无法直接处理原始文本，因此该管道的第一步是将文本转换为模型能够理解的数字，tensor （张量）
- tokenizer, 基于单词的 tokenization无法表示相似词的相似性，基于字符的 tokenization 失去了单词的含义，基于子词的 tokenization 结合了前面二者的优点
- 使用 (注意 tensor 必须是矩形，一维向量（张量）长度一致)

- tokenize示例
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
print(f"词汇表大小: {len(tokenizer.vocab)}")
print(f"词汇表内容示例: {list(tokenizer.vocab.keys())[:20]}")

# 结果
"""
['Using', 'a', 'Trans', '##former', 'network', 'is', 'simple']
[7993, 170, 13809, 23763, 2443, 1110, 3014]
Using a Transformer network is simple
词汇表大小: 28996
词汇表内容示例: ['##up', 'applying', '##ample', 'replaced', 'jumper', 'destroyed', 'convened', 'iPhone', 'therefore', '##master', '##本', 'Bosnia', '##utile', 'spaced', '##roon', 'tensions', 'brows', '##ogan', 'altitude', 'Karnataka']
"""

"""
模型识别文字的方式?

  BERT使用WordPiece分词算法，流程如下：

  1. 分词: 将文本拆分成子词单元
    - 常见词保持完整：Using, a, network, is, simple
    - 长词拆成前缀+后缀：Transformer → Trans + ##former（##表示后缀）
  2. 映射ID: 每个token对应词汇表中的一个固定ID
  3. 解码: ID序列还原为文本

  模型认识的文字范围

  bert-base-cased的词汇表大小是28996个token。

  词汇表内容：
  - 英文常见单词和子词
  - 标点符号
  - 特殊token：[CLS], [SEP], [UNK], [PAD], [MASK]

  限制：未知文字会被标记为[UNK]（unknown）。例如中文输入会产生大量[UNK]，因为词汇表主要是英文。
  
 词汇表存储位置?
 
 tokenizer.json │ JSON格式，model.vocab 字段包含 {token: id} 映射  
                                      
 tokenizer.json 结构                                                                                                                                                                
                                                                                                                                                                                   
  {
    "model": {
      "vocab": {"[PAD]": 0, "[unused1]": 1, ..., "Using": 7993, ...},                                                                                                                
      "unk_token": "[UNK]",                                                                                                                                                          
      "continuing_subword_prefix": "##"                                                                                                                                              
    },                                                                                                                                                                               
    "normalizer": {...},   // 文本规范化规则                                                                                                                                       
    "pre_tokenizer": {...}, // 预分词规则                                                                                                                                            
    "decoder": {...},       // 解码规则                                                                                                                                              
    "added_tokens": [...]   // 新增的特殊token                                                                                                                                       
  }                                                                                                                                                                                                         
  
 tokenizer.json 是HuggingFace统一格式（包含完整分词配置）
 tokenizer.json 是模型的“语言门户”。没有它，模型即使有万亿参数，也无法理解你输入的任何一个字。
"""
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
print(tokens)
output = model(**tokens)
print(output)

"""
{'input_ids': 
tensor([[  101,  1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662, 12172, 2607,  2026,  2878,  2166,  1012,   102],
        [  101,  2061,  2031,  1045,   999,   102,     0,     0,     0,     0, 0,     0,     0,     0,     0,     0],
        [  101,  7592,  2088,   102,     0,     0,     0,     0,     0,     0, 0,     0,     0,     0,     0,     0]]), 
'token_type_ids': 
tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), 
'attention_mask': 
tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])}
        
SequenceClassifierOutput(loss=None, 
logits=tensor([[-1.5607,  1.6123],
        [-3.6183,  3.9137],
        [-3.9943,  4.3083]], 
        grad_fn=<AddmmBackward0>), hidden_states=None, attentions=None)
"""
```

---

## 3.微调一个预训练模型
- 如何使用自己的数据集微调预训练模型呢？从模型中心（hub）加载大型数据集；使用高级 Trainer API 微调一个模型；自定义训练过程；利用 Accelerate 库在所有分布式设备上轻松运行自定义训练过程；

### 1. 从模型中心加载数据集
```Python
# 加载 GLUE 基准测试中的 MRPC（微软研究释义语料库）数据集
# MRPC 是一个句子对分类任务，判断两个句子是否语义相同（ paraphrase 或 not paraphrase）
from datasets import load_dataset

# load_dataset() 从 HuggingFace Hub 下载并加载数据集
# "glue" 是基准测试名称，"mrpc" 是具体子任务
raw_datasets = load_dataset("glue", "mrpc")
print(raw_datasets)  # 打印数据集结构，包含 train、validation、test 三个子集

# 查看训练集中的第 15 条样本
# 每条样本包含：sentence1、sentence2、label（0=不相同，1=相同）、idx
print(raw_datasets["train"][15])

# 查看验证集中的第 87 条样本
print(raw_datasets["validation"][87])

#结果
"""
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

label == 0 表示两句非同义，label == 1 表示两句同义
{'sentence1': 'Rudder was most recently senior vice president for the Developer & Platform Evangelism Business .', 'sentence2': 'Senior Vice President Eric Rudder , formerly head of the Developer and Platform Evangelism unit , will lead the new entity .', 'label': 0, 'idx': 16}
{'sentence1': 'However , EPA officials would not confirm the 20 percent figure .', 'sentence2': 'Only in the past few weeks have officials settled on the 20 percent figure .', 'label': 0, 'idx': 812}
"""
```

### 2. 预处理数据集 + 动态填充
- 预处理数据集的作用是，将数据集提前处理好，生成数字（模型只能读懂数字）保存到内存中（增量文件关联到原数据集文件）
- 动态填充，预处理之后的数据集一批一批加载到内存的同时进行动态填充，以此来保证每批数据的长度统一（模型的要求），同时可以只填充该批的最长长度，比较省空间


```Python
from datasets import load_dataset
from transformers import AutoTokenizer

# 加载的映射（索引），数据不进内存
raw_datasets = load_dataset("glue", "mrpc")
print(raw_datasets)
"""
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
"""
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# 输入句子数据进行分词，生成模型可以读懂的数字数据
ids1 = tokenizer("This is the first sentence.", "This is the second one.")
print(ids1)
"""
{'input_ids': [101, 2023, 2003, 1996, 2034, 6251, 1012, 102, 2023, 2003, 1996, 2117, 2028, 1012, 102], 
'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1], 
'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
"""

# 把数字转换回分词文字
t1 = tokenizer.convert_ids_to_tokens(ids1["input_ids"])
# 多句话都在同一个 input_ids列表中了，token_type_ids的值用来区分每一句话（有些模型有token_type_ids，取决于模型预训练时是否用过）
print(t1)
"""
['[CLS]', 'this', 'is', 'the', 'first', 'sentence', '.', '[SEP]', 'this', 'is', 'the', 'second', 'one', '.', '[SEP]']
"""


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
"""
DatasetDict({
    train: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx', 'input_ids', 'token_type_ids', 'attention_mask'],
        num_rows: 3668
    })
    validation: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx', 'input_ids', 'token_type_ids', 'attention_mask'],
        num_rows: 408
    })
    test: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx', 'input_ids', 'token_type_ids', 'attention_mask'],
        num_rows: 1725
    })
})
{'train': [{'filename': '/Users/<account>/.cache/huggingface/datasets/glue/mrpc/0.0.0/bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c/cache-9c65154dc04d15c8.arrow'}], 
'validation': [{'filename': '/Users/<account>/.cache/huggingface/datasets/glue/mrpc/0.0.0/bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c/cache-c476afed73355cef.arrow'}], 
'test': [{'filename': '/Users/<account>/.cache/huggingface/datasets/glue/mrpc/0.0.0/bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c/cache-49ddf8863e6bf95c.arrow'}]}

"""

# 取预处理后的数据集的前 8 行数据，并去除idx,sentence1,sentence2键
samples = tokenized_datasets["train"][:8]
samples = {k: v for k, v in samples.items() if k not in ["idx", "sentence1", "sentence2"]}
print([len(x) for x in samples["input_ids"]])
"""
结果如下，长度不统一
[50, 59, 47, 67, 59, 50, 62, 32]
"""

# 动态填充
# 因为每个句子长度不同，需要填充到统一长度，该工具会在一个 batch 内找到最长的那条，把其他短的补 0（padding token）到相同长度，这样比预先固定长度更省内存
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 进行动态填充，长度统一
batch2 = data_collator(samples)
print({k: v.shape for k, v in batch2.items()})
"""
{'input_ids': torch.Size([8, 67]), 
'token_type_ids': torch.Size([8, 67]), 
'attention_mask': torch.Size([8, 67]), 
'labels': torch.Size([8])}
"""
```

### 3. 使用 Trainer API 微调一个模型
- transformers 库提供了 Trainer 类，可以帮助你在数据集上微调任何预训练模型
```Python
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

# 加载数据集
# glue 是一个英文语言理解基准测试平台，包含多个 NLP 任务，用于评估预训练语言模型的理解能力。就像 AI 模型的"考试系统"，包含多道" 题目"（子任务）。
# mrpc 是 GLUE 中的一个子任务，微软研究院发布的句子对数据集。 任务类型：二分类 — 判断两个句子是否表达相同含义（是否是 paraphrase）
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

"""
测试每一步流程
from transformers import TrainingArguments
from transformers import AutoModelForSequenceClassification
from transformers import Trainer
import numpy as np
import evaluate

# 训练参数
raining_args = TrainingArguments("test-trainer", num_train_epochs=1, learning_rate=5e-4, per_device_train_batch_size=4)
# 模型，num_labels=2指定分类任务的类别数量
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
# 训练
trainer = Trainer(model,
                  training_args,
                  train_dataset=tokenized_datasets["train"],
                  eval_dataset=tokenized_datasets["validation"],
                  data_collator=data_collator,
                  processing_class=tokenizer)
trainer.train()
# 评估，使用数据集中的验证集对模型进行评估
predictions = trainer.predict(tokenized_datasets["validation"])
print(predictions.predictions.shape, predictions.label_ids.shape)
# 将模型输出的 logits 逻辑值浮点数转化为可以与标签进行比较的预测值，返回数组中最大值所在的索引，`axis=-1`沿最后一个维度（即类别维度）寻找最大值
preds = np.argmax(predictions.predictions, axis=-1)
# 将上一步生成的预测值和真实标签值进行对比计算准确率和 F1
metric = evaluate.load("glue", "mrpc")
metric.compute(predictions=preds, references=predictions.label_ids)
"""

from transformers import TrainingArguments
from transformers import AutoModelForSequenceClassification
from transformers import Trainer
import numpy as np
import evaluate


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

# ========== Baseline 评估（微调前）==========
print("Baseline 评估（微调前的原始模型）")
baseline_val = trainer.evaluate(tokenized_datasets["validation"])
baseline_test = trainer.evaluate(tokenized_datasets["test"])
print(f"Validation 集 baseline: accuracy={baseline_val['eval_accuracy']:.4f}, f1={baseline_val['eval_f1']:.4f}")
print(f"Test 集 baseline: accuracy={baseline_test['eval_accuracy']:.4f}, f1={baseline_test['eval_f1']:.4f}")

"""
Validation 集 baseline: accuracy=0.6838, f1=0.8122
Test 集 baseline: accuracy=0.6649, f1=0.7987
"""
# ========== 开始微调训练 ==========
print("开始微调训练...")
trainer.train()

"""
# 1.评估输出（带 eval_ 前缀）来源: Trainer.evaluate() 或每个 epoch 结束时的自动评估 
# 触发时机: eval_strategy="epoch" 设置后，每个 epoch 结束时自动运行
# 数据来源: validation 集
# 2. 训练输出（无 eval_ 前缀）
# 来源: Trainer.train() 内部的训练循环
# 触发时机: 每隔一定步数（默认每 500 步）
# 数据来源: 训练 batch 的数据   【grad_norm │ 梯度的范数（用于监控梯度爆炸）】

{'eval_loss': '0.4013', 'eval_model_preparation_time': '0.0012', 'eval_accuracy': '0.8309', 'eval_f1': '0.886', 'eval_runtime': '5.73', 'eval_samples_per_second': '71.2', 'eval_steps_per_second': '8.901', 'epoch': '1'}
{'loss': '0.4941', 'grad_norm': '12.77', 'learning_rate': '3.188e-05', 'epoch': '1.089'}

{'eval_loss': '0.5849', 'eval_model_preparation_time': '0.0012', 'eval_accuracy': '0.8333', 'eval_f1': '0.8859', 'eval_runtime': '5.802', 'eval_samples_per_second': '70.33', 'eval_steps_per_second': '8.791', 'epoch': '2'}
'loss': '0.2593', 'grad_norm': '0.1277', 'learning_rate': '1.373e-05', 'epoch': '2.179'}

{'eval_loss': '0.7269', 'eval_model_preparation_time': '0.0012', 'eval_accuracy': '0.8578', 'eval_f1': '0.899', 'eval_runtime': '5.461', 'eval_samples_per_second': '74.71', 'eval_steps_per_second': '9.338', 'epoch': '3'}
{'train_runtime': '320.2', 'train_samples_per_second': '34.37', 'train_steps_per_second': '4.301', 'train_loss': '0.3005', 'epoch': '3'}

"""

# ========== 微调后评估 ==========
print("微调后评估")
after_val = trainer.evaluate(tokenized_datasets["validation"])
after_test = trainer.evaluate(tokenized_datasets["test"])
print(f"Validation 集: accuracy={after_val['eval_accuracy']:.4f}, f1={after_val['eval_f1']:.4f}")
print(f"Test 集: accuracy={after_test['eval_accuracy']:.4f}, f1={after_test['eval_f1']:.4f}")
"""
Validation 集: accuracy=0.8578, f1=0.8990
Test 集: accuracy=0.8458, f1=0.8871
"""
# ========== 效果对比总结 ==========
print("微调效果对比总结")
print(
    f"Validation 集提升: accuracy +{after_val['eval_accuracy'] - baseline_val['eval_accuracy']:.4f}, f1 +{after_val['eval_f1'] - baseline_val['eval_f1']:.4f}")
print(
    f"Test 集提升: accuracy +{after_test['eval_accuracy'] - baseline_test['eval_accuracy']:.4f}, f1 +{after_test['eval_f1'] - baseline_test['eval_f1']:.4f}")

"""
Validation 集提升: accuracy +0.1740, f1 +0.0867
Test 集提升: accuracy +0.1809, f1 +0.0884

1. 相对提升幅度                                                                                                                                                                    
   
  从你的 baseline 到微调后：                                                                                                                                                         
                                                            
  ┌─────────────────────┬──────────┬────────┬───────┬────────────┐                                                                                                                   
  │        指标          │ Baseline │ 微调后 │ 提升    │ 相对提升率 │
  ├─────────────────────┼──────────┼────────┼───────┼────────────┤                                                                                                                   
  │ Validation accuracy │ ~0.66    │ ~0.83  │ +0.17 │ +26%       │
  ├─────────────────────┼──────────┼────────┼───────┼────────────┤                                                                                                                   
  │ Validation f1       │ ~0.80    │ ~0.89  │ +0.09 │ +11%       │                                                                                                                   
  └─────────────────────┴──────────┴────────┴───────┴────────────┘                                                                                                                   
                                                                                                                                                                                     
  相对提升率 = 提升 / baseline，accuracy 提升了 26%，这是明显的。                                                                                                                    
                                                            
  2. 与任务基准对比                                                                                                                                                                  
                                                            
  MRPC 任务的历史水平：                                                                                                                                                              
                                                            
  ┌───────────────────────┬──────────┬─────────┐                                                                                                                                     
  │         模型           │ Accuracy │   F1    │            
  ├───────────────────────┼──────────┼─────────┤                                                                                                                                     
  │ 原始 BERT（未微调）     │ ~60-70%  │ ~75-80% │            
  ├───────────────────────┼──────────┼─────────┤                                                                                                                                     
  │ 微调后 BERT-base       │ 84-85%   │ 88-89%  │                                                                                                                                     
  ├───────────────────────┼──────────┼─────────┤                                                                                                                                     
  │ 更强模型（RoBERTa等）   │ ~90%     │ ~92%    │                                                                                                                                     
  ├───────────────────────┼──────────┼─────────┤                                                                                                                                     
  │ 人类水平               │ ~90%     │ ~92%    │            
  └───────────────────────┴──────────┴─────────┘                                                                                                                                     
                                                            
  你的微调结果（accuracy ~83%, f1 ~89%）接近 BERT-base 的典型水平，说明微调有效。                                                                                                    
                                                            
  3. 判断提升大小的标准                                                                                                                                                              
                                                            
  ┌──────────┬──────────────────┐                                                                                                                                                    
  │ 提升幅度   │       评价       │                           
  ├──────────┼──────────────────┤                                                                                                                                                    
  │ < 1%     │ 微小，可能不显著    │                           
  ├──────────┼──────────────────┤                                                                                                                                                    
  │ 1-5%     │ 小但有意义        │                                                                                                                                                    
  ├──────────┼──────────────────┤
  │ 5-10%    │ 中等，值得注意     │                                                                                                                                                    
  ├──────────┼──────────────────┤                                                                                                                                                    
  │ 10-20%   │ 较大，明显改进     │
  ├──────────┼──────────────────┤                                                                                                                                                    
  │ > 20%    │ 显著提升          │                           
  └──────────┴──────────────────┘                                                                                                                                                    
                                                            
  你的 +17% accuracy 属于较大提升。                                                                                                                                                  
                                                            
  4. 为什么 baseline 较低                                                                                                                                                            
                                                            
  原始 BERT 模型的分类头是随机初始化的，所以 baseline 表现接近随机猜测（二分类约 50%）。微调后分类头被训练，才能发挥作用。
"""
```

### 4. 一个完整的训练过程
- 在不使用transformers库 Trainer 类的情况下实现一样的训练步骤和效果

```Python
# 在不使用 Trainer 类的情况下实现一样的训练步骤和效果

from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
# 以上是数据准备阶段

# 删除模型不需要的列
tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
# 将列名 label 重命名为 labels，因为模型默认的输入是 labels
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
# 设置数据集的格式，使其返回 pytorch 张量而不是列表
tokenized_datasets.set_format("torch")
print(tokenized_datasets["train"].column_names)
"""
['labels', 'input_ids', 'token_type_ids', 'attention_mask']
"""

from torch.utils.data import DataLoader

# 定义训练数据加载器
train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
)
print(len(train_dataloader), len(tokenized_datasets["train"]))
"""
# 第一个返回的是 batch 数量（注意每 batch 是 8 条数据），第二个返回的是总样本数量3668条，训练完所有 459个 batch 算作一次 epoch
459 3668
"""
# 定义评估数据加载器
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
)
# 快速验证数据处理中有没有错误，可以检验其中的一个 batch
for batch in train_dataloader:
    print({k: v.shape for k, v in batch.items()})
    break
"""
{'labels': torch.Size([8]), 'input_ids': torch.Size([8, 65]), 'token_type_ids': torch.Size([8, 65]), 'attention_mask': torch.Size([8, 65])}
"""

from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
# 传入一个 batch 的数据到模型中测试一下，batch 中有 labels 时，transformer 模型都将返回这个 batch 的 loss
for batch in train_dataloader:
    outputs = model(**batch)
    print(outputs.loss, outputs.logits.shape)
    print(outputs)
    break
"""
tensor(0.7678, grad_fn=<NllLossBackward0>) torch.Size([8, 2])
SequenceClassifierOutput(loss=tensor(0.7678, grad_fn=<NllLossBackward0>), 
logits=tensor([[ 0.4507, -0.3374],
        [ 0.4436, -0.3329],
        [ 0.4424, -0.3326],
        [ 0.4530, -0.3285],
        [ 0.4412, -0.3299],
        [ 0.4282, -0.3515],
        [ 0.4315, -0.3450],
        [ 0.4229, -0.3282]], grad_fn=<AddmmBackward0>), hidden_states=None, attentions=None)
"""
# 训练过程
#   1 epoch:
#   ├── batch 1: 8 样本   → 前向传播 → 计算损失 → 反向传播 → 更新权重
#   ├── batch 2: 8 样本   → 前向传播 → 计算损失 → 反向传播 → 更新权重
#   ├── ...
#   ├── batch 458: 8 样本 → 前向传播 → 计算损失 → 反向传播 → 更新权重
#   └── batch 459: 4 样本 → 前向传播 → 计算损失 → 反向传播 → 更新权重
#                       ↑
#                 这就是 1 个 epoch 完成

# 一个 Batch 是一起训练的
#
#   所有样本同时进行前向传播、计算损失、反向传播。
#
#   训练过程
#   一个 batch (8条数据):
#
#   前向传播:
#   ┌─────────────────────────────────────┐
#   │ 数据1 ─┐                              │
#   │ 数据2 ─┤                              │
#   │ 数据3 ─┼──→ 模型 ──→ 8个输出 ──→ 1个loss │
#   │ ...  ─┤      (并行计算)                 │
#   │ 数据8 ─┘                              │
#   └─────────────────────────────────────┘
#               ↓
#   反向传播: 更新一次参数
#
#   代码验证
#
#   for batch in train_dataloader:
#       # batch 包含 8 条数据，形状 [8, 81]
#       outputs = model(**batch)  # 8条数据一起前向传播
#
#       print(outputs.logits.shape)  # torch.Size([8, 2]) - 8条数据的预测结果
#       print(outputs.loss)          # 一个标量 - 8条数据的平均损失
#
#       loss.backward()              # 基于这个平均损失反向传播
#       optimizer.step()             # 更新一次参数
#
#   对比两种方式
#
#   ┌────────────┬───────────────────────┬────────────────────────────┐
#   │    方式     │     参数更新次数        │            特点            │
#   ├────────────┼───────────────────────┼────────────────────────────┤
#   │ Batch 训练  │ 3668/8 = 459 次/epoch │ 并行计算，速度快，梯度稳定     │
#   ├────────────┼───────────────────────┼────────────────────────────┤
#   │ 逐条训练    │ 3668 次/epoch          │ 串行计算，慢，梯度波动大       │
#   └────────────┴───────────────────────┴────────────────────────────┘
#
#   为什么一起训练
#
#   # 损失是 batch 内所有样本的平均
#   loss = (loss_样本1 + loss_样本2 + ... + loss_样本8) / 8
#
#   # 优点:
#   # 1. GPU 并行计算，速度快
#   # 2. 平均梯度更稳定，训练更平滑
#   # 3. 充分利用 GPU 显存带宽
#
#   总结
#
#   ┌──────────┬────────────────────────────────────┐
#   │   概念    │                说明                │
#   ├──────────┼────────────────────────────────────┤
#   │ Batch 内 │ 8 条数据同时计算，产生一个平均损失       │
#   ├──────────┼────────────────────────────────────┤
#   │ 参数更新  │ 每个 batch 结束后更新一次             │
#   ├──────────┼────────────────────────────────────┤
#   │ 1 epoch  │ 参数更新 459 次（= batch 数量）       │
#   └──────────┴────────────────────────────────────┘


# 优化器（使用 AdamW）和学习率调度器
#   AdamW vs Adam：
#   - Adam：经典自适应学习率优化器
#   - AdamW：带权重衰减的 Adam，正则化效果更好，是 BERT 训练的标准选择
from torch.optim import AdamW

# model.parameters()是模型所有可训练参数，lr=5e-5学习率 0.00005（BERT 常用的小学习率）
optimizer = AdamW(model.parameters(), lr=5e-5)

# 学习率调度器
from transformers import get_scheduler

num_epoches = 3
num_training_steps = num_epoches * len(train_dataloader)
print(num_training_steps)
"""
1377
# 训练步数 = epoch 数量 * batch 批次数（一个 batch 全丢进去训练一次）= 3 * 459 = 1377【总训练步数 1377 整个训练过程参数更新 1377 次】
"""

# "linear" # 调度策略
# 学习率变化曲线（linear策略）：
#
# 学习率
# │
# 5e-5 ─────┐
#           │ \
#           │  \
#           │   \
#           │    \
#           │     \
#           │      \
#      ─────└──────└────→ 训练步数
#           0
#                 1377
#
# 对比不同调度策略：
#
# ┌────────────────────────┬──────────────┬──────────────────────┐
# │          策略           │   曲线形状    │         特点          │
# ├────────────────────────┼──────────────┼──────────────────────┤
# │ "linear"               │ 线性下降到0    │ BERT常用，简单有效     │
# ├────────────────────────┼──────────────┼──────────────────────┤
# │ "cosine"               │ 余弦曲线下降   │ 平滑，后期下降慢        │
# ├────────────────────────┼──────────────┼──────────────────────┤
# │ "cosine_with_restarts" │ 余弦重启      │ 多次周期，适合长训练     │
# ├────────────────────────┼──────────────┼──────────────────────┤
# │ "constant"             │ 保持不变      │ 学习率始终5e-5         │
# └────────────────────────┴──────────────┴──────────────────────┘

# num_warmup_steps=0 # 预热步数，设置为 0 则不使用预热，预热的作用： 训练初期学习率从小逐渐增大，避免一开始大学习率破坏预训练权重。
# num_training_steps=num_training_steps  # 总训练步数
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)
# 总结
#
# ┌────────────────────┬──────────────────────────────────────┐
# │        组件         │                 作用                  │
# ├────────────────────┼──────────────────────────────────────┤
# │ AdamW              │ 优化器，决定如何更新参数                 │
# ├────────────────────┼──────────────────────────────────────┤
# │ num_training_steps │ 总训练步数 = epoch × batch数           │
# ├────────────────────┼──────────────────────────────────────┤
# │ lr_scheduler       │ 学习率调度器，让学习率随训练逐渐降低       │
# └────────────────────┴──────────────────────────────────────┘
#
# 为什么需要学习率调度？
# - 训练初期：需要较大学习率快速学习
# - 训练后期：需要小学习率精细调整，避免震荡

# 访问 GPU 设置，目的是加快训练（不过我这台机器目前没有 cuda😭）
# import torch
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# model.to(device)

# 训练循环♻️

# 使用 tqdm 库，在训练步骤数上添加一个进度条
from tqdm.auto import tqdm

progress_bar = tqdm(range(num_training_steps))
# 设置训练模式
# ┌───────────────┬────────────────────────────────────────┐
# │     模式       │                  作用                  │
# ├───────────────┼────────────────────────────────────────┤
# │ model.train() │ 训练模式，启用Dropout、BatchNorm更新      │
# ├───────────────┼────────────────────────────────────────┤
# │ model.eval()  │ 评估模式，关闭Dropout、固定BatchNorm      │
# └───────────────┴────────────────────────────────────────┘
model.train()

for epoch in range(num_epoches):
    for batch in train_dataloader:
        # 使用 GPU 加速，因为本机配置问题，暂不用
        # batch = {k: v.to(device) for k, v in batch.items()}
        # 前向传播
        # **是 Python 的字典解包语法，将字典的键值对展开为函数的关键字参数
        outputs = model(**batch)
        # 输出一个 batch 的平均损失值
        loss = outputs.loss
        # 反向传播，计算梯度（累加到现有梯度）
        loss.backward()
        # 用梯度更新参数
        optimizer.step()
        # 更新学习率
        lr_scheduler.step()
        # 清零梯度，准备下一轮
        optimizer.zero_grad()
        # 训练完一个 batch 更新进度条
        progress_bar.update(1)

# 上述训练循环不会告诉我们任何关于模型目前的状态，我们需要为此添加一个评估循环
# 评估循环
# evaluate 库，这是 Hugging Face 开发的评估库（原名 datasets.metrics），专门用于计算机器学习模型的各种评估指标。
import evaluate, torch

# "glue" │ 基准测试名称（GLUE 是一个包含多个 NLP 任务的评测基准）
# "mrpc" │ 具体任务名称（MRPC = Microsoft Research Paraphrase Corpus）
# 加载 GLUE 基准测试中的 MRPC 任务的评估指标。MRPC 是微软研究院的同义改写语料库，用于判断两个句子是否表达相同意思。指标会计算准确率和 F1 分数。
metric = evaluate.load("glue", "mrpc")
# 设置模型的训练模式为评估模式
model.eval()
for batch in eval_dataloader:
    # batch = {k: v.to(device) for k, v in batch.items()}
    # 遍历数据批次。torch.no_grad() 禁用梯度计算，节省内存并加速推理。
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    # 提取模型的原始输出，然后用 argmax 取最后一维的最大值索引，得到预测类别（0 或 1）。【dim=-1 就是对每一行找最大值的列索引】
    predictions = torch.argmax(logits, dim=-1)
    # 将预测结果和真实标签累积到指标对象中。
    metric.add_batch(predictions=predictions, references=batch["labels"])

# 打印评估结果
print(metric.compute())
"""
{'accuracy': 0.875, 'f1': 0.9122203098106713}
"""
# 微调模型保存
model.save_pretrained("./test-trainer")
tokenizer.save_pretrained("./test-trainer")

# 显式清理资源，避免 Python 退出时崩溃
import gc

del model, optimizer, lr_scheduler
del train_dataloader, eval_dataloader
del tokenized_datasets, raw_datasets

gc.collect()
gc.collect()
print("训练完成，资源已清理")

"""
# 评估示例数据：
labels = torch.tensor([1, 0])

完整示例：

batch = {
    "input_ids": torch.tensor([
        [101, 2023, 2003, ...],
        # 样本0的输入token
        [101, 2054, 2856, ...]
        # 样本1的输入token
    ]),
    "attention_mask": torch.tensor([
        [1, 1, 1, ...],
        [1, 1, 0, ...]
    ]),
    "labels": torch.tensor([1, 0])
}
#              ↑  ↑
#              │  └── 样本1的真实标签：0（不是同义改写）
#              └───── 样本0的真实标签：1（是同义改写）


predictions和labels的对应关系?

predictions = torch.tensor([1, 0])  # 模型预测
labels = torch.tensor([1, 0])  # 真实标签

# 对比：
# 样本0：预测1，真实1 → 正确 ✓
# 样本1：预测0，真实0 → 正确 ✓

# 这批样本全部预测正确

另一个例子（有错误预测）：

predictions = torch.tensor([1, 0, 1, 0])  # 模型预测
labels = torch.tensor([1, 1, 1, 0])  # 真实标签

# 对比：
# 样本0：预测1，真实1 → 正确 ✓
# 样本1：预测0，真实1 → 错误 ✗
# 样本2：预测1，真实1 → 正确 ✓
# 样本3：预测0，真实0 → 正确 ✓

MRPC
标签含义：
- 0 = 两个句子不是同义改写
- 1 = 两个句子是同义改写
"""
```

---

## 4.使用Accelerate 加速你的训练循环
- 之前定义的训练循环在单个 CPU 或者 GPU 上运行良好，通过使用 Accelerate 库，只需进行一些调整，我们就可以在多个 GPU 和 TPU 上启用分布式训练
- 手动训练循环的完整代码
```Python
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, get_scheduler

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
optimizer = AdamW(model.parameters(), lr=3e-5)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

num_epochs = 3
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

progress_bar = tqdm(range(num_training_steps))

model.train()
for epoch in range(num_epochs):
    for batch in train_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)
```
- 加上 accelerate库之后的代码（-表示要删除的代码，+表示新增的代码）
```Python
+ from accelerate import Accelerator
  from transformers import AutoModelForSequenceClassification, get_scheduler

+ accelerator = Accelerator()

  model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
  optimizer = AdamW(model.parameters(), lr=3e-5)

- device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
- model.to(device)

+ train_dataloader, eval_dataloader, model, optimizer = accelerator.prepare(
+     train_dataloader, eval_dataloader, model, optimizer
+ )

  num_epochs = 3
  num_training_steps = num_epochs * len(train_dataloader)
  lr_scheduler = get_scheduler(
      "linear",
      optimizer=optimizer,
      num_warmup_steps=0,
      num_training_steps=num_training_steps
  )

  progress_bar = tqdm(range(num_training_steps))

  model.train()
  for epoch in range(num_epochs):
      for batch in train_dataloader:
-         batch = {k: v.to(device) for k, v in batch.items()}
          outputs = model(**batch)
          loss = outputs.loss
-         loss.backward()
+         accelerator.backward(loss)

          optimizer.step()
          lr_scheduler.step()
          optimizer.zero_grad()
          progress_bar.update(1)
```
---

## 5.分享你的模型和标记器
- 使用预训练模型
- 模型中心 hub 选择合适的模型很简单，只需几行代码即可在任何下游库中使用它

- 从 hub上拉模型下来使用
```Python
# 导入 pipeline 函数，这是 Hugging Face 提供的高层 API，用于快速加载预训练模型
from transformers import pipeline

# - 创建一个"填空"类型的 pipeline
# - model="camembert-base" 指定使用 CamemBERT 模型，这是一个专门针对法语训练的 BERT 模型
camembert_fill_mask = pipeline("fill-mask", model="camembert-base")
# 输入法语句子，其中 <mask> 是需要预测的词位
results = camembert_fill_mask("Le camembert est <mask> :)")
# 输出预测结果，通常返回一个列表，包含多个候选词及其概率
print(results)
"""
[{'score': 0.49091655015945435, 'token': 7200, 'token_str': 'délicieux', 'sequence': 'Le camembert est délicieux :)'}, 
{'score': 0.10557064414024353, 'token': 2183, 'token_str': 'excellent', 'sequence': 'Le camembert est excellent :)'}, 
{'score': 0.03453359007835388, 'token': 26202, 'token_str': 'succulent', 'sequence': 'Le camembert est succulent :)'}, 
{'score': 0.03303172439336777, 'token': 528, 'token_str': 'meilleur', 'sequence': 'Le camembert est meilleur :)'}, 
{'score': 0.030076900497078896, 'token': 1654, 'token_str': 'parfait', 'sequence': 'Le camembert est parfait :)'}]

"""

# 还可以使用模型架构实例化 checkpoint，建议使用 Auto*类，因为它们在设计时不依赖模型架构
from transformers import AutoTokenizer, AutoModelForMaskedLM
tokenizer = AutoTokenizer.from_pretrained("camembert-base")
model = AutoModelForMaskedLM.from_pretrained("camembert-base")
```

- 分享预训练的模型
- huggingface客户端方式推送到 hub 仓库
```bash

# Install the Hugging Face CLI
brew install hf

# (optional) Login with your Hugging Face credentials
hf auth login

# Push your model files
hf upload <your-account>/<your repository name> . 
```

- python代码方式推送
```Python
from huggingface_hub import login, upload_folder

# (optional) Login with your Hugging Face credentials
login('<token-with-write-access>')

# Push your model files
upload_folder(folder_path=".", repo_id="<your-account>/<your repository name>", repo_type="model")

```

- git 方式推送
```bash
# Make sure git-xet is installed (https://hf.co/docs/hub/git-xet)
git xet install

git clone https://huggingface.co/<your-account>/<your repository name>

# You'll be prompted for your HF credentials
git push

```

---
- 构建模型卡片
- hub网页端进行编辑

## 6.参考文档
- huggingface上著名的 llm学习课程，中文版链接：https://huggingface.co/learn/llm-course/zh-CN/chapter0/1