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