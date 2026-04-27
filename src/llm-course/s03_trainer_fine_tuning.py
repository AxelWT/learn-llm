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