"""
翻译模型微调脚本
使用 Helsinki-NLP/opus-mt-en-fr 模型在 KDE4 数据集上进行英法翻译微调
适配 M系列芯片 Mac，使用 MPS 加速
"""

import torch

# 自动检测设备：M系列芯片 使用 MPS，否则使用 CPU
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

from datasets import load_dataset

# 加载 KDE4 英法平行语料库
raw_datasets = load_dataset("kde4", lang1="en", lang2="fr")
print(raw_datasets)
print(raw_datasets["train"][1000])

# 将训练集按 9:1 划分为训练集和验证集
split_datasets = raw_datasets["train"].train_test_split(train_size=0.9, seed=20)
# 将测试集重命名为验证集
split_datasets["validation"] = split_datasets.pop("test")
print(split_datasets)

from transformers import AutoTokenizer

# 使用预训练的 MarianMT 英法翻译模型
model_checkpoint = "Helsinki-NLP/opus-mt-en-fr"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, return_tensors="pt")

# 序列最大长度
max_length = 128


def preprocess_function(examples):
    """
    数据预处理函数：将翻译样本 tokenize
    输入：英文句子，目标：法文句子
    """
    inputs = [ex["en"] for ex in examples["translation"]]
    targets = [ex["fr"] for ex in examples["translation"]]
    model_input = tokenizer(
        inputs, text_target=targets, max_length=max_length, truncation=True
    )
    return model_input


# 对数据集进行 tokenize，移除原始列，只保留 tokenized 结果
tokenized_datasets = split_datasets.map(preprocess_function,
                                        batched=True,
                                        remove_columns=split_datasets["train"].column_names,
                                        )
print(tokenized_datasets["train"][:5])
print(f"==============load tokenized datasets:{tokenized_datasets}================")

from transformers import AutoModelForSeq2SeqLM

# 加载预训练的 Seq2Seq 模型并移动到 MPS/CPU 设备
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint).to(device)

from transformers import DataCollatorForSeq2Seq

# 数据整理器：动态 padding，将样本整理成批次
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)


def print_collator_samples():
    """
    打印数据整理器处理后的样本，验证数据格式
    """
    samples = [tokenized_datasets["train"][i] for i in range(2)]
    print(samples)
    batch = data_collator(samples)
    print(batch)


print_collator_samples()
print(f"==============print_collator_samples================")

import evaluate

# 加载 SacreBLEU 评估指标，用于评估翻译质量
# 加载之前可以先安装：pip install sacrebleu
metric = evaluate.load("sacrebleu")


def print_metric_test():
    """
    测试 BLEU 指标计算
    BLEU 分数越高表示翻译质量越好
    """
    predictions = [
        "This plugin lets you translate web pages between several languages automatically."
    ]
    references = [
        [
            "This plugin allows you to automatically translate web pages between several languages."
        ]
    ]
    metric_res = metric.compute(predictions=predictions, references=references)
    print(metric_res)

    # 重复单词的预测会有较低的 BLEU 分数
    predictions_2 = ["This This This This"]
    references_2 = [
        [
            "This plugin allows you to automatically translate web pages between several languages."
        ]
    ]
    metric_res_2 = metric.compute(predictions=predictions_2, references=references_2)
    print(metric_res_2)


print_metric_test()
print(f"==============print_metric_test================")

import numpy as np


def compute_metrics(eval_preds):
    """
    计算 BLEU 评估指标
    Args:
        eval_preds: 模型预测结果和标签
    Returns:
        dict: 包含 BLEU 分数的字典
    """
    preds, labels = eval_preds
    # 如果模型返回的内容超过了预测的 logits，只取第一部分
    if isinstance(preds, tuple):
        preds = preds[0]

    # 解码预测结果，跳过特殊 token
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

    # 将标签中的 -100（忽略索引）替换为 pad_token_id 以便解码
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # 后处理：去除前后空白
    decoded_preds = [pred.strip() for pred in decoded_preds]
    # references 需要是列表的列表格式
    decoded_labels = [[label.strip()] for label in decoded_labels]

    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {"bleu": result["score"]}


# ==================== 模型微调配置 ====================

from transformers import Seq2SeqTrainingArguments

# 训练参数配置
args = Seq2SeqTrainingArguments(
    f"marian-finetuned-kde4-en-to-fr",  # 输出目录名称
    eval_strategy="no",  # 不在训练过程中评估（节省时间）
    save_strategy="epoch",  # 每个 epoch 保存一次
    learning_rate=2e-5,  # 学习率
    per_device_train_batch_size=32,  # 每设备训练批次大小
    per_device_eval_batch_size=64,  # 每设备评估批次大小
    weight_decay=0.01,  # 权重衰减（防止过拟合）
    save_total_limit=3,  # 最多保存 3 个 checkpoint
    num_train_epochs=3,  # 训练 3 个 epoch
    predict_with_generate=True,  # 评估时使用生成模式
    fp16=False,  # M1 chip 不支持 CUDA fp16
    use_cpu=False,  # 自动使用 MPS 加速
    push_to_hub=True,  # 训练完成后推送到 Hugging Face Hub
)
print(f"==============print training args:{args}================")

from transformers import Seq2SeqTrainer

# 创建 Trainer，负责训练、评估和保存模型
trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],  # 训练数据集
    eval_dataset=tokenized_datasets["validation"],  # 验证数据集
    data_collator=data_collator,  # 数据整理器
    processing_class=tokenizer,  # tokenizer
    compute_metrics=compute_metrics,  # 评估指标函数
)

# 训练前先评估，查看基线性能
print(trainer.evaluate(max_length=max_length))
print(f"==============print trainer:{trainer}================")

# 开始训练
trainer.train()
print(f"==============train end================")

# 训练后评估，查看提升效果
print(trainer.evaluate(max_length=max_length))

# 将模型推送到 Hugging Face Hub
trainer.push_to_hub(tags="translation", commit_message="Training complete")
print(f"==============push to hub end================")
