from datetime import time, datetime

from datasets import load_dataset

# ===================== 加载 CoNLL-2003 数据集 =====================
# CoNLL-2003 是经典的 NER（命名实体识别）数据集
# 包含人名(PER)、组织名(ORG)、地名(LOC)、杂项(MISC)四种实体类型
raw_datasets = load_dataset('conll2003', trust_remote_code=True)
print(raw_datasets)
print(raw_datasets["train"][0])


# 展示数据样本及其 POS 标签（词性标注）
def show_label(datasets):
    """将单词和对应标签对齐打印，便于查看"""
    pos_feature = datasets["train"].features["pos_tags"]
    print(pos_feature)
    label_names = pos_feature.feature.names
    words = datasets["train"][0]["tokens"]
    labels = datasets["train"][0]["pos_tags"]
    line1 = ""
    line2 = ""

    for word, label in zip(words, labels):
        full_label = label_names[label]
        max_length = max(len(word), len(full_label))
        line1 += word + " " * (max_length - len(word) + 1)
        line2 += full_label + " " * (max_length - len(full_label) + 1)

    print(line1)
    print(line2)


# show_label(raw_datasets)


# ===================== 加载分词器 =====================
from transformers import AutoTokenizer

# 使用 bert-base-cased 预训练模型（保留大小写信息，对 NER 更有利）
model_checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
print(f"load tokenizer: {tokenizer}")


# ===================== 标签对齐函数 =====================
# BERT 分词器会将单词拆分成多个 token（如 "playing" → ["play", "ing"]）
# 原始数据是单词级别的标签，分词后需要将标签扩展到 token 级别


def align_labels_with_tokens(labels, word_ids):
    """
    将单词级别的标签对齐到 token 级别

    参数:
        labels: 原始单词级别的标签列表（整数索引）
        word_ids: 每个 token 对应的单词索引，None 表示特殊 token（如 [CLS], [SEP]）

    返回:
        对齐后的 token 级别标签列表

    对齐策略:
        - 特殊 token 标为 -100（不参与损失计算）
        - 每个单词的第一个 token 使用原标签
        - 同一单词后续 token：B-XXX 改为 I-XXX（保持实体连续性）
    """
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # 新单词的开始
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # 特殊 token
            new_labels.append(-100)
        else:
            # 同一单词的后续 token
            label = labels[word_id]
            # B-XXX 标签（奇数）改为 I-XXX（偶数+1）
            if label % 2 == 1:
                label += 1
            new_labels.append(label)

    return new_labels


def align_labels_with_tokens_opt(labels, word_ids):
    """
    简化版标签对齐：同一单词后续 token 全部标为 -100

    为什么用 -100？
        - -100 会被 PyTorch 损失函数特殊处理，不参与计算
        - 模型仍会预测这些位置，但训练时不计算损失
        - 防止长单词（拆分后 token 多）过度影响训练

    适用场景:
        - 当你只想用每个单词的第一个 token 来训练时
        - 减少长单词对模型的影响
    """
    aligned_labels = []
    previous_label = None

    for word_id in word_ids:
        if word_id is None:
            aligned_labels.append(-100)
        elif word_id != previous_label:
            aligned_labels.append(labels[word_id])
        else:
            aligned_labels.append(-100)
        previous_label = word_id
    return aligned_labels


# ===================== 测试函数：验证标签对齐 =====================
def test_align_labels(datasets):
    """验证单个样本的分词和标签对齐结果"""
    inputs = tokenizer(datasets["train"][0]["tokens"], is_split_into_words=True)
    print(inputs.tokens())  # 分词后的 token 列表
    labels = datasets["train"][0]["ner_tags"]  # 原始标签
    word_ids = inputs.word_ids()  # token 与单词的对应关系
    print(labels)  # 原始标签
    print(align_labels_with_tokens_opt(labels, word_ids))  # 对齐后的标签


# test_align_labels(raw_datasets)


def tokenize_and_align_labels(examples):
    """
    批量处理：分词 + 标签对齐

    参数:
        examples: 包含 “tokens” 和 “ner_tags” 的批次数据

    返回:
        tokenized_inputs: 包含分词结果和已对齐的 “labels” 字段

    流程:
        1. 使用 Fast Tokenizer 对句子进行分词
        2. 通过 word_ids() 获取每个 token 对应的单词索引
        3. 使用 align_labels_with_tokens 将标签扩展到 token 级别
    """
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )
    all_labels = examples["ner_tags"]
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs["labels"] = new_labels
    return tokenized_inputs


# ===================== 执行数据预处理 =====================
# 使用 map 方法批量处理整个数据集
# batched=True：启用批处理，大幅提升处理速度（Fast Tokenizer 的优势）
# remove_columns：移除原始列，只保留分词后的列（input_ids, attention_mask, labels）
tokenized_datasets = raw_datasets.map(tokenize_and_align_labels, batched=True,
                                      remove_columns=raw_datasets["train"].column_names)
print(f"tokenized_datasets: {tokenized_datasets}")

# ===================== 数据整理器 =====================
from transformers import DataCollatorForTokenClassification

# 创建数据整理器，用于将多个样本整理成一个批次
# 主要功能：动态填充，使同一批次内所有样本长度一致
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
print(f"data_collator: {data_collator}")

# ===================== 评估指标加载 =====================
import evaluate

# 加载 seqeval 评估指标，专为序列标注任务设计
# seqeval 会按实体级别评估（如 B-PER + I-PER 组成一个完整实体）
metric = evaluate.load("seqeval")
print(f"metric: {metric}")


# 测试评估指标函数
def test_eval(datasets):
    """测试 seqeval 指标的计算方式"""
    # 获取第一条样本的 NER 标签（整数索引）
    labels = datasets["train"][0]["ner_tags"]
    label_names = datasets["train"].features["ner_tags"].feature.names
    print(labels)
    print([label_names[i] for i in labels])

    # 模拟预测结果：将第3个标签改为 0（"O"），模拟预测错误
    predictions = labels.copy()
    predictions[2] = 0

    # seqeval 要求输入字符串标签名，而非整数索引
    predictions_str = [label_names[i] for i in predictions]
    labels_str = [label_names[i] for i in labels]
    print(metric.compute(predictions=[predictions_str], references=[labels_str]))


# test_eval(raw_datasets)

import numpy as np


def compute_metrics(eval_preds):
    """
    计算模型评估指标（用于 Trainer 的 eval 步骤）

    参数:
        eval_preds: 包含 (logits, labels) 的元组
            - logits: 模型输出的预测概率，形状为 (batch_size, seq_len, num_labels)
            - labels: 真实标签，形状为 (batch_size, seq_len)，包含 -100 表示忽略的位置

    返回:
        dict: 包含 precision, recall, f1, accuracy 四个指标
    """
    # 获取标签名称列表，用于将数字索引转换为字符串标签
    label_names = raw_datasets["train"].features["ner_tags"].feature.names

    # 解包预测结果和真实标签
    logits, labels = eval_preds

    # 对每个位置取概率最大的标签索引作为预测结果
    # axis=-1 表示沿着最后一个维度（标签类别维度）取最大值
    predictions = np.argmax(logits, axis=-1)

    # 处理真实标签：过滤掉 -100（特殊 token 和填充位置），并将索引转为标签名
    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]

    # 处理预测标签：同样过滤掉对应真实标签为 -100 的位置，避免影响评估
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    # 使用 seqeval 计算序列标注的评估指标
    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)

    # 返回整体指标（seqeval 还会返回每个实体类型的详细指标）
    return {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
    }


# ===================== 模型定义 =====================
# 获取标签名称列表
label_names = raw_datasets["train"].features["ner_tags"].feature.names

# 创建标签索引到标签名的映射字典（用于模型输出解码）
# 例如：{0: "O", 1: "B-PER", 2: "I-PER", ...}
id2label = {str(i): label for i, label in enumerate(label_names)}

# 创建标签名到索引的反向映射字典（用于模型输入编码）
label2id = {v: k for k, v in id2label.items()}

from transformers import AutoModelForTokenClassification

# 加载预训练的 BERT 模型，并配置为 token 分类任务（NER）
# num_labels 会自动根据 id2label 的长度设置
model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    id2label=id2label,
    label2id=label2id,
)
print(f"load model:{model}")

# 打印模型配置的标签数量，验证配置是否正确
print(model.config.num_labels)

from transformers import TrainingArguments

# ===================== 训练参数配置 =====================
args = TrainingArguments(
    "bert-finetuned-ner",  # 输出目录，保存模型和日志
    eval_strategy="epoch",  # 每个 epoch 结束后进行评估
    save_strategy="epoch",  # 每个 epoch 结束后保存模型
    learning_rate=2e-5,  # 学习率，BERT 微调常用 2e-5
    num_train_epochs=3,  # 训练轮数
    weight_decay=0.01,  # 权重衰减（正则化），防止过拟合
    push_to_hub=True,  # 训练完成后自动推送到 Hugging Face Hub
)

print(f"train begins:{datetime.now()}")
from transformers import Trainer

# ===================== 创建训练器并开始训练 =====================
trainer = Trainer(
    model=model,  # 要训练的模型
    args=args,  # 训练参数配置
    train_dataset=tokenized_datasets["train"],  # 训练数据集
    eval_dataset=tokenized_datasets["validation"],  # 验证数据集
    data_collator=data_collator,  # 数据整理器，负责填充和批处理
    compute_metrics=compute_metrics,  # 评估指标计算函数
    processing_class=tokenizer,  # 分词器（新版 transformers 使用 processing_class 参数名）
)

# 开始训练
trainer.train()
print(f"train ends:{datetime.now()}")

trainer.save_model()
trainer.push_to_hub(commit_message="training completed")
print("push model to hub success")

# 使用模型，可以使用远程 hub 的（要拉下来，比较慢），也可以使用本地的
from transformers import pipeline
from rich import print

# 使用本地保存的模型（训练完成后的最终模型）
# 如果想使用某个 checkpoint，可以改为 "./bert-finetuned-ner/checkpoint-5268"
token_classifier = pipeline("token-classification", model="./bert-finetuned-ner", aggregation_strategy="simple")

res = token_classifier("My name is Sylvain and I work at Hugging Face in Brooklyn.")
print(res)
