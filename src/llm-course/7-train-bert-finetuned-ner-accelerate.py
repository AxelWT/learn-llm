"""
BERT 微调进行命名实体识别（NER）任务
使用 CoNLL-2003 数据集和 Accelerate 库实现分布式训练
"""

# ==================== 第一部分：数据集加载 ====================
from datasets import load_dataset

raw_datasets = load_dataset('conll2003', trust_remote_code=True)
print(raw_datasets)
print(raw_datasets["train"][0])

# ==================== 第二部分：分词器加载 ====================
from transformers import AutoTokenizer

model_checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
print(f"load tokenizer: {tokenizer}")


def align_labels_with_tokens(labels, word_ids):
    """
    将单词级别的标签对齐到 token 级别

    参数:
        labels: 原始单词级别的标签列表（整数索引）
        word_ids: 每个 token 对应的单词索引，None 表示特殊 token

    返回:
        对齐后的 token 级别标签列表

    对齐策略:
        - 特殊 token 标为 -100（不参与损失计算）
        - 每个单词的第一个 token 使用原标签
        - 同一单词后续 token：B-XXX 改为 I-XXX
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


def tokenize_and_align_labels(examples):
    """
    批量处理：分词 + 标签对齐

    参数:
        examples: 包含 "tokens" 和 "ner_tags" 的批次数据

    返回:
        tokenized_inputs: 包含分词结果和已对齐的 "labels" 字段

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


# ==================== 第四部分：执行数据预处理 ====================
tokenized_datasets = raw_datasets.map(tokenize_and_align_labels, batched=True,
                                      remove_columns=raw_datasets["train"].column_names)
print(f"tokenized_datasets: {tokenized_datasets}")

# ==================== 第五部分：数据整理器 ====================
from transformers import DataCollatorForTokenClassification

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
print(f"data_collator: {data_collator}")

# ==================== 第六部分：评估指标加载 ====================
import evaluate

metric = evaluate.load("seqeval")
print(f"metric: {metric}")


def test_eval(datasets):
    """测试 seqeval 指标的计算方式"""
    labels = datasets["train"][0]["ner_tags"]
    label_names = datasets["train"].features["ner_tags"].feature.names
    print(labels)
    print([label_names[i] for i in labels])

    predictions = labels.copy()
    predictions[2] = 0

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
            - logits: 模型输出，形状为 (batch_size, seq_len, num_labels)
            - labels: 真实标签，形状为 (batch_size, seq_len)

    返回:
        dict: 包含 precision, recall, f1, accuracy 四个指标
    """
    label_names = raw_datasets["train"].features["ner_tags"].feature.names
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
    }


# ==================== 第七部分：模型定义 ====================
label_names = raw_datasets["train"].features["ner_tags"].feature.names
id2label = {str(i): label for i, label in enumerate(label_names)}
label2id = {v: k for k, v in id2label.items()}

from transformers import AutoModelForTokenClassification

model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    id2label=id2label,
    label2id=label2id,
)
print(f"load model:{model}")

# ==================== 第八部分：自定义训练循环 ====================
from torch.utils.data import DataLoader

train_dataloader = DataLoader(
    tokenized_datasets["train"],
    shuffle=True,
    collate_fn=data_collator,
    batch_size=8,
)
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], collate_fn=data_collator, batch_size=8
)

from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=2e-5)

import torch

# 检查 MPS (Apple Silicon GPU) 是否可用
if torch.backends.mps.is_available():
    device = "mps"
    print(f"使用 MPS (Apple Silicon GPU) 加速")
elif torch.cuda.is_available():
    device = "cuda"
    print(f"使用 CUDA GPU 加速")
else:
    device = "cpu"
    print(f"使用 CPU 运行（无 GPU 加速）")

from accelerate import Accelerator

# 配置 Accelerator 使用正确的设备
accelerator = Accelerator(device_placement=True)
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)
print(f"模型运行设备: {accelerator.device}")

from transformers import get_scheduler

num_train_epochs = 3
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

from huggingface_hub import create_repo, get_full_repo_name

model_name = "bert-finetuned-ner-accelerate"
repo_name = get_full_repo_name(model_name)
print(repo_name)

output_dir = "bert-finetuned-ner-accelerate"
create_repo(repo_name, repo_type="model", exist_ok=True)


def postprocess(predictions, labels):
    """
    后处理：将模型输出转换为评估格式

    参数:
        predictions: 预测标签 tensor，形状为 (batch_size, seq_len)
        labels: 真实标签 tensor，形状为 (batch_size, seq_len)

    返回:
        true_labels: 真实标签名列表
        true_predictions: 预测标签名列表
    """
    predictions = predictions.detach().cpu().clone().numpy()
    labels = labels.detach().cpu().clone().numpy()
    true_labels = [[label_names[l] for l in label if l != -100] for label in labels]
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    return true_labels, true_predictions


from tqdm.auto import tqdm
import torch

progress_bar = tqdm(range(num_training_steps))

# ==================== 第九部分：训练循环 ====================
for epoch in range(num_train_epochs):
    # 训练阶段
    model.train()
    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    # 评估阶段
    model.eval()
    for batch in eval_dataloader:
        with torch.no_grad():
            outputs = model(**batch)
        predictions = outputs.logits.argmax(dim=-1)
        labels = batch["labels"]
        predictions = accelerator.pad_across_processes(predictions, dim=1, pad_index=-100)
        labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
        predictions_gathered = accelerator.gather(predictions)
        labels_gathered = accelerator.gather(labels)
        true_labels, true_predictions = postprocess(predictions_gathered, labels_gathered)
        metric.add_batch(predictions=true_predictions, references=true_labels)

    results = metric.compute()
    print(
        f"epoch {epoch}:",
        {
            key: results[f"overall_{key}"]
            for key in ["precision", "recall", "f1", "accuracy"]
        },
    )

    # 保存并上传
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
    if accelerator.is_main_process:
        tokenizer.save_pretrained(output_dir)
        unwrapped_model.push_to_hub(
            repo_name, commit_message=f"Training in progress epoch {epoch}"
        )

# ==================== 第十部分：训练完成保存和上传 ====================
# 等待所有进程完成训练
accelerator.wait_for_everyone()

# 解包装模型，获取原始模型
unwrapped_model = accelerator.unwrap_model(model)

# 保存最终模型到本地目录
unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
print(f"模型已保存到本地目录: {output_dir}")

# 主进程执行分词器保存和 Hub 上传
if accelerator.is_main_process:
    # 保存分词器到本地
    tokenizer.save_pretrained(output_dir)
    print(f"分词器已保存到本地目录: {output_dir}")

    # 上传最终模型到 Hugging Face Hub
    unwrapped_model.push_to_hub(
        repo_name,
        commit_message="Training completed - final model"
    )
    print(f"模型已上传到 Hub: {repo_name}")