"""
翻译模型微调脚本
使用 Helsinki-NLP/opus-mt-en-fr 模型在 KDE4 数据集上进行英法翻译微调
适配 M系列芯片 Mac，使用 MPS 加速

使用 Accelerate 库进行分布式训练支持

注意事项：
- Accelerate 与 MPS 后端存在兼容性问题，建议使用 CPU 进行单机训练
- 如需分布式训练，请使用 CUDA 设备
"""

import torch
import numpy as np  # BUG修复：postprocess函数需要numpy进行标签处理

# 自动检测设备：M系列芯片使用 MPS，否则使用 CPU
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

from datasets import load_dataset

# 加载 KDE4 英法平行语料库
# KDE4 是 KDE 软件项目的本地化翻译数据集，包含大量英法平行句子
raw_datasets = load_dataset("kde4", lang1="en", lang2="fr")
print(raw_datasets)
# 打印一个样本查看数据格式
print(raw_datasets["train"][1000])  # 示例：{'translation': {'en': '...', 'fr': '...'}}

# 将训练集按 9:1 划分为训练集和验证集
split_datasets = raw_datasets["train"].train_test_split(train_size=0.9, seed=20)
# 将测试集重命名为验证集
split_datasets["validation"] = split_datasets.pop("test")
print(split_datasets)

from transformers import AutoTokenizer

# 使用预训练的 MarianMT 英法翻译模型
# MarianMT 是基于 Transformer 的序列到序列翻译模型
model_checkpoint = "Helsinki-NLP/opus-mt-en-fr"
# return_tensors="pt" 参数指定返回 PyTorch 张量格式
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
# BUG修复：结合 Accelerator 的 device_placement=False 设置，
# 手动将模型放在正确设备上，避免 Accelerator 与 MPS 的冲突
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint).to(device)

from transformers import DataCollatorForSeq2Seq

# 数据整理器：动态 padding，将样本整理成批次
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

from torch.utils.data import DataLoader

tokenized_datasets.set_format("torch")
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

from accelerate import Accelerator

# Accelerator 用于管理分布式训练和设备分配
# BUG修复：MPS 后端与 Accelerate 不完全兼容
# 解决方案：检测设备类型，在 MPS 上禁用 Accelerator 的设备迁移
if device == "mps":
    # MPS 设备：使用原生 PyTorch 训练，不依赖 Accelerator 的设备管理
    # Accelerator 主要用于分布式训练，单机 MPS 训练可以直接使用原生代码
    accelerator = Accelerator(device_placement=False)
    # 手动确保模型和数据在正确设备上
else:
    # CPU 或 CUDA：使用 Accelerator 自动管理设备
    accelerator = Accelerator()

# prepare() 会将模型和数据加载器移动到正确的设备，并设置分布式训练
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)

from transformers import get_scheduler

# 训练配置：3个epoch
num_train_epochs = 3
# 每个epoch的更新步数 = 数据集大小 / batch_size
num_update_steps_per_epoch = len(train_dataloader)
# 总训练步数
num_training_steps = num_train_epochs * num_update_steps_per_epoch

# 使用线性学习率调度器，无预热阶段
# 学习率从初始值线性衰减到0
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

from huggingface_hub import HfApi, get_full_repo_name, create_repo

# 设置模型名称和 Hub 仓库
model_name = "marian-finetuned-kde4-en-to-fr-accelerate"
repo_name = get_full_repo_name(model_name)
print(f"================get repo:{repo_name}==============")

output_dir = "marian-finetuned-kde4-en-to-fr-accelerate"

# BUG修复：Repository 类已弃用，使用 HfApi 替代
# HfApi 是新版 huggingface_hub 推荐的 API，提供更简洁的文件上传接口
api = HfApi()

# 确保远程仓库存在（如果不存在则创建）
try:
    create_repo(repo_id=repo_name, exist_ok=True)
    print(f"Repository {repo_name} is ready")
except Exception as e:
    print(f"Warning: Could not create/access repository: {e}")


def postprocess(predictions, labels):
    """
    后处理函数：将模型输出和标签从token IDs转换为文本

    Args:
        predictions: 模型生成的预测token IDs张量
        labels: 真实标签token IDs张量（-100表示被mask的位置）

    Returns:
        decoded_preds: 解码后的预测文本列表
        decoded_labels: 解码后的标签文本列表（每个标签包装在列表中，供BLEU计算）
    """
    # 将张量移到CPU并转为numpy数组（用于decode）
    predictions = predictions.cpu().numpy()
    labels = labels.cpu().numpy()

    # 解码预测结果，跳过特殊token（如padding、eos等）
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)

    # 替换标签中的 -100 为 pad_token_id，因为 -100 无法被解码
    # -100 是 PyTorch cross-entropy loss 默认的 ignore_index
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # 简单后处理：去除首尾空白字符
    decoded_preds = [pred.strip() for pred in decoded_preds]
    # BLEU指标期望references是列表的列表形式
    decoded_labels = [[label.strip()] for label in decoded_labels]
    return decoded_preds, decoded_labels


import evaluate

# 加载 SacreBLEU 评估指标，用于评估翻译质量
# BLEU (Bilingual Evaluation Understudy) 是机器翻译的标准评估指标
metric = evaluate.load("sacrebleu")

from tqdm.auto import tqdm

# 初始化进度条，显示总训练步数
progress_bar = tqdm(range(num_training_steps))

# ==================== 训练循环 ====================
for epoch in range(num_train_epochs):
    # 训练阶段
    model.train()
    for batch in train_dataloader:
        # 前向传播：计算模型输出
        outputs = model(**batch)
        loss = outputs.loss
        # 反向传播：计算梯度（使用Accelerator处理分布式梯度）
        accelerator.backward(loss)

        # 更新参数和学习率
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()  # 清零梯度，防止累积
        progress_bar.update(1)

    # ==================== 评估阶段 ====================
    model.eval()
    # BUG修复：添加 tqdm 描述参数，让进度条更清晰
    for batch in tqdm(eval_dataloader, desc=f"Evaluation epoch {epoch}"):
        # 使用torch.no_grad()禁用梯度计算，节省内存
        with torch.no_grad():
            # 生成翻译结果（使用beam search或其他策略）
            # accelerator.unwrap_model() 获取原始模型（去除分布式包装）
            generated_tokens = accelerator.unwrap_model(model).generate(
                batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_length=128,  # 生成的最大长度
            )
        labels = batch["labels"]

        # 跨进程填充预测和标签，确保形状一致（用于gather）
        generated_tokens = accelerator.pad_across_processes(
            generated_tokens, dim=1, pad_index=tokenizer.pad_token_id
        )
        labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)

        # 收集所有进程的预测和标签（分布式训练同步）
        predictions_gathered = accelerator.gather(generated_tokens)
        labels_gathered = accelerator.gather(labels)

        # 后处理并添加到评估指标
        decoded_preds, decoded_labels = postprocess(predictions_gathered, labels_gathered)
        metric.add_batch(predictions=decoded_preds, references=decoded_labels)

    # 计算并打印BLEU分数
    results = metric.compute()
    print(f"epoch {epoch}, BLEU score: {results['score']:.2f}")

    # ==================== 保存和上传 ====================
    # 等待所有进程完成评估
    accelerator.wait_for_everyone()
    # 获取原始模型（去除分布式包装）
    unwrapped_model = accelerator.unwrap_model(model)
    # 保存模型checkpoint
    unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
    # 只在主进程保存tokenizer和推送到Hub
    if accelerator.is_main_process:
        tokenizer.save_pretrained(output_dir)
        # BUG修复：使用 HfApi 替代已弃用的 Repository.push_to_hub
        # upload_folder 直接上传本地目录到 Hub，更简洁高效
        api.upload_folder(
            folder_path=output_dir,
            repo_id=repo_name,
            commit_message=f"Training in progress epoch {epoch}",
        )

from transformers import pipeline

# ==================== 推理测试 ====================
# 使用训练好的模型进行翻译推理
# 注意：这里使用的是预训练checkpoint作为示例，实际应使用本地训练的模型：
# model_checkpoint = "marian-finetuned-kde4-en-to-fr-accelerate" 或 output_dir
model_checkpoint = "huggingface-course/marian-finetuned-kde4-en-to-fr"
translator = pipeline("translation", model=model_checkpoint)
# 测试翻译一个英文句子
print(translator("Default to expanded threads"))
