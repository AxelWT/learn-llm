# ========================================
# SQuAD 问答模型训练脚本（使用 Accelerate 自定义循环）
# 使用 BERT 模型在 SQuAD 数据集上进行问答任务训练
# 特点：使用 Accelerate 库实现自定义训练循环，不使用 Trainer
# ========================================

print("=" * 60)
print("步骤1: 加载 SQuAD 问答数据集")
print("=" * 60)

from datasets import load_dataset

raw_datasets = load_dataset("squad")

print("数据集结构:")
print(raw_datasets)
print(f"训练集样本数: {len(raw_datasets['train'])}")
print(f"验证集样本数: {len(raw_datasets['validation'])}")

# 展示第一个样本
print("\n第一个样本:")
print("Context: ", raw_datasets["train"][0]["context"][:100] + "...")
print("Question: ", raw_datasets["train"][0]["question"])
print("Answer: ", raw_datasets["train"][0]["answers"])


# ========================================
# 步骤2: 加载分词器
# ========================================

print("\n" + "=" * 60)
print("步骤2: 加载 BERT 分词器")
print("=" * 60)

from transformers import AutoTokenizer

model_checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

print(f"分词器: {model_checkpoint}")
print(f"词汇表大小: {len(tokenizer)}")


# ========================================
# 步骤3: 定义预处理参数
# ========================================

print("\n" + "=" * 60)
print("步骤3: 定义预处理参数")
print("=" * 60)

max_length = 384  # 每个样本的最大长度
stride = 128      # 滑动窗口步长（重叠区域）

print(f"最大长度: {max_length}")
print(f"滑动窗口步长: {stride}")


# ========================================
# 步骤4: 定义训练数据预处理函数
# ========================================

print("\n" + "=" * 60)
print("步骤4: 定义训练数据预处理函数")
print("=" * 60)

def preprocess_training_examples(examples):
    """
    训练数据预处理：
    1. 分词问题和上下文，使用滑动窗口处理长文本
    2. 计算答案在 token 序列中的起始和结束位置

    输出：input_ids, attention_mask, start_positions, end_positions
    """
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",      # 只截断上下文
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,   # token 到字符位置的映射
        padding="max_length",
    )

    # offset_mapping: 每个 token 对应原始文本中的位置 (start_char, end_char)
    offset_mapping = inputs.pop("offset_mapping")
    # overflow_to_sample_mapping: 每个 tokenized 片段对应哪个原始样本
    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers = examples["answers"]

    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        # 答案在原始文本中的字符位置
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        # sequence_ids: 0=问题部分，1=上下文部分，None=特殊token
        sequence_ids = inputs.sequence_ids(i)

        # ===== 找到上下文在 token 序列中的范围 =====
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # ===== 判断答案是否完全在上下文中 =====
        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            # 答案不在当前片段，标签为 (0, 0)
            start_positions.append(0)
            end_positions.append(0)
        else:
            # ===== 将字符位置转换为 token 位置 =====
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs

print("训练数据预处理函数定义完成！")


# ========================================
# 步骤5: 执行训练数据预处理
# ========================================

print("\n" + "=" * 60)
print("步骤5: 执行训练数据预处理")
print("=" * 60)

train_dataset = raw_datasets["train"].map(
    preprocess_training_examples,
    batched=True,
    remove_columns=raw_datasets["train"].column_names,
)

print(f"原始训练集样本数: {len(raw_datasets['train'])}")
print(f"预处理后训练集样本数: {len(train_dataset)}")
# 样本数增加是因为长上下文被滑动窗口分成多个片段


# ========================================
# 步骤6: 定义验证数据预处理函数
# ========================================

print("\n" + "=" * 60)
print("步骤6: 定义验证数据预处理函数")
print("=" * 60)

def preprocess_validation_examples(examples):
    """
    验证数据预处理：
    - 与训练预处理类似，但保留 example_id 和 offset_mapping
    - 用于后处理阶段将 token 位置映射回原始文本位置
    """
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = inputs.pop("overflow_to_sample_mapping")
    example_ids = []

    for i in range(len(inputs["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(examples["id"][sample_idx])

        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        # 只保留上下文部分的 offset_mapping，问题部分设为 None
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]

    inputs["example_id"] = example_ids
    return inputs

print("验证数据预处理函数定义完成！")


# ========================================
# 步骤7: 执行验证数据预处理
# ========================================

print("\n" + "=" * 60)
print("步骤7: 执行验证数据预处理")
print("=" * 60)

validation_dataset = raw_datasets["validation"].map(
    preprocess_validation_examples,
    batched=True,
    remove_columns=raw_datasets["validation"].column_names,
)

print(f"原始验证集样本数: {len(raw_datasets['validation'])}")
print(f"预处理后验证集样本数: {len(validation_dataset)}")


# ========================================
# 步骤8: 定义评估指标计算函数
# ========================================

print("\n" + "=" * 60)
print("步骤8: 定义评估指标计算函数")
print("=" * 60)

import collections
import numpy as np
import evaluate
from tqdm.auto import tqdm

# 加载 SQuAD 评估指标
metric = evaluate.load("squad")

# 后处理参数
n_best = 20             # 考虑前 20 个最佳候选答案
max_answer_length = 30  # 答案最大长度

def compute_metrics(start_logits, end_logits, features, examples):
    """
    计算 SQuAD 评估指标（Exact Match 和 F1）：
    1. 建立 sample_id 到 feature 索引的映射
    2. 从 logits 提取最佳答案
    3. 计算 EM 和 F1
    """
    # 每个原始样本可能对应多个特征（滑动窗口）
    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)

    predicted_answers = []
    for example in tqdm(examples, desc="计算评估指标"):
        example_id = example["id"]
        context = example["context"]
        answers = []

        # 遍历该样本对应的所有特征
        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]

            # 获取 logits 最高的 n_best 个索引
            start_indexes = np.argsort(start_logit)[-1: -n_best - 1: -1].tolist()
            end_indexes = np.argsort(end_logit)[-1: -n_best - 1: -1].tolist()

            for start_index in start_indexes:
                for end_index in end_indexes:
                    # 跳过问题部分（offset 为 None）
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    # 跳过无效答案
                    if (
                            end_index < start_index
                            or end_index - start_index + 1 > max_answer_length
                    ):
                        continue

                    answer = {
                        "text": context[offsets[start_index][0]: offsets[end_index][1]],
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    }
                    answers.append(answer)

        # 选择得分最高的答案
        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append(
                {"id": example_id, "prediction_text": best_answer["text"]}
            )
        else:
            predicted_answers.append({"id": example_id, "prediction_text": ""})

    theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
    return metric.compute(predictions=predicted_answers, references=theoretical_answers)

print("评估指标计算函数定义完成！")


# ========================================
# 步骤9: 创建 DataLoader
# ========================================

print("\n" + "=" * 60)
print("步骤9: 创建 DataLoader")
print("=" * 60)

from torch.utils.data import DataLoader
from transformers import default_data_collator

# 设置数据格式为 PyTorch tensor
train_dataset.set_format("torch")

# 验证集需要移除 example_id 和 offset_mapping（模型输入不需要）
validation_set = validation_dataset.remove_columns(["example_id", "offset_mapping"])
validation_set.set_format("torch")

# 创建 DataLoader
train_dataloader = DataLoader(
    train_dataset,
    shuffle=True,
    collate_fn=default_data_collator,
    batch_size=8,
)
eval_dataloader = DataLoader(
    validation_set,
    collate_fn=default_data_collator,
    batch_size=8
)

print(f"训练 DataLoader 批次数: {len(train_dataloader)}")
print(f"验证 DataLoader 批次数: {len(eval_dataloader)}")


# ========================================
# 步骤10: 初始化模型
# ========================================

print("\n" + "=" * 60)
print("步骤10: 初始化 BERT 问答模型")
print("=" * 60)

import torch
from transformers import AutoModelForQuestionAnswering

model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
model_size = sum(t.numel() for t in model.parameters())
print(f"模型: {model_checkpoint}")
print(f"模型参数量: {model_size / 1000 ** 2:.1f}M")


# ========================================
# 步骤11: 配置优化器
# ========================================

print("\n" + "=" * 60)
print("步骤11: 配置优化器")
print("=" * 60)

from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=2e-5)
print(f"优化器: AdamW")
print(f"学习率: 2e-5")


# ========================================
# 步骤12: 配置 Accelerator
# ========================================

print("\n" + "=" * 60)
print("步骤12: 配置 Accelerator")
print("=" * 60)

from accelerate import Accelerator

# Accelerator 支持多 GPU、分布式训练
# 注意: MPS (Mac) 不稳定支持 fp16，建议禁用混合精度
import torch
if torch.backends.mps.is_available():
    # Mac MPS 设备：禁用混合精度，使用 bf16 或 no
    accelerator = Accelerator(mixed_precision="no")
    print("检测到 MPS 设备，禁用混合精度（fp16 在 MPS 上不稳定）")
else:
    # CUDA 设备：可以使用 fp16
    accelerator = Accelerator(mixed_precision="fp16")
    print("使用 fp16 混合精度")

# 准备模型、优化器和 DataLoader
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)

print(f"设备: {accelerator.device}")
print(f"混合精度设置完成！")


# ========================================
# 步骤13: 配置学习率调度器
# ========================================

print("\n" + "=" * 60)
print("步骤13: 配置学习率调度器")
print("=" * 60)

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

print(f"训练轮数: {num_train_epochs}")
print(f"每轮更新步数: {num_update_steps_per_epoch}")
print(f"总训练步数: {num_training_steps}")
print(f"调度器类型: linear")


# ========================================
# 步骤14: 配置模型保存
# ========================================

print("\n" + "=" * 60)
print("步骤14: 配置模型保存路径")
print("=" * 60)

from huggingface_hub import create_repo, get_full_repo_name

model_name = "bert-finetuned-squad-accelerate"
output_dir = "bert-finetuned-squad-accelerate"

try:
    repo_name = get_full_repo_name(model_name)
    print(f"Hub 仓库: {repo_name}")
    create_repo(repo_name, exist_ok=True)
    print("Hub 仓库已创建/存在！")
except Exception as e:
    print(f"Hub 配置失败（可继续本地训练）: {e}")


# ========================================
# 步骤15: 开始训练循环
# ========================================

print("\n" + "=" * 60)
print("步骤15: 开始训练循环")
print("=" * 60)
print("使用 Accelerate 自定义训练循环")
print("=" * 60)

progress_bar = tqdm(range(num_training_steps), desc="训练进度")

for epoch in range(num_train_epochs):
    print(f"\n=== Epoch {epoch + 1}/{num_train_epochs} ===")

    # ===== 训练阶段 =====
    model.train()
    for step, batch in enumerate(train_dataloader):
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

        # 每100步打印损失
        if step % 100 == 0:
            accelerator.print(f"Step {step}, Loss: {loss.item():.4f}")

    # ===== 评估阶段 =====
    model.eval()
    start_logits = []
    end_logits = []
    accelerator.print("\n开始评估...")

    for batch in tqdm(eval_dataloader, desc="评估进度"):
        with torch.no_grad():
            outputs = model(**batch)

        # 收集 logits（使用 accelerator.gather 处理多 GPU）
        start_logits.append(accelerator.gather(outputs.start_logits).cpu().numpy())
        end_logits.append(accelerator.gather(outputs.end_logits).cpu().numpy())

    # 合并所有 batch 的 logits
    start_logits = np.concatenate(start_logits)
    end_logits = np.concatenate(end_logits)
    # 截断到实际验证集大小（可能因为 batch 对齐而有多余）
    start_logits = start_logits[: len(validation_dataset)]
    end_logits = end_logits[: len(validation_dataset)]

    # 计算评估指标
    metrics = compute_metrics(
        start_logits, end_logits, validation_dataset, raw_datasets["validation"]
    )
    accelerator.print(f"Epoch {epoch + 1} 评估结果: {metrics}")

    # ===== 保存模型 =====
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)

    if accelerator.is_main_process:
        tokenizer.save_pretrained(output_dir)
        accelerator.print(f"模型已保存到 {output_dir}")

print("\n训练完成！")


# ========================================
# 步骤16: 最终评估
# ========================================

print("\n" + "=" * 60)
print("步骤16: 最终评估")
print("=" * 60)

# 执行最后一次评估
model.eval()
start_logits = []
end_logits = []

for batch in tqdm(eval_dataloader, desc="最终评估"):
    with torch.no_grad():
        outputs = model(**batch)
    start_logits.append(accelerator.gather(outputs.start_logits).cpu().numpy())
    end_logits.append(accelerator.gather(outputs.end_logits).cpu().numpy())

start_logits = np.concatenate(start_logits)[: len(validation_dataset)]
end_logits = np.concatenate(end_logits)[: len(validation_dataset)]

final_metrics = compute_metrics(
    start_logits, end_logits, validation_dataset, raw_datasets["validation"]
)
print(f"最终评估结果: {final_metrics}")


# ========================================
# 总结
# ========================================

print("\n" + "=" * 60)
print("脚本执行总结")
print("=" * 60)

print("""
本脚本使用 Accelerate 实现自定义训练循环，特点：

删除的冗余代码：
- Trainer 方式训练（与自定义循环冲突）
- 预训练模型推理测试（非必要）
- 分词器测试代码（非必要）
- pipeline 推理示例（非必要）
- Repository（已废弃，改用 create_repo）

核心流程：
1. 数据预处理：滑动窗口 + offset mapping
2. DataLoader：使用 default_data_collator
3. Accelerate：混合精度 + 多 GPU 支持
4. 训练循环：forward → backward → optimizer step
5. 评估：收集 logits → compute_metrics → SQuAD EM/F1

训练参数：
- batch_size: 8
- learning_rate: 2e-5
- epochs: 3
- scheduler: linear
- mixed_precision: fp16
""")
print("=" * 60)
print("所有步骤完成！")
print("=" * 60)