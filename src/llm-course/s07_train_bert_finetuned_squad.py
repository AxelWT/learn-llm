# ========================================
# SQuAD 问答模型微调脚本
# 使用 BERT 模型在 SQuAD 数据集上进行问答任务训练
# 任务：给定问题和上下文，预测答案的起始和结束位置
# ========================================

print("=" * 60)
print("步骤1: 加载 SQuAD 问答数据集")
print("=" * 60)

from datasets import load_dataset

# SQuAD (Stanford Question Answering Dataset) 是经典的问答数据集
# 每个样本包含：context（上下文段落）、question（问题）、answers（答案）
raw_datasets = load_dataset("squad")

print("数据集结构:")
print(raw_datasets)
print(f"训练集样本数: {len(raw_datasets['train'])}")
print(f"验证集样本数: {len(raw_datasets['validation'])}")
print("数据集加载完成！")

# ========================================
# 步骤2: 查看数据集样本结构
# ========================================

print("\n" + "=" * 60)
print("步骤2: 查看第一个样本的结构")
print("=" * 60)

print("Context: ", raw_datasets["train"][0]["context"])
print("Question: ", raw_datasets["train"][0]["question"])
print("Answer: ", raw_datasets["train"][0]["answers"])

# 解释 answers 字段结构：
# - text: 答案文本列表（可能有多个答案）
# - answer_start: 答案在 context 中的起始位置列表


# ========================================
# 步骤3: 检查数据集中的多答案样本
# ========================================

print("\n" + "=" * 60)
print("步骤3: 检查数据集中的答案数量分布")
print("=" * 60)

# 过滤出答案数量不为1的样本
multi_answer_samples = raw_datasets["train"].filter(lambda x: len(x["answers"]["text"]) != 1)
print(f"答案数量不为1的样本数: {len(multi_answer_samples)}")

# SQuAD 中大部分样本只有一个答案，但有些样本可能有多个答案


# ========================================
# 步骤4: 加载分词器
# ========================================

print("\n" + "=" * 60)
print("步骤4: 加载 BERT 分词器")
print("=" * 60)

from transformers import AutoTokenizer

model_checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

print(f"分词器: {model_checkpoint}")
print(f"词汇表大小: {len(tokenizer)}")
print("分词器加载完成！")

# ========================================
# 步骤5: 测试分词器基本使用
# ========================================

print("\n" + "=" * 60)
print("步骤5: 测试分词器基本使用（问题和上下文）")
print("=" * 60)

context = raw_datasets["train"][0]["context"]
question = raw_datasets["train"][0]["question"]

print(f"问题: {question}")
print(f"上下文片段: {context[:100]}...")

# 分词器会将问题和上下文合并，格式为：[CLS] 问题 [SEP] 上下文 [SEP]
inputs = tokenizer(question, context)
print("\n分词结果（解码）:")
print(tokenizer.decode(inputs["input_ids"]))

# [CLS] 标记表示句子开始，[SEP] 标记分隔问题和上下文


# ========================================
# 步骤6: 测试分词器的长文本处理（滑动窗口）
# ========================================

print("\n" + "=" * 60)
print("步骤6: 测试分词器的滑动窗口处理")
print("=" * 60)

# 当上下文过长时，使用滑动窗口将上下文分成多个片段
# stride（步长）表示相邻窗口之间的重叠区域

inputs = tokenizer(
    question,
    context,
    max_length=100,  # 每个片段的最大长度
    truncation="only_second",  # 只截断第二部分（上下文）
    stride=50,  # 滑动窗口步长（重叠50个token）
    return_overflowing_tokens=True,  # 返回溢出的tokens
)

print(f"原始上下文长度较长，分词后产生 {len(inputs['input_ids'])} 个片段")
print("\n各片段解码结果:")
for i, ids in enumerate(inputs["input_ids"]):
    print(f"\n片段 {i + 1}:")
    print(tokenizer.decode(ids)[:150] + "...")

# 解释：stride 使得相邻片段有重叠，确保答案不会在边界处被截断


# ========================================
# 步骤7: 定义预处理参数
# ========================================

print("\n" + "=" * 60)
print("步骤7: 定义预处理参数")
print("=" * 60)

max_length = 384  # 每个样本的最大长度（问题和上下文合并后）
stride = 128  # 滑动窗口步长

print(f"最大长度: {max_length}")
print(f"滑动窗口步长: {stride}")
print("说明: stride=128 确保相邻片段有足够的重叠，避免答案被截断")

# ========================================
# 步骤8: 定义训练数据预处理函数
# ========================================

print("\n" + "=" * 60)
print("步骤8: 定义训练数据预处理函数")
print("=" * 60)


def preprocess_training_examples(examples):
    """
    训练数据预处理函数：
    1. 分词问题和上下文，使用滑动窗口处理长文本
    2. 计算答案在 token 序列中的起始和结束位置（start_positions, end_positions）

    关键步骤：
    - offset_mapping: 每个 token 对应原始文本中的位置
    - sequence_ids: 区分问题和上下文（0=问题，1=上下文）
    - 将字符级别的答案位置转换为 token 级别
    """
    # 清理问题文本（去除多余空格）
    questions = [q.strip() for q in examples["question"]]

    # 分词
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,  # 返回 token 到字符的映射
        padding="max_length",
    )

    # offset_mapping: 每个 token 对应 (start_char, end_char) 位置
    offset_mapping = inputs.pop("offset_mapping")
    # overflow_to_sample_mapping: 每个 tokenized 片段对应哪个原始样本
    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers = examples["answers"]

    start_positions = []
    end_positions = []

    # 遍历每个分词后的片段
    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]  # 原始样本索引
        answer = answers[sample_idx]
        # 答案在原始文本中的字符位置
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        # sequence_ids: 0=问题部分，1=上下文部分，None=特殊token
        sequence_ids = inputs.sequence_ids(i)

        # ===== 找到上下文在 token 序列中的范围 =====
        idx = 0
        while sequence_ids[idx] != 1:  # 找到上下文开始
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:  # 找到上下文结束
            idx += 1
        context_end = idx - 1

        # ===== 判断答案是否完全在上下文中 =====
        # 如果答案不在当前片段的上下文范围内，标签为 (0, 0)
        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # ===== 将字符位置转换为 token 位置 =====
            # 找到答案起始 token
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            # 找到答案结束 token
            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs


print("训练数据预处理函数定义完成！")

# ========================================
# 步骤9: 执行训练数据预处理
# ========================================

print("\n" + "=" * 60)
print("步骤9: 执行训练数据预处理")
print("=" * 60)

train_dataset = raw_datasets["train"].map(
    preprocess_training_examples,
    batched=True,
    remove_columns=raw_datasets["train"].column_names,
)

print(f"原始训练集样本数: {len(raw_datasets['train'])}")
print(f"预处理后训练集样本数: {len(train_dataset)}")
# 样本数增加是因为长上下文被分成多个片段
print("训练数据预处理完成！")

# ========================================
# 步骤10: 定义验证数据预处理函数
# ========================================

print("\n" + "=" * 60)
print("步骤10: 定义验证数据预处理函数")
print("=" * 60)


def preprocess_validation_examples(examples):
    """
    验证数据预处理函数：
    - 与训练预处理类似，但保留 example_id 和 offset_mapping
    - 用于后处理阶段将 token 位置映射回原始文本位置

    区别：
    - 保留 offset_mapping（用于将答案从 token 映射回字符）
    - 添加 example_id（用于关联预测结果和原始样本）
    - 只保留上下文部分的 offset_mapping（问题部分设为 None）
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
        # 这样在后处理时可以跳过问题部分的 token
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]

    inputs["example_id"] = example_ids
    return inputs


print("验证数据预处理函数定义完成！")

# ========================================
# 步骤11: 执行验证数据预处理
# ========================================

print("\n" + "=" * 60)
print("步骤11: 执行验证数据预处理")
print("=" * 60)

validation_dataset = raw_datasets["validation"].map(
    preprocess_validation_examples,
    batched=True,
    remove_columns=raw_datasets["validation"].column_names,
)

print(f"原始验证集样本数: {len(raw_datasets['validation'])}")
print(f"预处理后验证集样本数: {len(validation_dataset)}")
print("验证数据预处理完成！")

# ========================================
# 步骤12: 加载预训练模型进行推理测试
# ========================================

print("\n" + "=" * 60)
print("步骤12: 使用预训练模型进行推理测试")
print("=" * 60)

# 使用一个已经训练好的问答模型进行测试
small_eval_set = raw_datasets["validation"].select(range(100))
trained_checkpoint = "distilbert-base-cased-distilled-squad"

print(f"使用预训练模型: {trained_checkpoint}")
print(f"测试样本数: {len(small_eval_set)}")

# 重新加载分词器（匹配预训练模型）
tokenizer = AutoTokenizer.from_pretrained(trained_checkpoint)
eval_set = small_eval_set.map(
    preprocess_validation_examples,
    batched=True,
    remove_columns=raw_datasets["validation"].column_names,
)

# 切换回原来的分词器
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

import torch
from transformers import AutoModelForQuestionAnswering

# 准备数据
eval_set_for_model = eval_set.remove_columns(["example_id", "offset_mapping"])
eval_set_for_model.set_format("torch")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"使用设备: {device}")

batch = {k: eval_set_for_model[:len(eval_set_for_model)][k] for k in eval_set_for_model.column_names}
trained_model = AutoModelForQuestionAnswering.from_pretrained(trained_checkpoint).to(device)

# 执行推理
with torch.no_grad():
    outputs = trained_model(**batch)

# 模型输出：start_logits 和 end_logits
# start_logits: 每个 token 作为答案起始位置的概率
# end_logits: 每个 token 作为答案结束位置的概率
start_logits = outputs.start_logits.cpu().numpy()
end_logits = outputs.end_logits.cpu().numpy()

print(f"Start logits shape: {start_logits.shape}")
print(f"End logits shape: {end_logits.shape}")
print("推理完成！")

# ========================================
# 步骤13: 后处理 - 建立样本到特征的映射
# ========================================

print("\n" + "=" * 60)
print("步骤13: 后处理 - 建立样本到特征的映射")
print("=" * 60)

import collections

# 每个原始样本可能对应多个特征（因为滑动窗口）
# 需要建立映射关系，以便将预测结果汇总
example_to_features = collections.defaultdict(list)
for idx, feature in enumerate(eval_set):
    example_to_features[feature["example_id"]].append(idx)

print(f"原始样本数: {len(small_eval_set)}")
print(f"特征数: {len(eval_set)}")
print(f"示例: 第一个样本对应的特征索引: {example_to_features[small_eval_set[0]['id']]}")

# ========================================
# 步骤14: 后处理 - 从 logits 提取答案
# ========================================

print("\n" + "=" * 60)
print("步骤14: 后处理 - 从 logits 提取最佳答案")
print("=" * 60)

import numpy as np

n_best = 20  # 考虑前 n_best 个最佳候选
max_answer_length = 30  # 答案最大长度
predicted_answers = []

for example in small_eval_set:
    example_id = example["id"]
    context = example["context"]
    answers = []

    # 遍历该样本对应的所有特征
    for feature_index in example_to_features[example_id]:
        start_logit = start_logits[feature_index]
        end_logit = end_logits[feature_index]
        offsets = eval_set["offset_mapping"][feature_index]

        # 获取 logits 最高的 n_best 个索引
        start_indexes = np.argsort(start_logit)[-1: -n_best - 1: -1].tolist()
        end_indexes = np.argsort(end_logit)[-1: -n_best - 1: -1].tolist()

        # 遍历所有 (start, end) 组合
        for start_index in start_indexes:
            for end_index in end_indexes:
                # 跳过问题部分的 token（offset 为 None）
                if offsets[start_index] is None or offsets[end_index] is None:
                    continue
                # 跳过无效答案（结束在开始之前，或长度过长）
                if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                ):
                    continue

                # 计算答案文本和得分
                answers.append(
                    {
                        "text": context[offsets[start_index][0]: offsets[end_index][1]],
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    }
                )

    # 选择得分最高的答案
    best_answer = max(answers, key=lambda x: x["logit_score"])
    predicted_answers.append({"id": example_id, "prediction_text": best_answer["text"]})

print(f"预测答案数量: {len(predicted_answers)}")
print("答案提取完成！")

# ========================================
# 步骤15: 计算 SQuAD 评估指标
# ========================================

print("\n" + "=" * 60)
print("步骤15: 计算 SQuAD 评估指标")
print("=" * 60)

import evaluate

metric = evaluate.load("squad")

# 准备真实答案（reference）
theoretical_answers = [
    {"id": ex["id"], "answers": ex["answers"]} for ex in small_eval_set
]

print("\n示例预测结果:")
print(f"预测答案: {predicted_answers[0]}")
print(f"真实答案: {theoretical_answers[0]}")

# SQuAD 评估指标：
# - Exact Match (EM): 答案完全匹配的比例
# - F1: 答案词级别的 F1 分数
results = metric.compute(predictions=predicted_answers, references=theoretical_answers)
print(f"\n评估结果: {results}")
print("评估完成！")

# ========================================
# 步骤16: 定义完整的评估指标计算函数
# ========================================

print("\n" + "=" * 60)
print("步骤16: 定义完整的 compute_metrics 函数")
print("=" * 60)

from tqdm.auto import tqdm


def compute_metrics(start_logits, end_logits, features, examples):
    """
    计算评估指标的完整函数：
    将 logits 转换为答案文本，然后计算 Exact Match 和 F1
    """
    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)

    predicted_answers = []
    for example in tqdm(examples, desc="计算指标"):
        example_id = example["id"]
        context = example["context"]
        answers = []

        # 遍历与该示例相关联的所有特征
        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]

            start_indexes = np.argsort(start_logit)[-1: -n_best - 1: -1].tolist()
            end_indexes = np.argsort(end_logit)[-1: -n_best - 1: -1].tolist()

            for start_index in start_indexes:
                for end_index in end_indexes:
                    # 跳过不完全位于上下文中的答案
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    # 跳过长度小于 0 或大于 max_answer_length 的答案
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


print("compute_metrics 函数定义完成！")

# 使用预训练模型的结果验证函数
print("\n验证 compute_metrics 函数:")
print("\n验证 compute_metrics 函数:")
print(compute_metrics(start_logits, end_logits, eval_set, small_eval_set))

# ========================================
# 步骤17: 初始化待训练模型
# ========================================

print("\n" + "=" * 60)
print("步骤17: 初始化待训练的 BERT 问答模型")
print("=" * 60)

model = AutoModelForQuestionAnswering.from_pretrained(model_checkpoint)
model_size = sum(t.numel() for t in model.parameters())
print(f"模型: {model_checkpoint}")
print(f"模型参数量: {model_size / 1000 ** 2:.1f}M")
print("说明: 模型输出两个 logits: start_positions 和 end_positions")

# ========================================
# 步骤18: 配置训练参数
# ========================================

print("\n" + "=" * 60)
print("步骤18: 配置训练参数")
print("=" * 60)

from transformers import TrainingArguments

args = TrainingArguments(
    "bert-finetuned-squad",
    eval_strategy="no",  # 不在训练中评估（使用自定义评估）
    save_strategy="epoch",  # 每个 epoch 保存
    learning_rate=2e-5,  # 学习率
    num_train_epochs=3,  # 训练轮数
    weight_decay=0.01,  # 权重衰减
    fp16=False,  # 禁用混合精度（MPS 不稳定）
    push_to_hub=True,  # 开启 Hub 推送
)

print("训练参数配置:")
print(f"  - 输出目录: {args.output_dir}")
print(f"  - 学习率: {args.learning_rate}")
print(f"  - 训练轮数: {args.num_train_epochs}")
print(f"  - 混合精度: {args.fp16} (禁用，因为 MPS 不稳定)")

# ========================================
# 步骤19: 创建训练器
# ========================================

print("\n" + "=" * 60)
print("步骤19: 创建 Trainer")
print("=" * 60)

from transformers import Trainer

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    processing_class=tokenizer,
)

print("Trainer 创建完成！")
print(f"  - 训练样本数: {len(train_dataset)}")
print(f"  - 验证样本数: {len(validation_dataset)}")

# ========================================
# 步骤20: 开始训练
# ========================================

print("\n" + "=" * 60)
print("步骤20: 开始训练")
print("=" * 60)
print("注意: 训练可能需要数小时，取决于硬件配置")
print("=" * 60)

trainer.train()
print("\n训练完成！")

# ========================================
# 步骤21: 预测和评估
# ========================================

print("\n" + "=" * 60)
print("步骤21: 使用训练后的模型进行预测和评估")
print("=" * 60)

# trainer.predict 返回 (predictions, label_ids, metrics)
predictions, _, _ = trainer.predict(validation_dataset)
start_logits, end_logits = predictions

# 计算最终评估指标
final_metrics = compute_metrics(
    start_logits, end_logits, validation_dataset, raw_datasets["validation"]
)
print(f"最终评估结果: {final_metrics}")

# ========================================
# 步骤22: 保存模型（本地） + hub
# ========================================

print("\n" + "=" * 60)
print("步骤22: 保存模型到本地 + hub")
print("=" * 60)

# 保存模型到本地目录（不推送到 Hub，避免 SSL 证书问题）
trainer.save_model("bert-finetuned-squad")
tokenizer.save_pretrained("bert-finetuned-squad")
print("模型已保存到本地目录 bert-finetuned-squad！")

trainer.push_to_hub(commit_message="Training complete")
print("模型已推送到 Hub！")

# ========================================
# 总结
# ========================================

print("\n" + "=" * 60)
print("脚本执行总结")
print("=" * 60)

print("""
本脚本微调 BERT 模型用于 SQuAD 问答任务，流程如下:

数据预处理（核心）:
1. 滑动窗口: 处理超长上下文，stride 确保重叠
2. Offset Mapping: token → 字符位置映射
3. 标签生成: 将答案字符位置转换为 token 位置
   - start_positions: 答案起始 token 索引
   - end_positions: 答案结束 token 紑引

模型架构:
- 输入: [CLS] 问题 [SEP] 上下文 [SEP]
- 输出: start_logits + end_logits (每个 token 的概率)

后处理（验证阶段）:
1. 选取 n_best 个最高概率的 start/end 组合
2. 过滤无效答案（长度过长、不在上下文中）
3. 选择得分最高的答案
4. 计算 Exact Match 和 F1 指标

关键参数:
- max_length=384: 每个片段最大长度
- stride=128: 滑动窗口步长（重叠区域）
- n_best=20: 候选答案数量
- max_answer_length=30: 答案最大长度
""")
print("=" * 60)
print("所有步骤完成！")
print("=" * 60)
