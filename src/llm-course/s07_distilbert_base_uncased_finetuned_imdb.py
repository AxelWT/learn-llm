"""
掩码语言模型（Masked Language Model, MLM）微调示例
====================================================

本脚本演示如何使用 Hugging Face Transformers 库对预训练的 DistilBERT 模型
进行掩码语言建模的微调。我们使用 IMDB 电影评论数据集作为训练数据。

掩码语言建模是一种自监督学习方法，通过随机遮蔽输入文本中的某些词，
然后让模型预测这些被遮蔽的词，从而学习语言的统计规律。

主要步骤：
1. 加载预训练的 DistilBERT 模型和分词器
2. 测试模型的掩码预测能力
3. 加载并预处理 IMDB 数据集
4. 实现数据分块和随机遮蔽策略
5. 配置训练参数并进行微调
6. 评估模型并上传到 Hugging Face Hub
"""

# =============================================================================
# 第一部分：加载预训练模型
# =============================================================================
# AutoModelForMaskedLM 是一个通用的模型加载类，专门用于掩码语言建模任务
# 它会根据模型名称自动选择合适的模型架构（如 BERT、DistilBERT、RoBERTa 等）
from transformers import AutoModelForMaskedLM

# 选择用于掩码语言建模的预训练模型
# "distilbert-base-uncased" 是 DistilBERT 的基础版本：
#   - DistilBERT 是 BERT 的轻量级版本，保留了 97% 的性能但参数量减少 40%
#   - "uncased" 表示不分大小写，所有文本会被转换为小写
model_checkpoint = "distilbert-base-uncased"
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
print(f"===========load model {model}===========")

# 比较 DistilBERT 和原始 BERT 的参数量
# num_parameters() 返回模型的总参数数量
distilbert_num_parameters = model.num_parameters() / 1_000_000
print(f"'>>> DistilBERT number of parameters: {round(distilbert_num_parameters)}M'")
print(f"'>>> BERT number of parameters: 110M'")

# =============================================================================
# 第二部分：加载分词器（Tokenizer）
# =============================================================================
# 分词器负责将原始文本转换为模型可以理解的数字序列（token IDs）
# AutoTokenizer 会根据模型名称自动加载与模型匹配的分词器
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
print(f"===========load tokenizer {tokenizer}===========")

# =============================================================================
# 第三部分：测试掩码预测功能
# =============================================================================
# 定义一个测试文本，包含 [MASK] token
# [MASK] 是掩码语言建模中的特殊 token，表示需要被预测的位置
# 模型会根据上下文预测 [MASK] 位置最可能的词
text_mask = "This is a great [MASK]."


def print_mask_prediction(text):
    """
    测试模型的掩码预测能力

    Args:
        text: 包含 [MASK] token 的输入文本

    工作流程：
    1. 使用分词器将文本转换为模型输入格式
    2. 模型预测每个位置的 logits（未归一化的概率分数）
    3. 找到 [MASK] token 的位置
    4. 提取 [MASK] 位置的 logits 并选出概率最高的 5 个候选词
    5. 打印每个候选词替换 [MASK] 后的完整句子
    """
    import torch

    # 将文本转换为模型输入张量
    # return_tensors="pt" 表示返回 PyTorch 张量格式
    inputs = tokenizer(text, return_tensors="pt")

    # 模型前向传播，获取 logits
    # logits 是模型对每个词在词汇表中的预测分数
    token_logits = model(**inputs).logits

    # 找到 [MASK] 的位置并提取其 logits
    # mask_token_id 是分词器中 [MASK] token 的唯一标识符
    mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
    mask_token_logits = token_logits[0, mask_token_index, :]

    # 选择具有最高 logits 的 [MASK] 候选词
    # torch.topk 返回前 k 个最大值及其索引
    top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()

    # 打印每个候选词替换 [MASK] 后的结果
    for token in top_5_tokens:
        # tokenizer.decode 将 token ID 转换回文本
        print(f"'>>> {text.replace(tokenizer.mask_token, tokenizer.decode([token]))}'")


# 执行掩码预测测试
print_mask_prediction(text_mask)
print("===========execute print_mask_prediction===========")

# =============================================================================
# 第四部分：加载 IMDB 数据集
# =============================================================================
# IMDB 数据集是一个常用的情感分析数据集，包含电影评论
# 数据集分为三个部分：
#   - train: 训练集（25,000 条评论）
#   - test: 测试集（25,000 条评论）
#   - unsupervised: 无标签数据（50,000 条评论）
# 对于掩码语言建模，标签信息不重要，我们主要使用文本内容
from datasets import load_dataset

imdb_dataset = load_dataset("imdb")
print(f"===========load datasets {imdb_dataset}===========")


def print_imdb_sample(field):
    """
    打印 IMDB 数据集的样本示例

    Args:
        field: 数据集的字段名称，可以是 "train"、"test" 或 "unsupervised"

    工作流程：
    1. 随机打乱数据集（使用固定种子确保可重复性）
    2. 选择前 3 个样本
    3. 打印每条评论的文本内容和标签
    """
    # shuffle(seed=42) 随机打乱数据，seed 参数确保每次运行结果一致
    # select(range(3)) 选择前 3 个样本
    sample = imdb_dataset[field].shuffle(seed=42).select(range(3))

    for row in sample:
        print(f"\n'>>> Review: {row['text']}'")
        print(f"'>>> Label: {row['label']}'")


# 打印训练集和无标签数据集的样本
print_imdb_sample("train")
print_imdb_sample("unsupervised")
print("===========execute print_imdb_sample===========")


# =============================================================================
# 第五部分：数据预处理 - 分词（Tokenization）
# =============================================================================
# 在训练之前，需要将所有文本转换为 token IDs
# 这一步是所有 NLP 模型训练的基础


def tokenize_function(examples):
    """
    对文本进行分词处理

    Args:
        examples: 包含 "text" 字段的样本批次

    Returns:
        result: 包含分词结果的字典，包括：
            - input_ids: token ID 序列
            - attention_mask: 注意力掩码（标记哪些位置是真实的 token）
            - word_ids: 单词级别的 ID 映射（用于全词遮蔽）

    注意：
    word_ids 是快速分词器的特性，可以将每个 token 映射到原始文本中的单词
    这对于实现全词遮蔽（Whole Word Masking）策略非常重要
    """
    result = tokenizer(examples["text"])
    if tokenizer.is_fast:
        # 快速分词器支持 word_ids 功能
        # word_ids 返回每个 token 对应的原始单词索引
        # 特殊 token（如 [CLS]、[SEP]）的 word_id 为 None
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result


# 使用 batched=True 来激活快速多线程处理
# 这可以显著提高大数据集的处理速度
# remove_columns=["text", "label"] 移除原始文本和标签列
# 因为我们只需要分词后的结果进行语言模型训练
tokenized_datasets = imdb_dataset.map(
    tokenize_function, batched=True, remove_columns=["text", "label"]
)
print(f"===========load tokenized_datasets {tokenized_datasets}===========")


# =============================================================================
# 第六部分：数据分块（Chunking）演示
# =============================================================================
# IMDB 评论的长度不一，但模型需要固定长度的输入
# 我们需要将所有文本拼接后按固定长度分块


def print_tokenized_datasets_sample():
    """
    演示数据分块的过程

    分块策略：
    1. 将多条评论的 token 序列拼接成一条长序列
    2. 按 chunk_size（如 128）将长序列分割成多个块
    3. 每个块作为独立的训练样本

    这种策略的好处：
    - 避免短文本被过度填充（padding），提高训练效率
    - 更好地利用 GPU 内存（固定长度的输入更高效）
    """
    # 切片会为每个特征生成一个列表的列表
    tokenized_samples = tokenized_datasets["train"][:3]

    # 打印每条评论的长度（token 数量）
    for idx, sample in enumerate(tokenized_samples["input_ids"]):
        print(f"'>>> Review {idx} length: {len(sample)}'")

    # 拼接所有样本的特征
    # sum(tokenized_samples[k], []) 将列表的列表合并成一个列表
    concatenated_examples = {
        k: sum(tokenized_samples[k], []) for k in tokenized_samples.keys()
    }
    total_length = len(concatenated_examples["input_ids"])
    print(f"'>>> Concatenated reviews length: {total_length}'")

    # 定义块大小（chunk_size）
    # 这是模型输入的最大长度
    chunk_size = 128

    # 按块大小分割拼接后的序列
    # 使用列表切片 [i: i + chunk_size] 从位置 i 开始截取 chunk_size 个 token
    chunks = {
        k: [t[i: i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }

    # 打印每个块的长度（最后一个块可能小于 chunk_size）
    for chunk in chunks["input_ids"]:
        print(f"'>>> Chunk length: {len(chunk)}'")


print_tokenized_datasets_sample()
print("===========execute print_tokenized_datasets_sample===========")


# =============================================================================
# 第七部分：正式数据分块处理
# =============================================================================
# 使用 map 函数对所有数据应用分块处理


def group_texts(examples):
    """
    将文本拼接并分块，创建训练样本

    Args:
        examples: 批次的分词样本

    Returns:
        result: 分块后的训练样本，包含：
            - input_ids: token ID 序列（长度固定为 chunk_size）
            - attention_mask: 注意力掩码
            - word_ids: 单词 ID 映射
            - labels: 标签序列（与 input_ids 相同，用于 MLM 训练）

    处理步骤：
    1. 拼接所有文本的 token 序列
    2. 计算总长度，丢弃不完整的最后一块
    3. 按固定长度分块
    4. 创建 labels 列（在 MLM 中，labels = input_ids）
    """
    chunk_size = 128

    # 拼接所有的文本
    # 将批次中所有样本的特征合并成一个长序列
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}

    # 计算拼接文本的长度
    total_length = len(concatenated_examples[list(examples.keys())[0]])

    # 如果最后一个块小于 chunk_size，我们将其丢弃
    # 这是为了确保所有训练样本长度一致，避免 padding
    # (total_length // chunk_size) * chunk_size 计算完整的块数对应的长度
    total_length = (total_length // chunk_size) * chunk_size

    # 按最大长度分块
    result = {
        k: [t[i: i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }

    # 创建一个新的 labels 列
    # 在掩码语言建模中，labels 就是原始的 input_ids
    # 训练时会随机遮蔽某些位置的 input_ids，让模型预测这些位置的 labels
    result["labels"] = result["input_ids"].copy()
    return result


# 使用 map 函数批量处理所有数据
lm_datasets = tokenized_datasets.map(group_texts, batched=True)
print(f"===========load lm_datasets {lm_datasets}===========")

# =============================================================================
# 第八部分：数据整理器（Data Collator）- 随机遮蔽
# =============================================================================
# DataCollatorForLanguageModeling 是专门用于 MLM 的数据整理器
# 它会在训练时随机遮蔽输入中的某些 token


# DataCollatorForLanguageModeling 负责在训练时动态遮蔽 token
from transformers import DataCollatorForLanguageModeling

# mlm_probability=0.15 表示随机遮蔽 15% 的 token
# 这是 BERT 论文中推荐的比例，在性能和难度之间取得了平衡
# 注意：遮蔽是在训练时动态进行的，每个 epoch 的遮蔽位置都不同
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
print(f"===========load data_collator {data_collator}===========")


def print_random_mask(collator):
    """
    演示随机遮蔽的效果

    Args:
        collator: 数据整理器

    工作流程：
    1. 取出几个训练样本
    2. 移除 word_ids（标准 MLM 不需要）
    3. 使用数据整理器进行随机遮蔽
    4. 打印遮蔽后的文本（[MASK] token 会替换被遮蔽的词）
    """
    samples = [lm_datasets["train"][i] for i in range(2)]
    for sample in samples:
        # 移除 word_ids，因为标准 MLM 遮蔽策略不使用它
        _ = sample.pop("word_ids")

    # 数据整理器会：
    # 1. 将样本打包成批次
    # 2. 随机选择 15% 的 token 进行遮蔽
    # 3. 将被遮蔽位置的 labels 设置为原始 token ID，其余位置设为 -100
    for chunk in data_collator(samples)["input_ids"]:
        print(f"\n'>>> {tokenizer.decode(chunk)}'")


print_random_mask(data_collator)
print("===========execute print_random_mask===========")

# =============================================================================
# 第九部分：全词遮蔽（Whole Word Masking）
# =============================================================================
# 标准的 MLM 随机遮蔽 token，可能只遮蔽单词的一部分
# 全词遮蔽策略确保整个单词一起被遮蔽，提高了任务的难度和实用性


import collections
import numpy as np

from transformers import default_data_collator

# 全词遮蔽概率设置为 20%
# 比标准 MLM 略高，因为遮蔽的是整个单词而非单个 token
wwm_probability = 0.2


def whole_word_masking_data_collator(features):
    """
    实现全词遮蔽（Whole Word Masking）策略

    Args:
        features: 批次的训练样本

    Returns:
        经过全词遮蔽处理后的批次数据

    全词遮蔽的特点：
    - 不会只遮蔽单词的一部分（如 "great" 只遮蔽 "gre"）
    - 整个单词的所有 token 都会被一起遮蔽
    - 这使预测任务更具挑战性，模型需要理解完整的语义

    实现步骤：
    1. 使用 word_ids 建立 token 到单词的映射
    2. 随机选择要遮蔽的单词（而非 token）
    3. 将选中单词的所有 token 都替换为 [MASK]
    """
    for feature in features:
        # 获取 word_ids，这是 token 到原始单词的映射
        word_ids = feature.pop("word_ids")

        # 创建一个单词与对应 token 索引之间的映射
        # mapping[word_index] = [token_index_1, token_index_2, ...]
        # 这样可以知道每个单词包含哪些 token
        mapping = collections.defaultdict(list)
        current_word_index = -1
        current_word = None
        for idx, word_id in enumerate(word_ids):
            if word_id is not None:
                if word_id != current_word:
                    # 遇到新的单词，更新索引
                    current_word = word_id
                    current_word_index += 1
                # 将 token 索引添加到对应单词的列表中
                mapping[current_word_index].append(idx)

        # 随机遮蔽单词
        # np.random.binomial 生成二元分布随机数
        # 对于每个单词，有 wwm_probability 的概率被选中遮蔽
        mask = np.random.binomial(1, wwm_probability, (len(mapping),))
        input_ids = feature["input_ids"]
        labels = feature["labels"]

        # 初始化新的 labels 列表
        # -100 是 PyTorch 交叉熵损失的忽略索引
        # 被 -100 标记的位置不参与损失计算
        new_labels = [-100] * len(labels)

        for word_id in np.where(mask)[0]:
            # 对于每个被选中遮蔽的单词
            word_id = word_id.item()
            for idx in mapping[word_id]:
                # 将该单词的所有 token 都遮蔽
                # labels 记录原始的 token ID（用于计算损失）
                # input_ids 被替换为 [MASK] token
                new_labels[idx] = labels[idx]
                input_ids[idx] = tokenizer.mask_token_id

        feature["labels"] = new_labels

    # 使用默认的数据整理器将处理后的样本打包成批次
    return default_data_collator(features)


def print_whole_word_masking():
    """
    演示全词遮蔽的效果

    与标准 MLM 遮蔽的对比：
    - 标准 MLM: "This is a gre[MASK]t movie"（只遮蔽了 token）
    - 全词遮蔽: "This is a [MASK][MASK][MASK] movie"（遮蔽了整个单词）
    """
    samples = [lm_datasets["train"][i] for i in range(2)]
    batch = whole_word_masking_data_collator(samples)

    for chunk in batch["input_ids"]:
        print(f"\n'>>> {tokenizer.decode(chunk)}'")


print_whole_word_masking()
print("===========execute print_whole_word_masking===========")

# =============================================================================
# 第十部分：数据集采样与划分
# =============================================================================
# 为了演示目的，我们使用较小的数据集
# 实际训练可以使用完整数据集以提高效果


# 训练集大小设置为 20,000 条（远小于原始的 25,000 条）
train_size = 20_000
# 测试集大小为训练集的 10%
test_size = int(0.1 * train_size)

# 使用 train_test_split 划分训练集和测试集
# seed=42 确保划分结果可重复
downsampled_dataset = lm_datasets["train"].train_test_split(
    train_size=train_size, test_size=test_size, seed=42
)
print(f"===========load downsampled_dataset {downsampled_dataset}===========")

# =============================================================================
# 第十一部分：配置训练参数
# =============================================================================
# TrainingArguments 包含所有训练相关的配置


from transformers import TrainingArguments

batch_size = 64
# 在每个 epoch 输出训练的 loss
# logging_steps 计算每个 epoch 内的日志记录次数
logging_steps = len(downsampled_dataset["train"]) // batch_size
# 从模型名称中提取最后部分作为输出目录名
model_name = model_checkpoint.split("/")[-1]

training_args = TrainingArguments(
    # 输出目录，用于保存模型和训练日志
    output_dir=f"{model_name}-finetuned-imdb",
    # 每个 epoch 结束时进行评估（新版参数名）
    eval_strategy="epoch",
    # 学习率设置为 2e-5，这是微调常用的较小学习率
    # 较小的学习率可以避免破坏预训练模型的知识
    learning_rate=2e-5,
    # 权重衰减（weight decay）用于防止过拟合
    # 类似于 L2 正则化，限制参数的幅度
    weight_decay=0.01,
    # 每个设备的训练批次大小
    per_device_train_batch_size=batch_size,
    # 每个设备的评估批次大小
    per_device_eval_batch_size=batch_size,
    # 训练完成后将模型上传到 Hugging Face Hub
    # 需要预先登录：huggingface-cli login
    push_to_hub=True,
    # 使用 bf16 混合精度训练（比 fp16 更稳定，推荐在现代 GPU 上使用）
    # 如果 GPU 不支持 bf16，可以改用 fp16=True，或禁用混合精度（删除此参数）
    bf16=True,
    # 日志记录频率
    logging_steps=logging_steps,
)

# =============================================================================
# 第十二部分：创建训练器并开始训练
# =============================================================================
# Trainer 封装了训练循环，简化了模型训练的代码


from transformers import Trainer

trainer = Trainer(
    # 要训练的模型
    model=model,
    # 训练参数配置
    args=training_args,
    # 训练数据集
    train_dataset=downsampled_dataset["train"],
    # 评估数据集
    eval_dataset=downsampled_dataset["test"],
    # 数据整理器，用于动态遮蔽 token
    data_collator=data_collator,
    # 分词器，用于处理评估时的文本
    processing_class=tokenizer,
)

# =============================================================================
# 第十三部分：评估与训练
# =============================================================================
# 困惑度（Perplexity）是语言模型的常用评估指标


import math

# 训练前的评估，计算初始困惑度
# 困惑度衡量模型对文本的预测能力，值越低越好
# 困惑度 = exp(loss)，可以理解为模型平均对每个词的不确定性
eval_results = trainer.evaluate()
print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

# 开始训练
# trainer.train() 会执行完整的训练过程：
# 1. 前向传播计算损失
# 2. 反向传播计算梯度
# 3. 更新模型参数
# 4. 定期评估和保存模型
trainer.train()

# 训练后的评估，计算最终困惑度
# 与初始困惑度对比可以看到模型训练的效果
eval_results = trainer.evaluate()
print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

# =============================================================================
# 第十四部分：上传模型到 Hub
# =============================================================================
# 将训练好的模型上传到 Hugging Face Model Hub
# 这使得其他人可以下载和使用你的模型


# push_to_hub() 将模型、分词器、训练配置等上传到 Hub
# 上传后，其他人可以通过：
#   model = AutoModelForMaskedLM.from_pretrained("your-username/distilbert-finetuned-imdb")
# 来加载你的模型
trainer.push_to_hub()
