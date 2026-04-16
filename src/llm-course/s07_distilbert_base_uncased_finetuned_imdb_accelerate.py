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

输出：
- 训练过程中的困惑度（Perplexity）指标
- 微调后的模型保存在 output_dir 目录
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
print(f"======load model {model}")

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
print(f"======load tokenizer {tokenizer}")

# =============================================================================
# 第三部分：测试掩码预测功能
# =============================================================================
# 定义一个测试文本，包含 [MASK] token
# [MASK] 是掩码语言建模中的特殊 token，表示需要被预测的位置
# 模型会根据上下文预测 [MASK] 位置最可能的词
text_mask = "This is a great [MASK]."


def print_mask_prediction(text: str) -> None:
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

    Returns:
        None（直接打印结果）

    示例输出：
        '>>> This is a great movie.'
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
print("======execute print_mask_prediction")

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
print(f"======load datasets {imdb_dataset}")


def print_imdb_sample(field: str) -> None:
    """
    打印 IMDB 数据集的样本示例

    Args:
        field: 数据集的字段名称，可以是 "train"、"test" 或 "unsupervised"

    工作流程：
    1. 随机打乱数据集（使用固定种子确保可重复性）
    2. 选择前 3 个样本
    3. 打印每条评论的文本内容和标签

    Returns:
        None（直接打印结果）

    注意：
    - seed=42 是机器学习中常用的随机种子，确保实验可重复
    - label=1 表示正面评论，label=0 表示负面评论
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
print("======execute print_imdb_sample")


# =============================================================================
# 第五部分：数据预处理 - 分词（Tokenization）
# =============================================================================
# 在训练之前，需要将所有文本转换为 token IDs
# 这一步是所有 NLP 模型训练的基础


def tokenize_function(examples: dict) -> dict:
    """
    对文本进行分词处理

    Args:
        examples: 包含 "text" 字段的样本批次，格式如：
            {"text": ["review1", "review2", ...]}

    Returns:
        result: 包含分词结果的字典，包括：
            - input_ids: List[List[int]] - token ID 序列，每个 token 对应词汇表中的索引
            - attention_mask: List[List[int]] - 注意力掩码，1 表示真实 token，0 表示 padding
            - word_ids: List[List[Optional[int]]] - 单词级别的 ID 映射（用于全词遮蔽）

    技术细节：
    - 快速分词器（Fast Tokenizer）基于 Rust 实现，比普通分词器快 10-20 倍
    - word_ids() 方法返回每个 token 对应的原始单词索引
    - 特殊 token（[CLS]、[SEP]、[PAD]）的 word_id 返回 None
    - 这对于实现全词遮蔽（Whole Word Masking）策略至关重要

    示例：
        原文: "Hello world"
        tokens: ["[CLS]", "hello", "world", "[SEP]"]
        word_ids: [None, 0, 1, None]  # hello 属于第 0 个单词，world 属于第 1 个
    """
    result = tokenizer(examples["text"])
    if tokenizer.is_fast:
        # 快速分词器支持 word_ids 功能
        # transformers 不同版本中，word_ids 可能是方法或属性
        # 某些版本需要调用 word_ids()，某些版本直接访问 word_ids 属性
        # 特殊 token（如 [CLS]、[SEP]）的 word_id 为 None
        result["word_ids"] = [enc.word_ids for enc in result.encodings]
    return result


# 使用 batched=True 来激活快速多线程处理
# 这可以显著提高大数据集的处理速度
# remove_columns=["text", "label"] 移除原始文本和标签列
# 因为我们只需要分词后的结果进行语言模型训练
tokenized_datasets = imdb_dataset.map(
    tokenize_function, batched=True, remove_columns=["text", "label"]
)
print(f"======load tokenized_datasets {tokenized_datasets}")


# =============================================================================
# 第六部分：数据分块（Chunking）演示
# =============================================================================
# IMDB 评论的长度不一，但模型需要固定长度的输入
# 我们需要将所有文本拼接后按固定长度分块


def print_tokenized_datasets_sample() -> None:
    """
    演示数据分块的过程

    分块策略：
    1. 将多条评论的 token 序列拼接成一条长序列
    2. 按 chunk_size（如 128）将长序列分割成多个块
    3. 每个块作为独立的训练样本

    这种策略的好处：
    - 避免短文本被过度填充（padding），提高训练效率
    - 更好地利用 GPU 内存（固定长度的输入更高效）
    - 节省计算资源：padding token 不参与有效训练

    为什么选择 128 作为 chunk_size：
    - DistilBERT 最大支持 512 tokens，但 128 在性能和效率间取得平衡
    - 较短的序列减少内存占用，允许更大的 batch_size
    - IMDB 评论平均长度约 200-300 tokens，128 能捕捉足够语义

    Returns:
        None（直接打印分块演示结果）
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
print("======execute print_tokenized_datasets_sample")


# =============================================================================
# 第七部分：正式数据分块处理
# =============================================================================
# 使用 map 函数对所有数据应用分块处理


def group_texts(examples: dict) -> dict:
    """
    将文本拼接并分块，创建训练样本

    Args:
        examples: 批次的分词样本，包含多个特征列：
            - input_ids: List[List[int]]
            - attention_mask: List[List[int]]
            - word_ids: List[List[Optional[int]]]

    Returns:
        result: 分块后的训练样本，包含：
            - input_ids: List[List[int]] - token ID 序列（长度固定为 128）
            - attention_mask: List[List[int]] - 注意力掩码（全为 1）
            - word_ids: List[List[Optional[int]]] - 单词 ID 映射
            - labels: List[List[int]] - 标签序列（与 input_ids 相同，用于 MLM 训练）

    处理步骤：
    1. 拼接所有文本的 token 序列：[sample1, sample2, ...] -> [all_tokens]
    2. 计算总长度，丢弃不完整的最后一块（确保长度一致性）
    3. 按固定长度 128 分块：[0:128], [128:256], [256:384], ...
    4. 创建 labels 列（在 MLM 中，labels = input_ids）

    为什么丢弃最后一块：
    - 最后一块长度可能 < 128，需要 padding
    - Padding 会引入无效计算，降低效率
    - 数据量足够大时，丢弃少量数据影响很小

    为什么 labels = input_ids：
    - MLM 是自监督学习，没有外部标签
    - 训练时随机遮蔽 input_ids 中的部分 token
    - 模型需要预测被遮蔽位置的原 token，labels 记录原值
    - 未被遮蔽位置的 labels 设为 -100（PyTorch 忽略索引）
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
print(f"======load lm_datasets {lm_datasets}")

# 验证数据集是否包含 word_ids 字段（用于全词遮蔽）
print(f">>> 数据集字段: {lm_datasets['train'].column_names}")
if "word_ids" in lm_datasets["train"].column_names:
    print(f">>> word_ids 字段已正确保留，可以使用全词遮蔽策略")
else:
    print(f">>> 警告: word_ids 字段缺失，将回退到标准 MLM 遮蔽策略")

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
print(f"======load data_collator {data_collator}")


def print_random_mask(collator: DataCollatorForLanguageModeling) -> None:
    """
    演示随机遮蔽的效果

    Args:
        collator: DataCollatorForLanguageModeling 数据整理器实例

    工作流程：
    1. 取出几个训练样本
    2. 移除 word_ids（标准 MLM 不需要）
    3. 使用数据整理器进行随机遮蔽
    4. 打印遮蔽后的文本（[MASK] token 会替换被遮蔽的词）

    Returns:
        None（直接打印遮蔽结果）

    遮蔽策略详解（BERT 论文方法）：
    - 15% 的 token 被选中进行遮蔽处理
    - 其中 80% 替换为 [MASK] token
    - 10% 替换为随机 token（防止模型只学习 [MASK] 位置）
    - 10% 保持原样（让模型学习正确表示）

    示例：
        原文: "This is a great movie"
        遮蔽后: "This is a [MASK] movie"（80% 情况）
                "This is a wonderful movie"（10% 随机替换）
                "This is a great movie"（10% 保持原样）
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
print("======execute print_random_mask")

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


def whole_word_masking_data_collator(features: list[dict]) -> dict:
    """
    实现全词遮蔽（Whole Word Masking）策略

    Args:
        features: 批次的训练样本，每个样本是包含以下键的字典：
            - "input_ids": List[int] - token ID 序列
            - "labels": List[int] - 原始 token ID（训练前与 input_ids 相同）
            - "word_ids": List[Optional[int]] - 单词 ID 映射（可选）

    Returns:
        dict: 经过全词遮蔽处理后的批次数据，包含：
            - "input_ids": torch.Tensor - 遮蔽后的 token ID
            - "attention_mask": torch.Tensor - 注意力掩码
            - "labels": torch.Tensor - 标签（被遮蔽位置为原 token ID，其他为 -100）

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
        input_ids = feature["input_ids"]
        labels = feature["labels"]

        # 检查是否有 word_ids 字段
        if "word_ids" in feature:
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
        else:
            # word_ids 缺失时，使用标准 MLM 随机遮蔽策略
            # 每个 token 有 wwm_probability 的概率被遮蔽
            mask = np.random.binomial(1, wwm_probability, (len(input_ids),))
            mapping = {i: [i] for i in range(len(input_ids))}

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


def print_whole_word_masking() -> None:
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
print("======execute print_whole_word_masking")

# =============================================================================
# 第十部分：数据集采样与划分
# =============================================================================
# 为了演示目的，我们使用较小的数据集
# 实际训练可以使用完整数据集以提高效果

# 为什么需要数据集采样：
# - IMDB 数据集有 25,000 条训练数据，处理时间长
# - 演示目的下，减少数据量可快速验证流程
# - 实际生产训练应使用完整数据集以获得最佳效果

# 训练集大小设置为 20,000 条
# - 比 IMDB 原始训练集（25,000）略少
# - 使用下采样加快训练速度，同时保持足够数据量
train_size = 20_000

# 测试集大小为训练集的 10%（2,000 条）
# - 评估集不需要太大，足以反映模型性能即可
# - 常用比例：训练集的 10%-20%
test_size = int(0.1 * train_size)

# 使用 train_test_split 划分训练集和测试集
# seed=42 确保划分结果可重复
# - 相同种子每次运行产生相同划分
# - 便于实验对比和调试
downsampled_dataset = lm_datasets["train"].train_test_split(
    train_size=train_size, test_size=test_size, seed=42
)
print(f"======load downsampled_dataset {downsampled_dataset}")


def insert_random_mask(batch: dict) -> dict:
    """
    为评估数据集预先应用随机遮蔽

    Args:
        batch: 批次数据，包含 input_ids, attention_mask, labels 等字段

    Returns:
        dict: 包含遮蔽后数据的字典，字段名添加 "masked_" 前缀

    为什么需要预先遮蔽：
    - 训练时：DataCollator 动态遮蔽，每个 epoch 遮蔽位置不同
    - 评估时：需要固定遮蔽位置，确保每次评估结果一致
    - 预先遮蔽确保评估指标的可重复性和可比性

    实现细节：
    1. 将批次转换为样本列表（zip(*batch.values()) 解包批次）
    2. 使用 data_collator 应用遮蔽
    3. 将结果转换为 numpy 格式并存入新字段
    """
    features = [dict(zip(batch, t)) for t in zip(*batch.values())]
    masked_inputs = data_collator(features)
    # 为数据集中的每一列创建一个新的"masked"列
    return {"masked_" + k: v.numpy() for k, v in masked_inputs.items()}


# =============================================================================
# 评估数据集预处理
# =============================================================================
# 为评估数据集预先应用遮蔽，确保评估结果的一致性

# 移除 word_ids 列（评估时不需要全词遮蔽）
# - 评估使用标准 MLM 遮蔽策略
# - word_ids 仅用于全词遮蔽训练
downsampled_dataset = downsampled_dataset.remove_columns(["word_ids"])

# 为测试集预先应用随机遮蔽
# 使用 map 函数批量处理，insert_random_mask 定义遮蔽逻辑
# remove_columns: 移除原始列，只保留遮蔽后的列
eval_dataset = downsampled_dataset["test"].map(
    insert_random_mask,
    batched=True,
    remove_columns=downsampled_dataset["test"].column_names,
)

# 重命名遮蔽后的列，恢复标准列名
# 将 "masked_input_ids" 等前缀去掉，便于后续使用
eval_dataset = eval_dataset.rename_columns(
    {
        "masked_input_ids": "input_ids",
        "masked_attention_mask": "attention_mask",
        "masked_labels": "labels",
    }
)

# =============================================================================
# 第十一部分：模型训练配置
# =============================================================================
# 使用 PyTorch 和 Hugging Face Accelerate 进行模型训练

from torch.utils.data import DataLoader
from transformers import default_data_collator

# 批次大小设置为 64
# 选择依据：
# - 较大的 batch_size 提高训练稳定性，但需要更多 GPU 内存
# - DistilBERT 在 batch_size=64 时通常能充分利用 GPU
# - 如果内存不足，可减小到 32 或 16
batch_size = 64

# 训练数据加载器
# shuffle=True: 每个 epoch 打乱数据顺序，防止模型记忆顺序模式
# collate_fn=data_collator: 使用动态遮蔽的数据整理器
train_dataloader = DataLoader(
    downsampled_dataset["train"],
    shuffle=True,
    batch_size=batch_size,
    collate_fn=data_collator,
)

# 评估数据加载器
# shuffle=False: 评估时保持固定顺序，确保结果可重复
# collate_fn=default_data_collator: 评估数据已预先遮蔽，只需打包成批次
eval_dataloader = DataLoader(
    eval_dataset, batch_size=batch_size, collate_fn=default_data_collator
)

# 重新加载模型（避免使用之前测试时的模型状态）
# 这是良好实践：确保训练从头开始，不受之前操作的影响
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)

# =============================================================================
# 第十二部分：优化器和学习率调度器
# =============================================================================

from torch.optim import AdamW

# AdamW 优化器（Adam with Weight Decay）
# - Adam: 自适应学习率优化器，适合深度学习
# - W (Weight Decay): 添加权重衰减（L2 正则化），防止过拟合
# - lr=5e-5: 学习率，是 NLP 微调的常用值
#   - 太大：训练不稳定，可能发散
#   - 太小：收敛太慢，可能陷入局部最优
optimizer = AdamW(model.parameters(), lr=5e-5)

# Accelerate 库：简化分布式训练和多设备训练
# 自动处理：
# - GPU/TPU/CPU 设备选择
# - 多卡分布式训练的同步
# - 混合精度训练（FP16/BF16）
from accelerate import Accelerator

accelerator = Accelerator()

# prepare() 方法：
# - 将模型、优化器、数据加载器包装为分布式版本
# - 自动将数据移动到正确的设备（GPU）
# - 处理梯度同步（多卡训练时）
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)

from transformers import get_scheduler

# 训练超参数
num_train_epochs = 3  # 训练轮数：3 轮通常足够微调
num_update_steps_per_epoch = len(train_dataloader)  # 每个 epoch 的更新步数
num_training_steps = num_train_epochs * num_update_steps_per_epoch  # 总训练步数

# 学习率调度器：线性衰减
# - 训练开始时学习率较高，快速收敛
# - 训练过程中逐渐降低学习率，精细化调整
# - num_warmup_steps=0: 无预热阶段（可设置为总步数的 10% 进行预热）
# - 最终学习率衰减到接近 0
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

# 模型输出目录
output_dir = "distilbert-base-uncased-finetuned-imdb-accelerate"

# =============================================================================
# 第十三部分：训练循环
# =============================================================================
# 标准的 PyTorch 训练流程：训练 -> 评估 -> 保存

from tqdm.auto import tqdm
import torch
import math

# 进度条：可视化训练进度
progress_bar = tqdm(range(num_training_steps))

# 主训练循环
for epoch in range(num_train_epochs):
    # ========== 训练阶段 ==========
    # model.train(): 设置模型为训练模式
    # - 启用 Dropout（防止过拟合）
    # - 启用 BatchNorm 的统计更新
    model.train()
    for batch in train_dataloader:
        # 前向传播：计算模型输出和损失
        # outputs.loss 是 MLM 任务的交叉熵损失
        outputs = model(**batch)
        loss = outputs.loss

        # 反向传播：计算梯度
        # accelerator.backward() 处理分布式训练的梯度同步
        accelerator.backward(loss)

        # 参数更新：应用梯度
        optimizer.step()

        # 学习率更新：按调度器调整学习率
        lr_scheduler.step()

        # 清零梯度：防止梯度累积
        # PyTorch 默认会累积梯度，需要手动清零
        optimizer.zero_grad()

        # 更新进度条
        progress_bar.update(1)

    # ========== 评估阶段 ==========
    # model.eval(): 设置模型为评估模式
    # - 禁用 Dropout（确定性输出）
    # - 禁用 BatchNorm 的统计更新
    model.eval()
    losses = []

    for step, batch in enumerate(eval_dataloader):
        # torch.no_grad(): 禁用梯度计算
        # - 评估不需要梯度，节省内存和计算
        with torch.no_grad():
            outputs = model(**batch)

        # 收集每个批次的损失
        loss = outputs.loss

        # accelerator.gather(): 收集所有设备上的损失
        # 多卡训练时，每个 GPU 计算部分数据，需要汇总
        # loss.repeat(batch_size): 扩展损失以便正确收集
        losses.append(accelerator.gather(loss.repeat(batch_size)))

    # 合并所有损失
    losses = torch.cat(losses)

    # 截取到实际评估数据集大小
    # 最后一个批次可能不完整，需要截取
    losses = losses[: len(eval_dataset)]

    # 计算困惑度（Perplexity）
    # 困惑度 = exp(平均交叉熵损失)
    # - 衡量模型对文本的"惊讶程度"
    # - 越低越好（模型更准确预测）
    # - 完美模型困惑度 = 1（总是正确预测）
    # - 随机模型困惑度 = 词汇表大小（约 30,000）
    try:
        perplexity = math.exp(torch.mean(losses))
    except OverflowError:
        # 损失过大时，困惑度溢出为无穷大
        perplexity = float("inf")

    print(f">>> Epoch {epoch}: Perplexity: {perplexity}")

    # ========== 保存模型 ==========
    # 等待所有进程完成当前 epoch
    # 确保分布式训练时所有设备同步
    accelerator.wait_for_everyone()

    # 解包模型：获取原始模型（去除分布式包装）
    unwrapped_model = accelerator.unwrap_model(model)

    # 保存模型权重和配置
    # save_function=accelerator.save: 使用 Accelerate 的保存方法
    unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)

    # 主进程保存分词器
    # 分布式训练时，只需主进程保存，避免重复
    if accelerator.is_main_process:
        tokenizer.save_pretrained(output_dir)

# =============================================================================
# 第十四部分：模型上传到 Hugging Face Hub
# =============================================================================
print("\n======开始保存模型到本地和上传到 Hub======")

# 1. 确保模型已保存到本地目录
# save_pretrained() 会保存模型权重、配置文件和分词器
print(f">>> 模型已保存到本地目录: {output_dir}")

# 2. 上传到 Hugging Face Hub
# 使用 push_to_hub() 方法将模型上传到 Hub
# 需要先登录 Hugging Face：在终端运行 `huggingface-cli login` 或设置 HF_TOKEN 环境变量
from huggingface_hub import HfApi

# 定义你的 Hub 模型 ID（格式：username/model-name）
# 请替换为你的实际用户名
hf_username = "your-username"  # 替换为你的 Hugging Face 用户名
hub_model_id = f"{hf_username}/distilbert-base-uncased-finetuned-imdb-accelerate"

try:
    # 创建 HfApi 实例
    api = HfApi()

    # 检查是否已登录（通过环境变量 HF_TOKEN 或 huggingface-cli login）
    # push_to_hub 会自动使用已配置的认证信息
    unwrapped_model.push_to_hub(
        repo_id=hub_model_id,
        commit_message="Upload fine-tuned DistilBERT for MLM on IMDB",
        private=False,  # 设置为 True 可创建私有仓库
    )

    # 同时上传分词器（分词器需要与模型一起使用）
    tokenizer.push_to_hub(
        repo_id=hub_model_id,
        commit_message="Upload tokenizer for fine-tuned model",
    )

    print(f">>> 模型已成功上传到 Hugging Face Hub!")
    print(f">>> Hub 地址: https://huggingface.co/{hub_model_id}")
    print(f">>> 你可以通过以下代码加载该模型:")
    print(f">>>   model = AutoModelForMaskedLM.from_pretrained('{hub_model_id}')")
    print(f">>>   tokenizer = AutoTokenizer.from_pretrained('{hub_model_id}')")

except Exception as e:
    print(f">>> 上传到 Hub 失败，错误信息: {e}")
    print(f">>> 请确保:")
    print(f">>>   1. 已登录 Hugging Face: 在终端运行 `huggingface-cli login`")
    print(f">>>   2. 或设置环境变量: export HF_TOKEN=your_token")
    print(f">>>   3. 用户名 '{hf_username}' 是你的有效 Hugging Face 用户名")
    print(f">>> 模型仍保存在本地目录: {output_dir}")

# =============================================================================
# 第十五部分：使用保存的模型进行测试
# =============================================================================
# 加载本地保存或 Hub 上的模型进行测试，验证保存是否成功

print("\n======测试保存的模型======")

# 从本地加载模型进行测试
print(f">>> 从本地目录加载模型...")
local_model = AutoModelForMaskedLM.from_pretrained(output_dir)
local_tokenizer = AutoTokenizer.from_pretrained(output_dir)

# 使用本地模型进行掩码预测测试
test_text = "This is a great [MASK]."
print(f">>> 测试文本: {test_text}")

inputs = local_tokenizer(test_text, return_tensors="pt")
token_logits = local_model(**inputs).logits
mask_token_index = torch.where(inputs["input_ids"] == local_tokenizer.mask_token_id)[1]
mask_token_logits = token_logits[0, mask_token_index, :]
top_5_tokens = torch.topk(mask_token_logits, 5, dim=1).indices[0].tolist()

print(f">>> 本地模型预测结果:")
for token in top_5_tokens:
    print(f"    '>>> {test_text.replace(local_tokenizer.mask_token, local_tokenizer.decode([token]))}'")

print("\n======训练和保存完成======")
