### 阅读笔记

#### 总结

#### 5.DATASETS 库
- **如果我的数据集不在 hub 上怎么办？ 用datasets工具进行加载，比较简单**
- 支持几种常见的数据格式

| 类型参数 | 加载的指令 |
|-----------|------------|
| CSV & TSV | `load_dataset("csv", data_files="my_file.csv")` |
| Text files | `load_dataset("text", data_files="my_file.txt")` |
| JSON & JSON Lines | `load_dataset("json", data_files="my_file.jsonl")` |
| Pickled DataFrames | `load_dataset("pandas", data_files="my_dataframe.pkl")` |

```Python
from dataclasses import field

from datasets import load_dataset

# 加载本地数据集

# 加载单个文件，测试 ok 的
# squad_it_dataset = load_dataset("json", data_files="./download/SQuAD_it-train.json", field="data")
#
# print(squad_it_dataset)
# print("加载完成")

# 加载多个文件
# data_files = {"train": "./download/SQuAD_it-train.json", "test": "./download/SQuAD_it-test.json"}
# squad_it_dataset = load_dataset("json", data_files=data_files, field="data")
# print(squad_it_dataset)
# print("加载完成")

# 从压缩包加载多个本地文件
data_files = {"train": "./download/SQuAD_it-train.json.gz", "test": "./download/SQuAD_it-test.json.gz"}
squad_it_dataset = load_dataset("json", data_files=data_files, field="data")
print(squad_it_dataset)

# 加载远程数据集
url = "https://github.com/crux82/squad-it/raw/master/"
data_files = {
    "train": url + "SQuAD_it-train.json.gz",
    "test": url + "SQuAD_it-test.json.gz",
}
squad_it_dataset = load_dataset("json", data_files=data_files, field="data")
```

- **是时候来学一下切片了**
```Python
from datasets import load_dataset
from pandas.tseries import frequencies

# 由于 tsv 仅仅是 csv 的一个变体，可以用加载 csv 文件的 load_dataset()函数并指定分隔符，来加载这些文件
data_files = {"train": "./download/drugsComTrain_raw.tsv", "test": "./download/drugsComTest_raw.tsv"}
drug_dataset = load_dataset("csv", data_files=data_files, delimiter="\t")
print(drug_dataset)

# Hugging Face datasets 库使用 Apache Arrow 作为后端，采用**惰性加载（lazy loading）**机制
# shuffle(seed=42) - 只创建一个打乱顺序的索引映射，不加载实际数据（shuffle(seed=42)相同的种子，每次运行的随机洗牌返回结果相同）
# select(range(1000)) - 创建一个包含 1000 个索引的视图，仍然不加载完整数据
drug_sample = drug_dataset["train"].shuffle(seed=42).select(range(1000))
print(drug_sample[:3])

# 该 case 中验证'Unnamed: 0'这个key是否为患者 ID 的猜想，验证结果是的
for split in drug_dataset.keys():
    print(split)
    assert len(drug_dataset[split]) == len(drug_dataset[split].unique("Unnamed: 0"))
# dataset工具使用，字段重命名
drug_dataset = drug_dataset.rename_column(
    original_column_name="Unnamed: 0", new_column_name="patient_id"
)
print(drug_dataset)


# map操作
def lowercase_condition(example):
    return {"condition": example["condition"].lower()}


# 过滤操作
def filter_nones(x):
    return x["condition"] is not None


drug_dataset = drug_dataset.filter(filter_nones).map(lowercase_condition)
# 验证map和filter是否成功，是的
print(drug_dataset["train"]["condition"][:3])


# 基于原数据集，创建新的列
# 定义一个函数，计算每条评论的字数
def compute_review_length(example):
    return {"review_length": len(example["review"].split())}


drug_dataset = drug_dataset.map(compute_review_length)
print(drug_dataset["train"][0])
# 根据上一步新生成的 review_length字段排序
print(drug_dataset["train"].sort("review_length")[:3])

# 🙋向数据集添加新列的另一种方法是使用函数 Dataset.add_column() ，在使用它时你可以通过 Python 列表或 NumPy 数组的方式提供数据，
# 在不适合使用 Dataset.map() 情况下可以很方便。

# 使用 Dataset.filter()功能来删除包含少于 30个单词评论
drug_dataset = drug_dataset.filter(lambda x: x["review_length"] > 30)
print(drug_dataset.num_rows)

# 对数据集中评论字段 HTML 字符数据进行解码
# "I&#039;m a transformer called BERT" => "I'm a transformer called BERT"
import html

drug_dataset = drug_dataset.map(lambda x: {"review": html.unescape(x["review"])})

# map()方法的超级加速
# batched=True  map函数会分批执行所需进行的操作（批量大小是可配置的，默认为 1000）,注意使用该字段后 map 函数接收的 lambda 表达式发生变化，返回的字典的值为列表
# 能加速的原因是列表推导式通常比同一代码中用 for 循环执行相同的代码更快，并未还通过同时访问多个元素而不是一个一个来处理提高了处理的速度
# num_proc=8 指定 map函数调用中使用的进程数 也是用于加速
new_drug_dataset = drug_dataset.map(
    lambda x: {"review": [html.unescape(o) for o in x["review"]]}, batched=True, num_proc=8
)

from transformers import AutoTokenizer

# 默认use_fast=True 为快速 tokenizer，也可设置为 False 慢速 tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=False)


def tokenize_function(examples):
    return tokenizer(examples["review"], truncation=True)


# 在机器学习中，一个样本通常可以为我们的模型提供一组特征，这组特征会储存在数据集的几个列，但在某些情况下，可以从单个样本提取多个特征
# 新增的特征多出来一行，处理方式有两种，
# 一是删除原数据集的其他行只使用该特征，那么多出来的行不用跟原数据集其他列数据的行数对齐；
# 二是复制原数据集的其他行来和新增的特征进行补齐；比如 一行有两列 A ｜ B，B 特征分开之后，形成两行 A ｜ B1，A ｜ B2

# 设置return_overflowing_tokens=True 分词时将一个特征超过的限定长度的部分再保存为新的特征
def tokenize_and_split(examples):
    return tokenizer(
        examples["review"],
        truncation=True,
        max_length=128,
        return_overflowing_tokens=True,
    )


result = tokenize_and_split(drug_dataset["train"][0])
# 验证 结果为[128, 49]表明结果变成了两行了，而不是原来的一行
print([len(inp) for inp in result["input_ids"]])

# 删除原数据集的其他行只使用该特征，那么多出来的行不用跟原数据集其他列数据的行数对齐；
tokenized_dataset = drug_dataset.map(
    tokenize_and_split, batched=True, remove_columns=drug_dataset["train"].column_names
)
# 结果为（206772, 138514）形成了新的多了很多行特征，同时未报错
print(len(tokenized_dataset["train"]), len(drug_dataset["train"]))


# 复制原数据集的其他行来和新增的特征进行补齐；比如 一行有两列 A ｜ B，B 特征分开之后，形成两行 A ｜ B1，A ｜ B2
def tokenize_and_split_mapping(examples):
    result = tokenizer(
        examples["review"],
        truncation=True,
        max_length=128,
        return_overflowing_tokens=True,
    )
    # 提取新旧索引之间的映射
    sample_map = result.pop("overflow_to_sample_mapping")
    for key, values in examples.items():
        result[key] = [values[i] for i in sample_map]
    return result


# 结果显示行数变多了但是字段数量未减少
tokenized_dataset = drug_dataset.map(tokenize_and_split_mapping, batched=True)
print(tokenized_dataset)

# Datasets 和 DataFrames 的相互转换

# 将数据集转换为 pandas
drug_dataset.set_format("pandas")
# 访问数据集的元素时，我们会得到一个 pandas.DataFrame而不是字典
print(drug_dataset["train"][:3])

# 从数据集中选择 drug_dataset[train]的所有数据来得到训练集数据
train_df = drug_dataset["train"][:]

# pandas 操作，聚合
frequencies = (
    train_df["condition"]
    .value_counts()
    .to_frame()
    .reset_index()
    .rename(columns={"index": "condition", "count": "frequency"})
)

# frequencies.head() 用于预览 DataFrame 的前几行数据。
frequencies.head()

from datasets import Dataset

# 将 pandas 的 dataFrame 格式数据转化为 dataset的字典json格式数据
freq_dataset = Dataset.from_pandas(frequencies)
print(freq_dataset)
# reset_format() 让数据集回到最适合 datasets 库原生操作（如 map、filter、save）的格式，避免 pandas 格式带来的副作用。
drug_dataset.reset_format()

# 创建验证集：训练过程，训练集训练模型参数，验证集调整超参数，测试集最终评估（模型训练时从来没见过来保证测试的有效性）
drug_dataset_clean = drug_dataset["train"].train_test_split(train_size=0.8, seed=42)
# 将默认的“test”部分重命名为“validation”
drug_dataset_clean["validation"] = drug_dataset_clean.pop("test")
# 将“test”部分添加到我们的“DatasetDict”中
drug_dataset_clean["test"] = drug_dataset["test"]
print(drug_dataset_clean)

# 保存数据集
# 虽然 Datasets 会缓存每个下载的数据集和对它执行的操作，但有时你会想要将数据集保存到磁盘（比如，以防缓存被删除）

# 1.保存数据为 Apache arrow 格式
# 将数据保存到当前脚本同层的 drug-review 文件夹
drug_dataset_clean.save_to_disk("drug-reviews")

# 保存数据集后，我们可以使用 load_from_disk()功能从磁盘读取数据：
from datasets import load_from_disk

drug_dataset_reloaded = load_from_disk("drug-reviews")
print(drug_dataset_reloaded)

# 2.保存数据为 json 格式
# 对于 CSV 和 JSON 格式，需要将每个部分（test，validation，train）存储为单独的文件
for split, dataset in drug_dataset_clean.items():
    dataset.to_json(f"drug-reviews-{split}.jsonl")

# 重新加载
data_files_jsonl = {
    "train": "drug-reviews-train.jsonl",
    "validation": "drug-reviews-validation.jsonl",
    "test": "drug-reviews-test.jsonl"
}
drug_dataset_reloaded_jsonl = load_dataset("json", data_files=data_files_jsonl)
print(drug_dataset_reloaded_jsonl)


# 附注
# 当你使用切片[:3]
# 时，Dataset
# 返回的是列式结构的字典，而不是
# 3
# 个独立的行记录。
#
# # 切片 [:3] 返回的结构：
# {
#     'patient_id': [1, 2, 3],
#     # 每个字段是一个包含3个元素的列表
#     'condition': ['a', 'b', 'c'],
#     'review': ['...', '...', '...'],
#     'review_length': [10, 20, 30]
# }
#
# # 而不是 3 个独立的行对象：
# {'patient_id': 1, 'condition': 'a',
#  ...}  # 第1行
# {'patient_id': 2, 'condition': 'b',
#  ...}  # 第2行
# {'patient_id': 3, 'condition': 'c',
#  ...}  # 第3行
#
# 这是因为
# datasets
# 库使用
# Apache
# Arrow
# 列式存储，切片操作按列返回数据，效率更高。
#
# 如何获取独立的行记录
#
# 如果你想逐行查看，可以这样：
#
# sorted_ds = drug_dataset["train"].sort("review_length")
#
# # 方法1：逐个索引（返回单个记录字典）
# for i in range(3):
#     print(sorted_ds[i])
#
#     # 方法2：遍历（返回单个记录）
# for example in sorted_ds.select(range(3)):
#     print(example)
#
# 这种列式返回的设计是
# datasets
# 库的正常行为，便于批量处理数据。

```

- **大数据？Datasets 来救援！**
- 参考文件 datasets_large_data_principles.md 来了解 Datasets 实现大数据轻松访问的原理和具体操作代码

- **创建自己的数据集**
- 参考 ./src/llm-course/5-fetch-issues-*.py脚本

- **使用 FAISS 进行语义搜索**
- - 参考 ./src/llm-course/embedding-then-faiss-search.py脚本

#### 6.TOKENIZERS 库
- **当我们需要微调模型时，我们需要使用与模型预训练相同的tokenizer**
- 但是当我们想从头开始训练模型时应该选用哪个 tokenizer？使用来自其他领域或语言的语料库上预训练的 tokenizer 通常是不理想的
- 本章讲述如何在一份文本语料库上训练一个全新的 tokenizer，然后将使用它来预训练语言模型

- ⚠️ 训练 tokenizer 与训练模型不同！模型训练使用随机梯度下降使每个 batch 的 loss 小一点。
- 它本质上是随机的（这意味着在即使两次训练的参数和算法完全相同，你也必须设置一些随机数种子才能获得相同的结果）。
- 训练 tokenizer 是一个统计过程，它试图确定哪些子词最适合为给定的语料库选择，确定的过程取决于分词算法。
- 它是确定性的，这意味着在相同的语料库上使用相同的算法进行训练时，得到的结果总是相同的。

#### 7.主要的 NLP 任务
- **标记（token）分类**
- 这个通用任务涵盖了所有可以表述为“给句子中的词或字贴上标签”的问题
- 例如：实体命名识别（NER）找出句子中的实体（如人物，地点或组织），词性标注（POS）将句子中每个单词标记为对应于特定的词性（名词动词等），分块（chunking）找出属于同一实体的 tokens 这个任务

```Python
"""
BERT 微调进行命名实体识别（NER）任务
==========================================

本脚本使用 Hugging Face Transformers 库对 BERT 模型进行微调，
用于 CoNLL-2003 数据集上的命名实体识别任务。

命名实体识别（NER）是自然语言处理中的一个重要任务，
目标是从文本中识别并分类命名实体，如人名、地名、组织名等。

CoNLL-2003 数据集包含以下四种实体类型：
- PER：人名（Person）
- ORG：组织名（Organization）
- LOC：地名（Location）
- MISC：其他命名实体（Miscellaneous）

标注格式采用 BIO 标注方案：
- B-XXX：实体的开始标记（Begin）
- I-XXX：实体的内部标记（Inside）
- O：非实体标记（Outside）

例如："John Smith lives in New York"
标注为：B-PER I-PER O O O B-LOC I-LOC
表示 "John Smith" 是一个人名实体，"New York" 是一个地名实体。

脚本使用 Accelerate 库实现分布式训练支持，
可以在单机多 GPU 或多机多 GPU 环境下运行。

作者：axelloo
日期：2026/04/13
"""

# ==================== 第一部分：数据集加载 ====================
# 导入 Hugging Face Datasets 库，用于加载和处理数据集
from datasets import load_dataset

# 加载 CoNLL-2003 数据集
# ------------------------------
# CoNLL-2003 是命名实体识别领域的经典基准数据集
# 该数据集来源于 2003 年 CoNLL 共享任务会议
# 数据集包含英语和德语两种语言，这里我们使用英语版本
#
# 数据集结构：
# - train：训练集，包含 14,041 个样本
# - validation：验证集，包含 3,250 个样本
# - test：测试集，包含 3,453 个样本
#
# 每个样本包含以下字段：
# - id：样本唯一标识符
# - tokens：分词后的单词列表
# - pos_tags：词性标注标签
# - chunk_tags：短语块标注标签
# - ner_tags：命名实体标注标签（我们主要关注这个字段）
#
# trust_remote_code=True 参数：
# 允许执行数据集加载脚本中的远程代码
# 这是因为某些数据集需要自定义加载逻辑
raw_datasets = load_dataset('conll2003', trust_remote_code=True)

# 打印数据集基本信息，了解数据集结构和规模
print(raw_datasets)

# 打印第一条训练样本，查看数据格式
# 输出示例：
# {'id': '0', 'tokens': ['EU', 'rejects', 'German', 'call', 'to', 'boycott', 'British', 'lamb', '.'],
#  'pos_tags': [22, 42, 16, 21, 35, 37, 16, 21, 7],
#  'chunk_tags': [11, 21, 11, 12, 21, 22, 11, 12, 0],
#  'ner_tags': [3, 0, 7, 0, 0, 0, 7, 0, 0]}
print(raw_datasets["train"][0])

# ==================== 第二部分：分词器加载 ====================
# 导入 Transformers 库的分词器模块
from transformers import AutoTokenizer

# 选择预训练模型检查点
# ------------------------------
# 这里使用 "bert-base-cased" 模型
# - bert：使用 BERT 模型架构
# - base：使用 Base 版本（12 层 Transformer，768 维隐藏层）
# - cased：保留大小写信息，对于 NER 任务很重要
#
# 为什么选择 cased 版本？
# NER 任务中，大小写信息对于识别实体很有帮助：
# - "Apple"（公司名）和 "apple"（水果）的区分
# - 人名通常首字母大写，如 "John"
# - 地名通常首字母大写，如 "London"
#
# 其他可选模型：
# - bert-base-uncased：不区分大小写，适用于一般文本理解任务
# - bert-large-cased：更大的模型，性能更好但训练更慢
# - roberta-base：RoBERTa 模型，BERT 的改进版本
# - distilbert-base-cased：轻量版 BERT，速度更快
model_checkpoint = "bert-base-cased"

# 加载与预训练模型配套的分词器
# ------------------------------
# 分词器的作用是将原始文本转换为模型可理解的 token 序列
#
# BERT 使用 WordPiece 分词算法：
# - 将单词拆分为子词单元（subword tokens）
# - 例如："unhappiness" → ["un", "happi", "ness"]
# - 使用 ## 标记子词的延续（如 "##ness"）
#
# AutoTokenizer 会自动选择正确的分词器类型
# 对于 bert-base-cased，会使用 BertTokenizerFast
# Fast 版本使用 Rust 实现，速度快且支持额外功能（如 word_ids()）
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
print(f"load tokenizer: {tokenizer}")


# ==================== 第三部分：标签对齐函数 ====================
# BERT 分词会产生子词（subword），需要将原始标签对齐到 token 级别
# 这是 NER 任务中使用预训练模型的关键步骤

def align_labels_with_tokens(labels, word_ids):
    """
    将单词级别的标签对齐到 token 级别

    背景：
    -------
    BERT 的 WordPiece 分词会将一个单词拆分为多个 token：
    例如："John" → ["J", "ohn"] 或 ["John"]（取决于词表）
    例如："New York" → ["New", "York"]（通常不会被拆分）
    例如："internationalization" → ["international", "##ization"]

    原始数据集的标签是单词级别的，每个单词一个标签：
    例如：单词 "John" 标签为 B-PER

    但模型需要 token 级别的标签，每个 token 一个标签：
    例如：如果 "John" 被拆分为 ["J", "ohn"]，
         我们需要为 "J" 和 "ohn" 分别分配标签

    对齐策略：
    -------
    1. 特殊 token（[CLS], [SEP], [PAD] 等）：标记为 -100
       - PyTorch 的 CrossEntropyLoss 默认忽略 -100 标签
       - 这些 token 不参与损失计算，也不属于任何实体

    2. 单词的第一个 token：使用原始标签
       - 如果单词 "John" 标签为 B-PER
       - 第一个 token "J" 标签也是 B-PER

    3. 单词的后续 token：将 B-XXX 改为 I-XXX
       - 保持实体标注的一致性
       - 例如：如果第一个 token 是 B-PER（标签索引 1）
         后续 token 应为 I-PER（标签索引 2）
       - CoNLL-2003 的标签编码中，B-XXX 是奇数，I-XXX 是偶数+1
         所以 B-XXX → I-XXX 只需要 +1

    参数：
    -------
    labels : list[int]
        原始单词级别的标签列表（整数索引）
        例如：[3, 0, 7, 0] 表示 [B-ORG, O, B-LOC, O]

    word_ids : list[int or None]
        每个 token 对应的单词索引
        - None 表示特殊 token（[CLS], [SEP]）
        - 0 表示对应第 0 个单词
        - 1 表示对应第 1 个单词
        - 同一单词的多个 token 会共享相同的 word_id
        例如：对于句子 "EU rejects"
        tokens: ["[CLS]", "EU", "re", "jects", "[SEP]"]
        word_ids: [None, 0, 1, 1, None]

    返回：
    -------
    new_labels : list[int]
        对齐后的 token 级别标签列表
        长度与 token 序列相同
        例如：[-100, 3, 0, 0, -100]
    """
    # 初始化新标签列表
    new_labels = []

    # 记录当前处理的单词索引
    # 用于判断是单词的第一个 token 还是后续 token
    current_word = None

    # 遍历每个 token 的 word_id
    for word_id in word_ids:
        # 判断是否是新单词的开始
        if word_id != current_word:
            # 这是新单词的第一个 token，或者是特殊 token
            current_word = word_id

            # 特殊 token（word_id 为 None）标记为 -100
            # 否则使用该单词的原始标签
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)

        elif word_id is None:
            # 这是后续的特殊 token（如多个 [PAD]）
            # 标记为 -100，不参与损失计算
            new_labels.append(-100)

        else:
            # 这是同一单词的后续 token（子词）
            # 获取该单词的原始标签
            label = labels[word_id]

            # 如果标签是 B-XXX（奇数索引），改为 I-XXX（偶数索引+1）
            # CoNLL-2003 标签编码规则：
            # - 0: O（非实体）
            # - 1: B-PER, 2: I-PER
            # - 3: B-ORG, 4: I-ORG
            # - 5: B-LOC, 6: I-LOC
            # - 7: B-MISC, 8: I-MISC
            # 可以看到 B-XXX 都是奇数，I-XXX 都是对应的偶数
            # 所以 B-XXX → I-XXX 只需要将标签 +1
            if label % 2 == 1:
                label += 1
            new_labels.append(label)

    return new_labels


def tokenize_and_align_labels(examples):
    """
    批量处理：分词 + 标签对齐

    这个函数会被 datasets.map() 方法调用，
    用于批量处理整个数据集。

    参数：
    -------
    examples : dict
        包含一批样本数据的字典
        关键字段：
        - "tokens": list[list[str]]，多个样本的单词列表
          例如：[["EU", "rejects"], ["John", "lives"]]
        - "ner_tags": list[list[int]]，多个样本的标签列表
          例如：[[3, 0], [1, 0]]

    返回：
    -------
    tokenized_inputs : dict
        包含分词结果和对齐后标签的字典
        关键字段：
        - "input_ids": token ID 序列
        - "attention_mask": 注意力掩码（1 表示真实 token，0 表示填充）
        - "labels": 对齐后的标签序列

    处理流程：
    -------
    1. 使用 tokenizer 对句子列表进行批量分词
       - truncation=True：超过最大长度时截断（BERT 最大长度 512）
       - is_split_into_words=True：输入已经是分词后的单词列表
         （不需要再次分词，只需进行 WordPiece 子词拆分）

    2. 对于每个样本，获取 word_ids() 映射
       - word_ids() 是 Fast Tokenizer 的特有功能
       - 返回每个 token 对应的原始单词索引

    3. 使用 align_labels_with_tokens() 对齐标签
       - 将单词级标签转换为 token 级标签

    4. 将对齐后的标签添加到 tokenized_inputs["labels"]
    """
    # 使用 tokenizer 进行批量分词
    # ------------------------------
    # examples["tokens"] 是单词列表的列表
    # tokenizer 会返回一个 BatchEncoding 对象
    # 包含 input_ids, attention_mask, token_type_ids 等字段
    #
    # truncation=True 参数：
    # - 当句子长度超过模型最大长度（512）时自动截断
    # - NER 任务中句子通常不会太长，但为了安全起见启用截断
    #
    # is_split_into_words=True 参数：
    # - 表示输入已经是预分词的单词列表
    # - tokenizer 只需要进行 WordPiece 子词拆分
    # - 如果不设置这个参数，tokenizer 会先进行空格分词
    #   可能与原始数据集的分词不一致
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True
    )

    # 获取原始 NER 标签
    all_labels = examples["ner_tags"]

    # 初始化新标签列表（用于存储对齐后的标签）
    new_labels = []

    # 遍历每个样本，进行标签对齐
    # ------------------------------
    # enumerate(all_labels) 遍历每个样本的标签列表
    # i 是样本在批次中的索引
    # labels 是该样本的标签列表
    for i, labels in enumerate(all_labels):
        # 获取该样本的 word_ids 映射
        # ------------------------------
        # tokenized_inputs.word_ids(i) 返回第 i 个样本的 word_ids
        # 这是一个列表，长度等于 token 序列长度
        # 每个元素是对应 token 的原始单词索引
        word_ids = tokenized_inputs.word_ids(i)

        # 使用 align_labels_with_tokens 进行标签对齐
        # ------------------------------
        # 将单词级标签转换为 token 级标签
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    # 将对齐后的标签添加到分词结果中
    # ------------------------------
    # tokenized_inputs 是一个字典，我们添加 "labels" 字段
    # 这样分词结果就包含了：input_ids, attention_mask, labels
    # 这些字段将用于模型训练
    tokenized_inputs["labels"] = new_labels

    return tokenized_inputs


# ==================== 第四部分：执行数据预处理 ====================
# 使用 datasets.map() 方法对整个数据集进行预处理
# ------------------------------
# map() 方法是 Datasets 库的核心功能之一
# 它会遍历数据集中的每个样本，应用指定的处理函数
#
# 参数说明：
# - tokenize_and_align_labels：处理函数
# - batched=True：启用批处理模式
#   - 处理函数接收一批样本而非单个样本
#   - Fast Tokenizer 的批处理速度比逐个处理快很多
#   - 典型情况下，批处理可以提速 10-100 倍
#
# - remove_columns：移除原始列
#   - 原始数据集包含：id, tokens, pos_tags, chunk_tags, ner_tags
#   - 分词后我们只需要：input_ids, attention_mask, labels
#   - 移除原始列可以减少内存占用，避免列名冲突
#
# map() 方法返回一个新的 Dataset 对象
# 包含处理后的数据
tokenized_datasets = raw_datasets.map(
    tokenize_and_align_labels,
    batched=True,
    remove_columns=raw_datasets["train"].column_names
)

# 打印预处理后的数据集信息
# ------------------------------
# 可以看到数据集列已经变为：input_ids, attention_mask, labels
print(f"tokenized_datasets: {tokenized_datasets}")

# ==================== 第五部分：数据整理器 ====================
# 导入 Token Classification 专用的数据整理器
from transformers import DataCollatorForTokenClassification

# 创建数据整理器
# ------------------------------
# 数据整理器（Data Collator）的作用：
# - 将多个样本整理成一个批次（batch）
# - 处理不同长度样本的填充（padding）
# - 创建模型需要的输入格式
#
# DataCollatorForTokenClassification 专门用于 token 分类任务：
# - 动态填充：使同一批次内所有样本长度一致
# - 填充到批次内最长样本的长度（而非固定最大长度）
# - 这样可以减少不必要的填充，提高计算效率
#
# tokenizer 参数：
# - 用于获取填充 token 的 ID（通常是 0）
# - 用于获取填充位置的 attention_mask（通常是 0）
#
# 对于 NER 任务，标签也需要填充：
# - 通常使用 -100 作为填充标签
# - 这样填充位置不参与损失计算
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
print(f"data_collator: {data_collator}")

# ==================== 第六部分：评估指标加载 ====================
# 导入 Evaluate 库，用于计算评估指标
import evaluate

# 加载 seqeval 评估指标
# ------------------------------
# seqeval 是专为序列标注任务设计的评估指标库
# 与传统的分类指标不同，seqeval 按实体级别评估
#
# 为什么需要 seqeval？
# ------------------------------
# 传统的 token-level 精确率/召回率不能很好地反映 NER 性能
# 例如：预测 "John Smith" 为 ["B-PER", "O"]
# - token-level：B-PER 正确（1/2），O 错误（0/1）
# - entity-level：整个 "John Smith" 实体预测错误
#
# seqeval 的评估方式：
# - 将连续的 B-XXX I-XXX I-XXX 序列视为一个完整实体
# - 只有当整个实体边界和类型都正确时才算正确
# - 例如：真实标签 ["B-PER", "I-PER"]
#         预测 ["B-PER", "I-PER"] → 完全正确
#         预测 ["B-PER", "O"] → 实体边界错误
#         预测 ["B-ORG", "I-ORG"] → 实体类型错误
#
# seqeval 返回的指标：
# - precision：精确率（正确预测实体数 / 预测实体总数）
# - recall：召回率（正确预测实体数 / 真实实体总数）
# - f1：F1 分数（precision 和 recall 的调和平均）
# - accuracy：准确率（正确 token 数 / 总 token 数）
# - 还会返回每种实体类型的详细指标
metric = evaluate.load("seqeval")
print(f"metric: {metric}")


# 测试评估指标函数
def test_eval(datasets):
    """
    测试 seqeval 指标的计算方式

    这个函数用于演示 seqeval 的使用方法
    不参与实际训练流程

    参数：
    -------
    datasets : DatasetDict
        原始数据集对象
    """
    # 获取第一条训练样本的 NER 标签
    # labels 是整数索引列表，如 [3, 0, 7, 0, 0, 0, 7, 0, 0]
    labels = datasets["train"][0]["ner_tags"]

    # 获取标签名称列表
    # label_names 是标签索引到标签名的映射列表
    # 如 ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
    label_names = datasets["train"].features["ner_tags"].feature.names

    # 打印标签索引和对应的标签名
    print(labels)
    print([label_names[i] for i in labels])

    # 模拟预测结果：将第 3 个标签改为 0（"O"）
    # ------------------------------
    # 这模拟了一个预测错误的情况
    # 例如：真实标签是 B-LOC，但预测为 O
    predictions = labels.copy()
    predictions[2] = 0

    # seqeval 要求输入字符串标签名，而非整数索引
    # ------------------------------
    # 将整数索引转换为字符串标签名
    # 例如：[3, 0] → ["B-ORG", "O"]
    predictions_str = [label_names[i] for i in predictions]
    labels_str = [label_names[i] for i in labels]

    # 计算评估指标
    # ------------------------------
    # predictions 和 references 都需要是列表的列表
    # 因为 seqeval 支持批量评估多个样本
    print(metric.compute(predictions=[predictions_str], references=[labels_str]))


# 注释掉测试函数，避免在训练时执行
# 如果需要测试，可以取消注释
# test_eval(raw_datasets)


# 导入 NumPy，用于处理预测结果
import numpy as np


def compute_metrics(eval_preds):
    """
    计算模型评估指标（用于 Trainer API）

    这个函数是为 Trainer API 设计的回调函数
    在每个评估步骤结束后被调用，计算评估指标

    注意：本脚本使用自定义训练循环，不使用 Trainer API
    所以这个函数在脚本中未被使用，仅供参考

    参数：
    -------
    eval_preds : tuple
        包含模型预测结果和真实标签的元组
        - logits: numpy.ndarray
          模型输出的未归一化概率（logits）
          形状为 (batch_size, seq_len, num_labels)
          num_labels 是标签类别数（CoNLL-2003 为 9）
        - labels: numpy.ndarray
          真实标签
          形状为 (batch_size, seq_len)
          -100 表示忽略的位置（特殊 token 和填充）

    返回：
    -------
    dict
        包含四个整体指标的字典：
        - precision：整体精确率
        - recall：整体召回率
        - f1：整体 F1 分数
        - accuracy：整体准确率
    """
    # 获取标签名称列表
    # ------------------------------
    # 用于将数字索引转换为字符串标签名
    # seqeval 需要字符串标签名作为输入
    label_names = raw_datasets["train"].features["ner_tags"].feature.names

    # 解包预测结果和真实标签
    logits, labels = eval_preds

    # 对每个位置取概率最大的标签索引作为预测结果
    # ------------------------------
    # np.argmax(logits, axis=-1) 的作用：
    # - logits 形状：(batch_size, seq_len, num_labels)
    # - axis=-1 表示沿着最后一个维度（标签类别维度）取最大值索引
    # - predictions 形状：(batch_size, seq_len)
    # - 每个位置的值是预测的标签索引（0-8）
    predictions = np.argmax(logits, axis=-1)

    # 处理真实标签
    # ------------------------------
    # 过滤掉 -100（特殊 token 和填充位置）
    # 并将索引转为标签名
    #
    # 列表推导式说明：
    # - 外层遍历每个样本：for label in labels
    # - 内层遍历每个位置的标签：for l in label
    # - 过滤条件：if l != -100（只保留有效标签）
    # - 转换为标签名：label_names[l]
    #
    # 结果示例：
    # labels = [[-100, 3, 0, -100], [-100, 1, 2, -100]]
    # → true_labels = [["B-ORG", "O"], ["B-PER", "I-PER"]]
    true_labels = [
        [label_names[l] for l in label if l != -100]
        for label in labels
    ]

    # 处理预测标签
    # ------------------------------
    # 同样过滤掉对应真实标签为 -100 的位置
    #
    # 为什么需要 zip(prediction, label)？
    # - 我们需要根据真实标签来判断哪些位置应该忽略
    # - 即使预测位置有值，如果真实标签是 -100，该位置也应忽略
    #
    # 列表推导式说明：
    # - zip(prediction, label) 同时遍历预测和真实标签
    # - if l != -100 根据真实标签过滤
    # - label_names[p] 将预测索引转为标签名
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    # 使用 seqeval 计算序列标注的评估指标
    # ------------------------------
    # seqeval.compute() 返回一个字典
    # 包含整体指标和每种实体类型的详细指标
    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)

    # 返回整体指标
    # ------------------------------
    # seqeval 返回的整体指标键名带有 "overall_" 前缀
    # 我们提取并返回四个关键指标
    return {
        "precision": all_metrics["overall_precision"],
        "recall": all_metrics["overall_recall"],
        "f1": all_metrics["overall_f1"],
        "accuracy": all_metrics["overall_accuracy"],
    }


# ==================== 第七部分：模型定义 ====================
# 获取标签名称列表
# ------------------------------
# 用于创建标签索引和标签名之间的映射
label_names = raw_datasets["train"].features["ner_tags"].feature.names

# 创建标签索引到标签名的映射字典（id2label）
# ------------------------------
# 这个字典用于模型输出解码
# 当模型预测一个标签索引时，可以转换为对应的标签名
#
# 例如：
# id2label = {
#     "0": "O",
#     "1": "B-PER",
#     "2": "I-PER",
#     "3": "B-ORG",
#     "4": "I-ORG",
#     "5": "B-LOC",
#     "6": "I-LOC",
#     "7": "B-MISC",
#     "8": "I-MISC"
# }
#
# 注意：键是字符串类型（"0", "1"），而非整数
# 这是因为 Hugging Face 模型配置约定使用字符串键
id2label = {str(i): label for i, label in enumerate(label_names)}

# 创建标签名到索引的反向映射字典（label2id）
# ------------------------------
# 这个字典用于将标签名转换为索引
#
# 例如：
# label2id = {
#     "O": "0",
#     "B-PER": "1",
#     "I-PER": "2",
#     ...
# }
#
# 这个字典在模型加载时会用到
# 用于配置模型的输出层
label2id = {v: k for k, v in id2label.items()}

# 导入 Token Classification 模型类
from transformers import AutoModelForTokenClassification

# 加载预训练的 BERT 模型，并配置为 token 分类任务
# ------------------------------
# AutoModelForTokenClassification 会自动加载正确的模型类
# 对于 bert-base-cased，会使用 BertForTokenClassification
#
# BertForTokenClassification 的结构：
# - BERT 编码器：12 层 Transformer
# - 分类头：一个线性层，将隐藏状态映射到标签空间
# - 输出层维度：num_labels（这里是 9）
#
# 参数说明：
# - model_checkpoint：预训练模型名称或路径
# - id2label：标签索引到标签名的映射
# - label2id：标签名到索引的映射
#
# 注意事项：
# - 预训练模型的分类头是随机初始化的
# - 因为预训练任务不是 token 分类
# - 所以需要微调来学习 NER 任务
# - 微调时，BERT 编码器的参数也会更新
model = AutoModelForTokenClassification.from_pretrained(
    model_checkpoint,
    id2label=id2label,
    label2id=label2id,
)

# 打印模型信息，查看模型结构
# ------------------------------
# 可以看到模型配置和参数数量
print(f"load model: {model}")

# ==================== 第八部分：自定义训练循环 ====================
# 本脚本使用自定义训练循环而非 Trainer API
# 这样可以更好地控制训练过程，并支持分布式训练

# 导入 PyTorch DataLoader
from torch.utils.data import DataLoader

# 创建训练数据加载器
# ------------------------------
# DataLoader 是 PyTorch 的数据加载工具
# 用于将数据集分批次加载到模型
#
# 参数说明：
# - tokenized_datasets["train"]：预处理后的训练数据集
# - shuffle=True：每个 epoch 开始时打乱数据顺序
#   - 避免模型学习数据顺序的模式
#   - 提高训练的稳定性和收敛性
# - collate_fn=data_collator：使用自定义数据整理器
#   - 处理批内样本的填充和对齐
# - batch_size=8：每批次包含 8 个样本
#   - 较小的批次适合 NER 任务
#   - 可以根据 GPU 内存调整（典型范围：8-32）
train_dataloader = DataLoader(
    tokenized_datasets["train"],
    shuffle=True,
    collate_fn=data_collator,
    batch_size=8,
)

# 创建验证数据加载器
# ------------------------------
# 参数说明：
# - tokenized_datasets["validation"]：预处理后的验证数据集
# - shuffle=False：验证时不需要打乱数据顺序
#   - 保持数据顺序便于调试和结果分析
# - collate_fn=data_collator：使用相同的数据整理器
# - batch_size=8：每批次包含 8 个样本
eval_dataloader = DataLoader(
    tokenized_datasets["validation"],
    collate_fn=data_collator,
    batch_size=8,
)

# 导入 AdamW 优化器
from torch.optim import AdamW

# 创建优化器
# ------------------------------
# AdamW 是 Adam 优化器的改进版本
# - Adam：自适应学习率优化器，结合了 RMSprop 和 Momentum
# - AdamW：在 Adam 基础上添加权重衰减（weight decay）的解耦
#   - 更好的正则化效果
#   - 训练更稳定
#
# 参数说明：
# - model.parameters()：模型的可训练参数
# - lr=2e-5：学习率
#   - 对于预训练模型的微调，通常使用较小的学习率
#   - 防止破坏预训练的知识
#   - 典型范围：1e-5 到 5e-5
optimizer = AdamW(model.parameters(), lr=2e-5)

# 导入 Accelerate 库
from accelerate import Accelerator

# 创建 Accelerator 对象
# ------------------------------
# Accelerate 是 Hugging Face 的分布式训练库
# 提供简洁的 API 支持多 GPU、多机器训练
#
# Accelerator 的功能：
# - 自动检测可用设备（CPU、单 GPU、多 GPU）
# - 自动配置分布式训练环境
# - 处理模型、优化器、数据加载器的设备迁移
# - 处理梯度同步和分布式评估
#
# 使用 Accelerator 的好处：
# - 同一份代码可以在不同硬件环境下运行
# - 无需手动处理设备迁移（.to(device)）
# - 自动处理分布式训练的复杂细节
accelerator = Accelerator()

# 使用 Accelerator 准备训练组件
# ------------------------------
# accelerator.prepare() 会自动处理：
# - 模型：包装为分布式模型（如 DistributedDataParallel）
# - 优化器：无变化，但确保在正确设备上
# - 数据加载器：包装为分布式数据加载器
#   - 每个进程只处理数据的一部分
#   - 分布式采样器自动处理数据分发
#
# prepare() 返回的是包装后的对象
# 我们用返回值替换原始对象
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)

# 导入学习率调度器
from transformers import get_scheduler

# 设置训练参数
# ------------------------------
# num_train_epochs：训练的总 epoch 数
# - 一个 epoch 是遍历整个训练数据集一次
# - 对于 NER 微调，通常 3-5 个 epoch 就足够
# - 过多 epoch 可能导致过拟合
num_train_epochs = 3

# 计算每个 epoch 的更新步数
# ------------------------------
# len(train_dataloader)：数据加载器中的批次数量
# - 对于单 GPU：批次数量 = 数据集大小 / batch_size
# - 对于多 GPU：每个 GPU 的批次数量减少
#   - 因为数据被分发到多个 GPU
num_update_steps_per_epoch = len(train_dataloader)

# 计算总训练步数
# ------------------------------
# 总训练步数 = epoch 数 × 每个 epoch 的更新步数
# 用于学习率调度器的配置
num_training_steps = num_train_epochs * num_update_steps_per_epoch

# 创建学习率调度器
# ------------------------------
# 学习率调度器控制训练过程中的学习率变化
#
# "linear" 调度器（线性衰减）：
# - 学习率从初始值线性衰减到 0
# - 是预训练模型微调的常用策略
# - 训练后期学习率减小，有助于稳定收敛
#
# 参数说明：
# - optimizer：要调度学习率的优化器
# - num_warmup_steps=0：预热步数
#   - 预热阶段学习率从 0 线性增加到初始值
#   - 对于微调，通常不需要预热
# - num_training_steps：总训练步数
#   - 衰减在训练结束时达到 0
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

# 导入 Hugging Face Hub 工具
# ------------------------------
# 用于将训练好的模型上传到 Hugging Face Hub
# 便于模型分享和后续使用
from huggingface_hub import create_repo, get_full_repo_name

# 设置模型名称
# ------------------------------
# 这是上传到 Hub 后的模型仓库名称
model_name = "bert-finetuned-ner-accelerate"

# 获取完整的仓库名称
# ------------------------------
# get_full_repo_name() 会将模型名转换为 Hub 上的完整路径
# 格式为：{用户名}/{模型名}
# 例如：your-username/bert-finetuned-ner-accelerate
repo_name = get_full_repo_name(model_name)
print(repo_name)

# 设置本地输出目录
# ------------------------------
# 模型会保存到这个目录
output_dir = "bert-finetuned-ner-accelerate"

# 创建 Hub 仓库
# ------------------------------
# create_repo() 在 Hugging Face Hub 上创建模型仓库
# 参数说明：
# - repo_name：仓库名称
# - repo_type="model"：仓库类型（模型仓库）
# - exist_ok=True：如果仓库已存在，不报错
#   - 允许多次运行脚本而不出错
create_repo(repo_name, repo_type="model", exist_ok=True)


def postprocess(predictions, labels):
    """
    后处理函数：将模型输出转换为评估格式

    在分布式训练的评估阶段使用
    将 GPU 上的 tensor 转换为 CPU 上的 numpy 数组
    并过滤掉填充位置，转换为标签名

    参数：
    -------
    predictions : torch.Tensor
        预测标签，形状为 (batch_size, seq_len)
        值为标签索引（0-8）或填充值（-100）

    labels : torch.Tensor
        真实标签，形状为 (batch_size, seq_len)
        值为标签索引（0-8）或填充值（-100）

    返回：
    -------
    true_labels : list[list[str]]
        真实标签名列表（每个样本一个列表）
        已过滤掉填充位置

    true_predictions : list[list[str]]
        预测标签名列表（每个样本一个列表）
        已过滤掉填充位置
    """
    # 将 tensor 从 GPU 移到 CPU，并转换为 numpy 数组
    # ------------------------------
    # .detach()：从计算图分离，不记录梯度
    # .cpu()：从 GPU 移到 CPU
    # .clone()：创建副本，避免修改原始数据
    # .numpy()：转换为 numpy 数组
    predictions = predictions.detach().cpu().clone().numpy()
    labels = labels.detach().cpu().clone().numpy()

    # 处理真实标签
    # ------------------------------
    # 过滤掉 -100（填充和特殊 token）
    # 将索引转换为标签名
    true_labels = [
        [label_names[l] for l in label if l != -100]
        for label in labels
    ]

    # 处理预测标签
    # ------------------------------
    # 根据真实标签过滤（如果真实标签是 -100，预测也忽略）
    # 将索引转换为标签名
    true_predictions = [
        [label_names[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    return true_labels, true_predictions


# 导入进度条工具
from tqdm.auto import tqdm

# 导入 PyTorch
import torch

# 创建进度条
# ------------------------------
# tqdm 会在训练过程中显示进度条
# 显示当前进度、预计剩余时间等
progress_bar = tqdm(range(num_training_steps))

# ==================== 第九部分：训练循环 ====================
# 主训练循环：遍历 epoch，执行训练和评估

for epoch in range(num_train_epochs):
    # ==================== 训练阶段 ====================
    # 设置模型为训练模式
    # ------------------------------
    # model.train() 会启用：
    # - Dropout：防止过拟合
    # - Batch Normalization 的更新
    # - Gradient 计算和存储
    #
    # 与 model.eval() 的区别：
    # - eval() 模式下 Dropout 关闭
    # - eval() 模式下不计算梯度
    model.train()

    # 遍历训练数据加载器的每个批次
    # ------------------------------
    # train_dataloader 是经过 accelerator.prepare() 包装的
    # 在分布式训练时，每个进程只处理一部分数据
    for batch in train_dataloader:
        # 前向传播：计算模型输出
        # ------------------------------
        # model(**batch) 会自动将 batch 中的数据输入模型
        # batch 包含：input_ids, attention_mask, labels
        # outputs 包含：logits（预测）, loss（损失）
        outputs = model(**batch)

        # 获取损失值
        # ------------------------------
        # BertForTokenClassification 会自动计算交叉熵损失
        # 损失只计算标签不为 -100 的位置
        loss = outputs.loss

        # 反向传播：计算梯度
        # ------------------------------
        # accelerator.backward(loss) 是分布式训练的梯度计算
        # 等价于 loss.backward()，但处理分布式梯度同步
        # 梯度会存储在每个参数的 .grad 属性中
        accelerator.backward(loss)

        # 参数更新：使用梯度更新模型参数
        # ------------------------------
        # optimizer.step() 根据梯度和学习率更新参数
        optimizer.step()

        # 学习率更新：更新学习率调度器
        # ------------------------------
        # lr_scheduler.step() 根据调度策略更新学习率
        # 对于线性调度器，学习率会逐渐减小
        lr_scheduler.step()

        # 清零梯度：为下一个批次做准备
        # ------------------------------
        # optimizer.zero_grad() 清除之前计算的梯度
        # PyTorch 默认会累积梯度，需要手动清零
        # 如果不清零，梯度会累积导致参数更新错误
        optimizer.zero_grad()

        # 更新进度条
        # ------------------------------
        # progress_bar.update(1) 增加进度条进度
        progress_bar.update(1)

    # ==================== 评估阶段 ====================
    # 设置模型为评估模式
    # ------------------------------
    # model.eval() 会：
    # - 关闭 Dropout（使用完整的模型能力）
    # - 关闭 Batch Normalization 的更新
    # - 不计算梯度（节省内存和时间）
    model.eval()

    # 遍历验证数据加载器的每个批次
    # ------------------------------
    # eval_dataloader 也经过 accelerator.prepare() 包装
    for batch in eval_dataloader:
        # 禁用梯度计算
        # ------------------------------
        # torch.no_grad() 上下文管理器
        # 在评估时不需要梯度，禁用可以：
        # - 节省内存（不存储梯度）
        # - 加速计算
        with torch.no_grad():
            # 前向传播：获取模型输出
            # ------------------------------
            # 评估时只关心 logits（预测结果）
            # 不计算损失（虽然也可以计算）
            outputs = model(**batch)

        # 获取预测结果
        # ------------------------------
        # outputs.logits 形状：(batch_size, seq_len, num_labels)
        # .argmax(dim=-1) 取每个位置概率最大的标签索引
        # predictions 形状：(batch_size, seq_len)
        predictions = outputs.logits.argmax(dim=-1)

        # 获取真实标签
        # ------------------------------
        # batch["labels"] 是对齐后的标签序列
        # 包含有效标签（0-8）和填充标签（-100）
        labels = batch["labels"]

        # 填充预测和标签以便跨进程收集
        # ------------------------------
        # accelerator.pad_across_processes() 用于分布式评估
        # 不同进程产生的预测/标签可能有不同长度
        # 因为分布式数据加载器分配的数据可能不同
        #
        # 填充使所有进程的预测/标签长度一致
        # 这样才能正确收集（gather）所有结果
        #
        # 参数说明：
        # - dim=1：沿着序列维度填充
        # - pad_index=-100：使用 -100 作为填充值
        predictions = accelerator.pad_across_processes(predictions, dim=1, pad_index=-100)
        labels = accelerator.pad_across_processes(labels, dim=1, pad_index=-100)

        # 收集所有进程的预测和标签
        # ------------------------------
        # accelerator.gather() 收集所有分布式进程的结果
        # 返回所有进程预测/标签的拼接结果
        # 这样可以评估整个验证集而非每个进程的部分
        predictions_gathered = accelerator.gather(predictions)
        labels_gathered = accelerator.gather(labels)

        # 后处理：转换为评估格式
        # ------------------------------
        # postprocess() 将 tensor 转换为标签名列表
        # 过滤掉填充位置
        true_labels, true_predictions = postprocess(predictions_gathered, labels_gathered)

        # 添加到评估指标计算器
        # ------------------------------
        # metric.add_batch() 将当前批次的结果添加到评估器
        # 评估器会累积所有批次的结果
        # 最后调用 metric.compute() 计算整体指标
        metric.add_batch(predictions=true_predictions, references=true_labels)

    # 计算评估指标
    # ------------------------------
    # metric.compute() 返回所有批次的累积指标
    # 包含整体指标和每种实体类型的详细指标
    results = metric.compute()

    # 打印评估结果
    # ------------------------------
    # 只打印整体指标（precision, recall, f1, accuracy）
    # 不打印每种实体类型的详细指标
    print(
        f"epoch {epoch}:",
        {
            key: results[f"overall_{key}"]
            for key in ["precision", "recall", "f1", "accuracy"]
        },
    )

    # ==================== 模型保存和上传 ====================
    # 等待所有进程完成评估
    # ------------------------------
    # accelerator.wait_for_everyone() 是同步点
    # 确保所有分布式进程都完成当前 epoch
    # 然后才能保存模型（避免部分进程还在运行）
    accelerator.wait_for_everyone()

    # 解包装模型
    # ------------------------------
    # accelerator.unwrap_model() 从分布式包装中取出原始模型
    # 分布式训练时模型被 DistributedDataParallel 包装
    # 保存时需要保存原始模型而非包装后的模型
    unwrapped_model = accelerator.unwrap_model(model)

    # 保存模型到本地
    # ------------------------------
    # unwrapped_model.save_pretrained() 保存模型参数和配置
    # 参数说明：
    # - output_dir：保存目录
    # - save_function：保存函数（由 accelerator 提供）
    #   - 用于分布式训练的保存
    #   - 确保只有主进程执行保存
    unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)

    # 主进程额外保存分词器并上传模型
    # ------------------------------
    # accelerator.is_main_process() 判断是否是主进程
    # 只有主进程执行上传操作，避免重复上传
    if accelerator.is_main_process:
        # 保存分词器
        # ------------------------------
        # tokenizer.save_pretrained() 保存分词器配置和词表
        # 模型加载时需要相同的分词器才能正确处理输入
        tokenizer.save_pretrained(output_dir)

        # 上传模型到 Hugging Face Hub
        # ------------------------------
        # unwrapped_model.push_to_hub() 上传模型到 Hub
        # 参数说明：
        # - repo_name：目标仓库名称
        # - commit_message：提交信息
        #   - 标注当前训练进度（epoch 数）
        unwrapped_model.push_to_hub(
            repo_name,
            commit_message=f"Training in progress epoch {epoch}"
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

```

#### 8.如何寻求帮助

#### 9.构建并分享你的模型