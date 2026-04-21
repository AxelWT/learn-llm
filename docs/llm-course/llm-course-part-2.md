### 阅读笔记

#### 总结

#### 5.DATASETS 库

- **如果我的数据集不在 hub 上怎么办？ 用datasets工具进行加载，比较简单**
- 支持几种常见的数据格式

| 类型参数               | 加载的指令                                                   |
|--------------------|---------------------------------------------------------|
| CSV & TSV          | `load_dataset("csv", data_files="my_file.csv")`         |
| Text files         | `load_dataset("text", data_files="my_file.txt")`        |
| JSON & JSON Lines  | `load_dataset("json", data_files="my_file.jsonl")`      |
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

---

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

---

- **大数据？Datasets 来救援！**
- 参考文件 datasets_large_data_principles.md 来了解 Datasets 实现大数据轻松访问的原理和具体操作代码

---

- **创建自己的数据集**
- 参考 ./src/llm-course/5-fetch-issues-*.py脚本

---

- **使用 FAISS 进行语义搜索**
-
    - 参考 ./src/llm-course/embedding-then-faiss-search.py脚本

---

#### 6.TOKENIZERS 库

- **当我们需要微调模型时，我们需要使用与模型预训练相同的tokenizer**
- 但是当我们想从头开始训练模型时应该选用哪个 tokenizer？使用来自其他领域或语言的语料库上预训练的 tokenizer 通常是不理想的
- 本章讲述如何在一份文本语料库上训练一个全新的 tokenizer，然后将使用它来预训练语言模型

- ⚠️ 训练 tokenizer 与训练模型不同！模型训练使用随机梯度下降使每个 batch 的 loss 小一点。
- 它本质上是随机的（这意味着在即使两次训练的参数和算法完全相同，你也必须设置一些随机数种子才能获得相同的结果）。
- 训练 tokenizer 是一个统计过程，它试图确定哪些子词最适合为给定的语料库选择，确定的过程取决于分词算法。
- 它是确定性的，这意味着在相同的语料库上使用相同的算法进行训练时，得到的结果总是相同的。

- doc: llm-course-part-2-tokenizers.md

---

#### 7.主要的 NLP 任务

##### **标记（token）分类**

- 参考答案是数据集提供的，对 token 进行分类预测之后，再对比参考答案计算 loss 来完成训练学习
- doc: llm-course-part-2-ner.md

##### **微调一个掩码（mask）语言模型**

- 参考答案是数据集中的原始文本的 copy ，collactor 对原始文本进行动态 mask并对齐原始文本 copy 的参考答案/labels，训练模型计算 loss 完成学习，属于自监督学习
- doc: llm-course-part-2-mlm.md

##### **翻译**

- 参考答案是原始数据集提供的，模型学习原始文本预测 logits 与参考答案比较计算 loss 完成学习，seq2seq任务
- doc: llm-course-part-2-translation.md

##### **文本摘要**

- 参考答案是原始数据集提供的（评论的标题），模型学习原始文本预测 logits 与参考答案比较计算 loss 完成学习，seq2seq任务
- doc: llm-course-part-2-summarization.md

##### **从头开始训练因果语言模型**

- 参考答案是数据集中的原始文本的 copy，训练模型计算 loss 完成学习，属于自监督学习
- doc: llm-course-part-2-clm.md

##### **问答系统**

- 参考答案是索引值（原始数据告诉了答案在文本中的起始字符位置），训练模型计算起始位置和结束位置的 loss 完成学习
- doc: llm-course-part-2-qa.md

---

#### 8.如何寻求帮助

- stackoverflow
- huggingface 论坛

---

#### 9.构建并分享你的模型

- gradio 使用

---