# 数据处理

## 目录

- [5.DATASETS 库](#5datasets-库)
  - [1.数据加载](#1数据加载)
  - [2.数据切片](#2数据切片)
  - [3.大数据处理](#3大数据处理)
  - [4.创建自己的数据集](#4创建自己的数据集)
  - [5.使用 FAISS 进行语义搜索](#5使用-faiss-进行语义搜索)
    - [什么是 Embedding？](#什么是-embedding)
    - [为什么是 768 维 Embedding？](#为什么是-768-维-embedding)
    - [Embedding 与 RAG 的联系](#embedding-与-rag-的联系)
    - [Agentic RAG（Agent 风格 RAG）](#agentic-ragagent-风格-rag)
    - [检索文档过长超出上下文的解决方案](#检索文档过长超出上下文的解决方案)

---

## 5.DATASETS 库

### 1.数据加载

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

### 2.数据切片

- 类似于 java 中 stream API 的 map filter reduce 等处理操作

```Python
from datasets import load_dataset

# 由于 tsv 仅仅是 csv 的一个变体，可以用加载 csv 文件的 load_dataset()函数并指定分隔符，来加载这些文件
data_files = {"train": "../download/drugsComTrain_raw.tsv", "test": "../download/drugsComTest_raw.tsv"}
drug_dataset = load_dataset("csv", data_files=data_files, delimiter="\t")
print(drug_dataset)
"""
DatasetDict({
    train: Dataset({
        features: ['Unnamed: 0', 'drugName', 'condition', 'review', 'rating', 'date', 'usefulCount'],
        num_rows: 161297
    })
    test: Dataset({
        features: ['Unnamed: 0', 'drugName', 'condition', 'review', 'rating', 'date', 'usefulCount'],
        num_rows: 53766
    })
})
"""

# Hugging Face datasets 库使用 Apache Arrow 作为后端，采用**惰性加载（lazy loading）**机制
# shuffle(seed=42) - 只创建一个打乱顺序的索引映射，不加载实际数据（shuffle(seed=42)相同的种子，每次运行的随机洗牌返回结果相同）
# select(range(1000)) - 创建一个包含 1000 个索引的视图，仍然不加载完整数据
drug_sample = drug_dataset["train"].shuffle(seed=42).select(range(1000))
print(drug_sample[:3])
"""
{'Unnamed: 0': [87571, 178045, 80482], 
'drugName': ['Naproxen', 'Duloxetine', 'Mobic'], 
'condition': ['Gout, Acute', 'ibromyalgia', 'Inflammatory Conditions'], 
'review': ['"like the previous person mention, I&#039;m a strong believer of aleve, it works faster for my gout than the prescription meds I take. No more going to the doctor for refills.....Aleve works!"', 
           '"I have taken Cymbalta for about a year and a half for fibromyalgia pain. It is great\r\nas a pain reducer and an anti-depressant, however, the side effects outweighed \r\nany benefit I got from it. I had trouble with restlessness, being tired constantly,\r\ndizziness, dry mouth, numbness and tingling in my feet, and horrible sweating. I am\r\nbeing weaned off of it now. Went from 60 mg to 30mg and now to 15 mg. I will be\r\noff completely in about a week. The fibro pain is coming back, but I would rather deal with it than the side effects."', 
           '"I have been taking Mobic for over a year with no side effects other than an elevated blood pressure.  I had severe knee and ankle pain which completely went away after taking Mobic.  I attempted to stop the medication however pain returned after a few days."'], 
'rating': [9.0, 3.0, 10.0], 
'date': ['September 2, 2015', 'November 7, 2011', 'June 5, 2013'], 
'usefulCount': [36, 13, 128]}
"""

# 该 case 中验证'Unnamed: 0'这个key是否为患者 ID 的猜想-是否唯一，验证结果是的
for split in drug_dataset.keys():
    print(split)
    assert len(drug_dataset[split]) == len(drug_dataset[split].unique("Unnamed: 0"))
"""
train
test
"""
# dataset工具使用，字段重命名
drug_dataset = drug_dataset.rename_column(
    original_column_name="Unnamed: 0", new_column_name="patient_id"
)
print(drug_dataset)
"""
    train: Dataset({
        features: ['patient_id', 'drugName', 'condition', 'review', 'rating', 'date', 'usefulCount'],
        num_rows: 161297
    })
    test: Dataset({
        features: ['patient_id', 'drugName', 'condition', 'review', 'rating', 'date', 'usefulCount'],
        num_rows: 53766
    })
})
"""


# map操作
def lowercase_condition(example):
    return {"condition": example["condition"].lower()}


# 过滤操作
def filter_nones(x):
    return x["condition"] is not None


drug_dataset = drug_dataset.filter(filter_nones).map(lowercase_condition)
# 验证map和filter是否成功，是的
print(drug_dataset["train"]["condition"][:3])
"""
['left ventricular dysfunction', 'adhd', 'birth control']
"""


# 基于原数据集，创建新的列
# 定义一个函数，计算每条评论的字数
def compute_review_length(example):
    return {"review_length": len(example["review"].split())}


drug_dataset = drug_dataset.map(compute_review_length)
print(drug_dataset["train"][0])
"""
{'patient_id': 206461, 
'drugName': 'Valsartan', 
'condition': 'left ventricular dysfunction', 
'review': '"It has no side effect, I take it in combination of Bystolic 5 Mg and Fish Oil"', 
'rating': 9.0, 
'date': 'May 20, 2012', 
'usefulCount': 27, 
'review_length': 17}

"""
# 根据上一步新生成的 review_length字段排序
print(drug_dataset["train"].sort("review_length")[:3])
"""
{'patient_id': [111469, 13653, 53602], 
'drugName': ['Ledipasvir / sofosbuvir', 'Amphetamine / dextroamphetamine', 'Alesse'], 
'condition': ['hepatitis c', 'adhd', 'birth control'], 
'review': ['"Headache"', '"Great"', '"Awesome"'], 
'rating': [10.0, 10.0, 10.0], 
'date': ['February 3, 2015', 'October 20, 2009', 'November 23, 2015'], 
'usefulCount': [41, 3, 0], 
'review_length': [1, 1, 1]}
"""

# 🙋向数据集添加新列的另一种方法是使用函数 Dataset.add_column() ，在使用它时你可以通过 Python 列表或 NumPy 数组的方式提供数据，
# 在不适合使用 Dataset.map() 情况下可以很方便。

# 使用 Dataset.filter()功能来删除包含少于 30个单词评论
drug_dataset = drug_dataset.filter(lambda x: x["review_length"] > 30)
print(drug_dataset.num_rows)
"""
{'train': 138514, 'test': 46108}
"""

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
print([len(inp) for inp in result["input_ids"]])
"""
# 验证 结果为[128, 45]表明结果变成了两行了，而不是原来的一行
"""

# 删除原数据集的其他行只使用该特征，那么多出来的行不用跟原数据集其他列数据的行数对齐；
tokenized_dataset = drug_dataset.map(
    tokenize_and_split, batched=True, remove_columns=drug_dataset["train"].column_names
)
print(len(tokenized_dataset["train"]), len(drug_dataset["train"]))
"""
204198 138514
# 结果为（204198 138514）形成了新的多了很多行特征，同时未报错
"""


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
"""
DatasetDict({
    train: Dataset({
        features: ['patient_id', 'drugName', 'condition', 'review', 'rating', 'date', 'usefulCount', 'review_length', 'input_ids', 'token_type_ids', 'attention_mask'],
        num_rows: 204198
    })
    test: Dataset({
        features: ['patient_id', 'drugName', 'condition', 'review', 'rating', 'date', 'usefulCount', 'review_length', 'input_ids', 'token_type_ids', 'attention_mask'],
        num_rows: 68023
    })
})
"""

# Datasets 和 DataFrames 的相互转换

# 将数据集转换为 pandas
drug_dataset.set_format("pandas")
# 访问数据集的元素时，我们会得到一个 pandas.DataFrame而不是字典
print(drug_dataset["train"][:3])
"""
   patient_id    drugName  ... usefulCount review_length
0       95260  Guanfacine  ...         192           141
1       92703      Lybrel  ...          17           134
2      138000  Ortho Evra  ...          10            89

[3 rows x 8 columns]
"""

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
print(frequencies.head())
"""
       condition  frequency
0  birth control      27655
1     depression       8023
2           acne       5209
3        anxiety       4991
4           pain       4744
"""

from datasets import Dataset

# 将 pandas 的 dataFrame 格式数据转化为 dataset的字典json格式数据
freq_dataset = Dataset.from_pandas(frequencies)
print(freq_dataset)
"""
Dataset({
    features: ['condition', 'frequency'],
    num_rows: 819
})
"""
# reset_format() 让数据集回到最适合 datasets 库原生操作（如 map、filter、save）的格式，避免 pandas 格式带来的副作用。
drug_dataset.reset_format()

# 创建验证集：训练过程，训练集训练模型参数，验证集调整超参数，测试集最终评估（模型训练时从来没见过来保证测试的有效性）
drug_dataset_clean = drug_dataset["train"].train_test_split(train_size=0.8, seed=42)
# 将默认的“test”部分重命名为“validation”
drug_dataset_clean["validation"] = drug_dataset_clean.pop("test")
# 将“test”部分添加到我们的“DatasetDict”中
drug_dataset_clean["test"] = drug_dataset["test"]
print(drug_dataset_clean)
"""
DatasetDict({
    train: Dataset({
        features: ['patient_id', 'drugName', 'condition', 'review', 'rating', 'date', 'usefulCount', 'review_length'],
        num_rows: 110811
    })
    validation: Dataset({
        features: ['patient_id', 'drugName', 'condition', 'review', 'rating', 'date', 'usefulCount', 'review_length'],
        num_rows: 27703
    })
    test: Dataset({
        features: ['patient_id', 'drugName', 'condition', 'review', 'rating', 'date', 'usefulCount', 'review_length'],
        num_rows: 46108
    })
})
"""

# 保存数据集
# 虽然 Datasets 会缓存每个下载的数据集和对它执行的操作，但有时你会想要将数据集保存到磁盘（比如，以防缓存被删除）

# 1.保存数据为 Apache arrow 格式
# 将数据保存到当前脚本同层的 drug-review 文件夹
drug_dataset_clean.save_to_disk("drug-reviews")

# 保存数据集后，我们可以使用 load_from_disk()功能从磁盘读取数据：
from datasets import load_from_disk

drug_dataset_reloaded = load_from_disk("drug-reviews")
print(drug_dataset_reloaded)
"""
DatasetDict({
    train: Dataset({
        features: ['patient_id', 'drugName', 'condition', 'review', 'rating', 'date', 'usefulCount', 'review_length'],
        num_rows: 110811
    })
    validation: Dataset({
        features: ['patient_id', 'drugName', 'condition', 'review', 'rating', 'date', 'usefulCount', 'review_length'],
        num_rows: 27703
    })
    test: Dataset({
        features: ['patient_id', 'drugName', 'condition', 'review', 'rating', 'date', 'usefulCount', 'review_length'],
        num_rows: 46108
    })
})
"""

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
"""
DatasetDict({
    train: Dataset({
        features: ['patient_id', 'drugName', 'condition', 'review', 'rating', 'date', 'usefulCount', 'review_length'],
        num_rows: 110811
    })
    validation: Dataset({
        features: ['patient_id', 'drugName', 'condition', 'review', 'rating', 'date', 'usefulCount', 'review_length'],
        num_rows: 27703
    })
    test: Dataset({
        features: ['patient_id', 'drugName', 'condition', 'review', 'rating', 'date', 'usefulCount', 'review_length'],
        num_rows: 46108
    })
})
"""

"""
附注
当你使用切片[:3]时，Dataset返回的是列式结构的字典，而不是3个独立的行记录。

# 切片 [:3] 返回的结构：
{
    'patient_id': [1, 2, 3],
    # 每个字段是一个包含3个元素的列表
    'condition': ['a', 'b', 'c'],
    'review': ['...', '...', '...'],
    'review_length': [10, 20, 30]
}

# 而不是 3 个独立的行对象：
{'patient_id': 1, 'condition': 'a',...}  # 第1行
{'patient_id': 2, 'condition': 'b',...}  # 第2行
{'patient_id': 3, 'condition': 'c',...}  # 第3行

这是因为datasets库使用Apache Arrow列式存储，切片操作按列返回数据，效率更高。
如何获取独立的行记录
如果你想逐行查看，可以这样：

sorted_ds = drug_dataset["train"].sort("review_length")
# 方法1：逐个索引（返回单个记录字典）
for i in range(3):
    print(sorted_ds[i])

    # 方法2：遍历（返回单个记录）
for example in sorted_ds.select(range(3)):
    print(example)

这种列式返回的设计是datasets库的正常行为，便于批量处理数据。
"""
```

---

### 3.大数据处理

**Apache Arrow 作为底层数据格式**

Datasets 库的核心是 **Apache Arrow** —— 一种列式内存格式，专门为大数据处理优化。

```
┌─────────────────────────────────────────────────────────────┐
│                    Hugging Face Datasets                      │
├─────────────────────────────────────────────────────────────┤
│                      Apache Arrow                             │
│   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│   │  Column 1   │  │  Column 2   │  │  Column 3   │  ...     │
│   │  (列式存储)  │  │  (列式存储)  │  │  (列式存储)  │          │
│   └─────────────┘  └─────────────┘  └─────────────┘          │
├─────────────────────────────────────────────────────────────┤
│                 Memory Mapping (内存映射)                      │
│                      ↓                                        │
│                 Disk Storage (磁盘存储)                        │
└─────────────────────────────────────────────────────────────┘
```

**Memory Mapping（内存映射）**

**原理：** 数据存储在磁盘上，通过操作系统虚拟内存机制映射到内存地址空间。

```python
# 加载 18GB 的 Wikipedia 数据集，仅占用 50MB RAM
import psutil
from datasets import load_dataset

wiki = load_dataset("wikimedia/wikipedia", "20220301.en", split="train")
# 实际内存占用: ~50 MB（而不是 18 GB！）
```

**工作流程：**

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   磁盘文件    │ ←─→ │  虚拟内存    │ ←─→  │   物理内存     │
│  (Arrow格式) │     │  (OS管理)    │      │  (按需加载)    │
└──────────────┘     └──────────────┘     └──────────────┘

请求数据 → OS 将所需页加载到 RAM → 访问完成 → 页可被回收
```

**Lazy Processing（惰性处理）**

**原理：** `map()`、`filter()` 等操作不立即执行，而是在迭代时按需执行。

```python
ids = ds.to_iterable_dataset()
ids = ids.filter(filter_fn).map(process_fn)

# 上述操作只是定义了处理管道，并未执行
# 只有开始迭代时才真正执行 filter 和 map
for example in ids:
    # 此时才执行 filter → map → 返回结果
    pass
```

- 更多细节参考文件 ./reference/datasets_large_data_principles.md

---

### 4.创建自己的数据集

- 从 GitHub 平台获取评论数据

```Python
# from rich import print
import requests
import os
from dotenv import load_dotenv
from pathlib import Path
import math
import time
import pandas as pd
from tqdm import tqdm

# 加载 .env 文件，override=True 覆盖已有环境变量
load_dotenv(override=True)
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

# 参考 GitHub 官方的 API 文档进行设置 https://docs.github.com/en/rest/issues/issues?apiVersion=2026-03-10
headers = {
    "Accept": "application/vnd.github+json",
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "X-GitHub-Api-Version": "2026-03-10",
}


def fetch_issues(
        owner="huggingface",
        repo="datasets",
        # 要获取的 issue 总数
        num_issues=10_000,
        # GitHub API 速率限制阈值
        rate_limit=5_000,
        # issue 数据保存路径
        issues_path=Path("."),
):
    if not issues_path.is_dir():
        issues_path.mkdir(parents=True, exist_ok=True)

    batch = []
    all_issues = []
    # 每页返回的 issue 的数量
    per_page = 100
    num_pages = math.ceil(num_issues / per_page)
    base_url = "https://api.github.com/repos"

    for page in tqdm(range(num_pages)):
        # 使用 state=all 进行查询来获取 open 和 closed 的 issue
        query = f"issues?page={page}&per_page={per_page}&state=all"
        # 重试机制，最多重试 3 次
        for attempt in range(3):
            try:
                issues = requests.get(
                    f"{base_url}/{owner}/{repo}/{query}",
                    headers=headers,
                    timeout=30,
                )
                issues.raise_for_status()
                batch.extend(issues.json())
                break
            except (requests.exceptions.ChunkedEncodingError,
                    requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout) as e:
                if attempt < 2:
                    print(f"\nRequest failed (attempt {attempt + 1}/3): {e}. Retrying in 5s...")
                    time.sleep(5)
                else:
                    print(f"\nRequest failed after 3 attempts: {e}. Skipping page {page}.")
        # extend 是把列表拆开每个元素都加入，append 是加入一个元素

        if len(batch) > rate_limit and len(all_issues) < num_issues:
            all_issues.extend(batch)
            # 重置 batch
            batch = []
            print(f"Reached GitHub rate limit. Sleeping for one hour ...")
            # 设置了 token 官方的 rate limit 是 5000，可以注释掉这行代码
            # time.sleep(60 * 60 + 1)

    all_issues.extend(batch)
    df = pd.DataFrame.from_records(all_issues)
    df.to_json(f"{issues_path}/{repo}-issues.jsonl", orient="records", lines=True)
    print(f"Downloaded all the issues for {repo}! Dataset stored at {issues_path}/{repo}-issues.jsonl")


if __name__ == "__main__":
    fetch_issues()

```

- 参考 ./src/llm-course/5-fetch-issues-*.py脚本

---

### 5.使用 FAISS 进行语义搜索

```Python
"""
GitHub Issues 数据集处理脚本
功能：从 HuggingFace Hub 加载 GitHub issues 数据，筛选并转换为可用于嵌入检索的文本格式
"""

from datasets import load_dataset

# 从 HuggingFace Hub 加载 lewtun/github-issues 数据集，使用 train（字段）分割
issues_dataset = load_dataset("axelloo/github-issues", split="train")

# 筛选数据集：
# - 只保留 issues（排除 pull requests）- pull_request 字段为 None 表示是 issue
# - 只保留有评论的 issues
issues_dataset = issues_dataset.filter(
    lambda x: (x["pull_request"] is None and len(x["comments"]) > 0)
)

print(issues_dataset)
"""
Dataset({
    features: ['url', 'repository_url', 'labels_url', 'comments_url', 'events_url', 'html_url', 'id', 'node_id', 'number', 'title', 'user', 'labels', 'state', 'locked', 'assignees', 'milestone', 'comments', 'created_at', 'updated_at', 'closed_at', 'author_association', 'type', 'active_lock_reason', 'draft', 'pull_request', 'body', 'closed_by', 'reactions', 'timeline_url', 'performed_via_github_app', 'state_reason', 'sub_issues_summary', 'issue_dependencies_summary', 'pinned_comment'],
    num_rows: 247
})
"""

# 获取所有列名，确定需要保留和删除的列
columns = issues_dataset.column_names
columns_to_keep = ["title", "body", "html_url", "comments"]
# 使用对称差集计算需要删除的列（即不在 columns_to_keep 中的列）
columns_to_remove = set(columns_to_keep).symmetric_difference(columns)
issues_dataset = issues_dataset.remove_columns(columns_to_remove)

# 将数据集转换为 pandas DataFrame 格式以便处理
issues_dataset.set_format("pandas")
df = issues_dataset[:]  # 获取整个数据集

# 打印第一条记录的评论列表（用于调试查看数据结构）
print(df["comments"][0].tolist())
"""
['`datasets` 4.8.4 is out and includes a fix :)']
"""

# 将 comments 列"展开"（explode）：
# 原来每行可能有多条评论（列表形式），展开后每条评论变成单独一行
# ignore_index=True 表示重置索引
comments_df = df.explode("comments", ignore_index=True)
# 显示前4行（用于调试）
print(comments_df.head(4))
"""
                                            html_url  ...                                               body
0  https://github.com/huggingface/datasets/issues...  ...  ### Describe the bug\n\nFor PyTorch 2.11 + tor...
1  https://github.com/huggingface/datasets/issues...  ...  The `.batch()` method currently assumes the in...
2  https://github.com/huggingface/datasets/issues...  ...  The `.batch()` method currently assumes the in...
3  https://github.com/huggingface/datasets/issues...  ...  ### Describe the bug\n\nFor PyTorch 2.11 + tor...

[4 rows x 4 columns]
"""

from datasets import Dataset

# 将 pandas DataFrame 转换回 HuggingFace Dataset 格式
comments_dataset = Dataset.from_pandas(comments_df)
print(comments_dataset)
"""
Dataset({
    features: ['html_url', 'title', 'comments', 'body'],
    num_rows: 851
})
"""
# 计算每条评论的词数，并筛选出词数大于15的评论
# 注意：这里假设 comments 字段是字符串类型
comments_dataset = comments_dataset.map(
    lambda x: {"comment_length": len(x["comments"].split())}
).filter(lambda x: x["comment_length"] > 15)


def concatenate_text(examples):
    """
    将 title、body 和 comments 拼接成一个完整的文本字段
    用于后续的嵌入检索或文本处理
    """
    title = examples["title"] or ""
    body = examples["body"] or ""
    comments = examples["comments"] or ""
    return {
        "text": title + " \n " + body + " \n " + comments
    }


# 对数据集应用文本拼接函数
comments_dataset = comments_dataset.map(concatenate_text)
print(comments_dataset)
print(comments_dataset[0])
"""
Dataset({
    features: ['html_url', 'title', 'comments', 'body', 'comment_length', 'text'],
    num_rows: 616
})
{'html_url': 'https://github.com/huggingface/datasets/issues/8075', 
'title': '`.batch()` error on formatted datasets', 
'comments': 'Hi ! Good catch :) Since table-formatted iterable datasets have an Arrow path (i.e. they have `.iter_arrow()`) I guess a simple fix would be to make `.batch()` make the dataset arrow-formatted and use a `batch_arrow_fn` in that case (instead of the current `_batch_fn` that expects dictionaries). It would also be better performance-wise. How does that sound ?', 
'body': 'The `.batch()` method currently assumes the input (batch) is always a dictionary, which causes errors when it isn\'t. This can happen with formatted datasets, since formats like `"pyarrow"`, `"pandas"` (only affects `IterableDataset`), and `"polars"` return tables/dataframes instead of dictionaries.\n\nFor example:\n```python\nfrom datasets import IterableDataset, Dataset\nlist(IterableDataset.from_dict({"a": [1, 2, 3, 4]}).with_format("pyarrow").batch(2))\n# AttributeError: \'pyarrow.lib.Table\' object has no attribute \'items\'\n```\n\nIdeally, the result should be the same whether the format is applied before or after batching, i.e., the following should hold for all the format types:\n```python\nassert list(IterableDataset.from_dict({"a": [1, 2, 3, 4]}).with_format(format_type).batch(2)) == list(IterableDataset.from_dict({"a": [1, 2, 3, 4]}).batch(2).with_format(format_type))\nassert list(Dataset.from_dict({"a": [1, 2, 3, 4]}).with_format(format_type).batch(2)) == list(Dataset.from_dict({"a": [1, 2, 3, 4]}).batch(2).with_format(format_type))\n```', 
'comment_length': 57, 
'text': '`.batch()` error on formatted datasets \n The `.batch()` method currently assumes the input (batch) is always a dictionary, which causes errors when it isn\'t. This can happen with formatted datasets, since formats like `"pyarrow"`, `"pandas"` (only affects `IterableDataset`), and `"polars"` return tables/dataframes instead of dictionaries.\n\nFor example:\n```python\nfrom datasets import IterableDataset, Dataset\nlist(IterableDataset.from_dict({"a": [1, 2, 3, 4]}).with_format("pyarrow").batch(2))\n# AttributeError: \'pyarrow.lib.Table\' object has no attribute \'items\'\n```\n\nIdeally, the result should be the same whether the format is applied before or after batching, i.e., the following should hold for all the format types:\n```python\nassert list(IterableDataset.from_dict({"a": [1, 2, 3, 4]}).with_format(format_type).batch(2)) == list(IterableDataset.from_dict({"a": [1, 2, 3, 4]}).batch(2).with_format(format_type))\nassert list(Dataset.from_dict({"a": [1, 2, 3, 4]}).with_format(format_type).batch(2)) == list(Dataset.from_dict({"a": [1, 2, 3, 4]}).batch(2).with_format(format_type))\n``` \n Hi ! Good catch :) Since table-formatted iterable datasets have an Arrow path (i.e. they have `.iter_arrow()`) I guess a simple fix would be to make `.batch()` make the dataset arrow-formatted and use a `batch_arrow_fn` in that case (instead of the current `_batch_fn` that expects dictionaries). It would also be better performance-wise. How does that sound ?'}

"""

# 加载 sentence-transformers 模型，用于生成文本嵌入向量
# multi-qa-mpnet-base-dot-v1 是一个专门针对问答检索任务优化的模型
from transformers import AutoTokenizer, AutoModel

model_ckpt = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)  # 加载分词器
model = AutoModel.from_pretrained(model_ckpt)  # 加载预训练模型


# 实现 CLS 池化，用于从 BERT 类模型的输出中提取句子级别的嵌入向量。
def cls_pooling(model_output):
    """
    1. model_output.last_hidden_state — 模型的隐藏层输出，形状为 (batch_size, seq_length, hidden_dim)。例如输入 2 个句子、每个 10 个 token，输出形状就是 (2, 10, 768)。
    2. [:, 0] — 取所有样本的第 0 个位置（即 [CLS] token）。结果是 (batch_size, hidden_dim)。
    为什么用 [CLS]？
    BERT 类模型在输入序列开头会添加一个特殊的 [CLS] token。经过 Transformer 的自注意力机制，[CLS] 会聚合整个序列的信息，因此它的嵌入向量常被用作整段文本的语义表示。

    举例：
    假设输入文本 "How can I load a dataset?" 经 tokenizer 后变成：
    [CLS] How can I load a dataset [SEP]
     0   1   2  3  4    5       6     7
    last_hidden_state[:, 0] 就提取位置 0 的向量，代表整个问题的语义，用于后续的相似度检索。
    """
    return model_output.last_hidden_state[:, 0]


def get_embeddings(text_list):
    """
    将文本列表转换为嵌入向量
    参数：text_list - 文本字符串列表
    返回：嵌入向量矩阵
    """
    # 使用 tokenizer 对文本进行编码
    # padding=True: 对短文本进行填充，使所有序列长度一致
    # truncation=True: 对超长文本进行截断
    # return_tensors="pt": 返回 PyTorch tensor 格式
    encoded_input = tokenizer(
        text_list, padding=True, truncation=True, return_tensors="pt"
    )
    # 将编码输入转换为字典格式（去除 batch 索引）
    encoded_input = {k: v for k, v in encoded_input.items()}
    # 将编码输入传入模型，获取输出
    model_output = model(**encoded_input)
    # 使用 CLS 池化提取嵌入向量
    return cls_pooling(model_output)


# 测试：对第一条文本生成嵌入向量，打印向量维度
embedding = get_embeddings(comments_dataset["text"][0])
print(embedding.shape)
"""
torch.Size([1, 768])
"""

# 对整个数据集计算嵌入向量
# 每条文本生成一个 768 维的嵌入向量（mpnet-base 模型的输出维度）
# .detach().cpu().numpy()[0] 用于将 PyTorch tensor 转换为 numpy 数组
embeddings_dataset = comments_dataset.map(
    lambda x: {"embeddings": get_embeddings(x["text"]).detach().cpu().numpy()[0]}
)
print(embeddings_dataset)
print(embeddings_dataset[0])
"""
Dataset({
    features: ['html_url', 'title', 'comments', 'body', 'comment_length', 'text', 'embeddings'],
    num_rows: 616
})
{'html_url': 'https://github.com/huggingface/datasets/issues/8075', 
'title': '`.batch()` error on formatted datasets', 
'comments': 'Hi ! Good catch :) Since table-formatted iterable datasets have an Arrow path (i.e. they have `.iter_arrow()`) I guess a simple fix would be to make `.batch()` make the dataset arrow-formatted and use a `batch_arrow_fn` in that case (instead of the current `_batch_fn` that expects dictionaries). It would also be better performance-wise. How does that sound ?', 
'body': 'The `.batch()` method currently assumes the input (batch) is always a dictionary, which causes errors when it isn\'t. This can happen with formatted datasets, since formats like `"pyarrow"`, `"pandas"` (only affects `IterableDataset`), and `"polars"` return tables/dataframes instead of dictionaries.\n\nFor example:\n```python\nfrom datasets import IterableDataset, Dataset\nlist(IterableDataset.from_dict({"a": [1, 2, 3, 4]}).with_format("pyarrow").batch(2))\n# AttributeError: \'pyarrow.lib.Table\' object has no attribute \'items\'\n```\n\nIdeally, the result should be the same whether the format is applied before or after batching, i.e., the following should hold for all the format types:\n```python\nassert list(IterableDataset.from_dict({"a": [1, 2, 3, 4]}).with_format(format_type).batch(2)) == list(IterableDataset.from_dict({"a": [1, 2, 3, 4]}).batch(2).with_format(format_type))\nassert list(Dataset.from_dict({"a": [1, 2, 3, 4]}).with_format(format_type).batch(2)) == list(Dataset.from_dict({"a": [1, 2, 3, 4]}).batch(2).with_format(format_type))\n```', 
'comment_length': 57, 
'text': '`.batch()` error on formatted datasets \n The `.batch()` method currently assumes the input (batch) is always a dictionary, which causes errors when it isn\'t. This can happen with formatted datasets, since formats like `"pyarrow"`, `"pandas"` (only affects `IterableDataset`), and `"polars"` return tables/dataframes instead of dictionaries.\n\nFor example:\n```python\nfrom datasets import IterableDataset, Dataset\nlist(IterableDataset.from_dict({"a": [1, 2, 3, 4]}).with_format("pyarrow").batch(2))\n# AttributeError: \'pyarrow.lib.Table\' object has no attribute \'items\'\n```\n\nIdeally, the result should be the same whether the format is applied before or after batching, i.e., the following should hold for all the format types:\n```python\nassert list(IterableDataset.from_dict({"a": [1, 2, 3, 4]}).with_format(format_type).batch(2)) == list(IterableDataset.from_dict({"a": [1, 2, 3, 4]}).batch(2).with_format(format_type))\nassert list(Dataset.from_dict({"a": [1, 2, 3, 4]}).with_format(format_type).batch(2)) == list(Dataset.from_dict({"a": [1, 2, 3, 4]}).batch(2).with_format(format_type))\n``` \n Hi ! Good catch :) Since table-formatted iterable datasets have an Arrow path (i.e. they have `.iter_arrow()`) I guess a simple fix would be to make `.batch()` make the dataset arrow-formatted and use a `batch_arrow_fn` in that case (instead of the current `_batch_fn` that expects dictionaries). It would also be better performance-wise. How does that sound ?',
'embeddings': [-0.40911683440208435, -0.1059914156794548, -0.05522923171520233, -0.0641660988330841, 0.11291451752185822, 0.08336920291185379, 0.463390976190567, 0.3956606388092041, -0.426876962184906, -0.07155115902423859, -0.06572770327329636, 0.47573480010032654, -0.19835688173770905, 0.11820483952760696, -0.1148185282945633, -0.1328001320362091, 0.08152212202548981, 0.15055111050605774, -0.07466396689414978, 0.03878754377365112, -0.19033733010292053, 0.08133885264396667, -0.4599868655204773, 0.22942528128623962, -0.21314433217048645, -0.2881758511066437, -0.12671679258346558, -0.1859939992427826, 0.0871286541223526, -0.7704533934593201, 0.3828081488609314, 0.12954838573932648, 0.3740334212779999, 0.6622698903083801, -0.00011272120173089206, -0.046194158494472504, 0.1864146739244461, -0.05348096787929535, -0.4844150245189667, -0.06087496876716614, -0.23521947860717773, -0.18493197858333588, 0.2199983447790146, -0.14592893421649933, -0.05526606738567352, -0.5972678661346436, -0.24288831651210785, -0.1606457531452179, 0.0378539115190506, 0.12746672332286835, 0.2100406140089035, 0.32998132705688477, -0.06503552198410034, 0.026027729734778404, 0.15730485320091248, 0.21658283472061157, -0.025716383010149002, 0.4210568368434906, 0.3901309072971344, -0.09501868486404419, 0.1241636574268341, 0.18036866188049316, -0.3338664770126343, 0.20260807871818542, -0.06366817653179169, 0.05272982642054558, 0.013951964676380157, -0.1316254734992981, -0.07426539063453674, 0.2758042812347412, 0.4868970513343811, -0.403902530670166, -0.5257353186607361, -0.3883194923400879, -0.030763130635023117, -0.3132793605327606, -0.20147162675857544, 0.17869015038013458, -0.051441874355077744, 0.014630984514951706, -0.07274423539638519, 0.189640611410141, -0.19236978888511658, -0.019875280559062958, -0.09078405052423477, 0.3942931592464447, -0.0760517418384552, 0.17314490675926208, -0.08642107248306274, -0.08095592260360718, 0.19773051142692566, -0.17343057692050934, -0.3543521463871002, 0.13075105845928192, -0.21657197177410126, -0.11495479941368103, 0.0751824676990509, -0.09402791410684586, 0.11524442583322525, 0.17278125882148743, -0.08004692196846008, -0.1250341832637787, 0.08936438709497452, 0.010875500738620758, 0.3309323489665985, 0.30324333906173706, 0.24935847520828247, 0.26325905323028564, 0.14833098649978638, 0.12230972945690155, -0.036081112921237946, -0.01875821128487587, 0.15182167291641235, -0.41670936346054077, 0.3561733365058899, 0.05621141567826271, 0.33688586950302124, -0.10796955227851868, -0.2942178547382355, -0.08334088325500488, -0.2224130928516388, -0.25148722529411316, 0.027613844722509384, 0.21299588680267334, 0.029774893075227737, 0.1607840657234192, -0.10163012146949768, 0.310607373714447, 0.0777539610862732, 0.039710793644189835, -0.07049227505922318, 0.06991730630397797, -0.22135759890079498, -0.23978058993816376, 0.13247384130954742, -0.3666727542877197, 0.021909480914473534, 0.14092865586280823, -0.030365586280822754, 0.08047424256801605, -0.08656249940395355, -0.2426605522632599, 0.44843384623527527, 0.10931888967752457, 0.0181749127805233, 0.2889542877674103, 0.15471233427524567, -0.05855479836463928, -0.1528550684452057, 0.15041112899780273, -0.12792238593101501, -0.1970234364271164, -0.12976720929145813, 0.1871112585067749, -0.19612595438957214, -0.12540996074676514, -0.260233610868454, 0.046286724507808685, 0.07848338037729263, -0.20184342563152313, 0.05439405515789986, -0.3447802662849426, 0.1798258125782013, -0.38337963819503784, 0.2408025562763214, 0.32196617126464844, -0.6573533415794373, 0.14890262484550476, 0.3685605823993683, -0.10237351059913635, 0.3370114862918854, 0.2689688801765442, -0.12471570819616318, 0.516861081123352, -0.28704091906547546, 0.012729249894618988, -0.08588264137506485, -0.20366843044757843, -0.2628403306007385, 0.2163202464580536, 0.208762988448143, 0.37994441390037537, -0.0020478665828704834, -0.04133594408631325, 0.16595767438411713, -0.2725694477558136, 0.11481555551290512, 0.42455917596817017, -0.20413823425769806, 0.0962647870182991, -0.08622147142887115, 0.04533044993877411, 0.2136079967021942, -0.04319401830434799, 0.077810138463974, 0.10014522075653076, -0.08352603763341904, -0.2792647182941437, 0.20784991979599, -0.32493242621421814, -0.04465719312429428, -0.24377663433551788, 0.36748170852661133, 0.09334681183099747, 0.16207553446292877, -0.1633572280406952, -0.3386428952217102, 0.14716538786888123, -0.12948977947235107, -0.013376109302043915, -0.3402308225631714, -0.26823925971984863, -0.04463218152523041, 0.2982430160045624, -0.3973882496356964, -0.17660671472549438, 0.15491606295108795, 0.107688307762146, 0.27986645698547363, 0.05588323995471001, -0.22562530636787415, -0.15298819541931152, -0.02753128856420517, -0.00020723987836390734, 0.07146941125392914, -0.05063937231898308, -0.02691287361085415, -0.2508769631385803, -0.10000093281269073, 0.28035426139831543, 0.05793580412864685, -0.15736331045627594, -0.10361672192811966, 0.44240665435791016, -0.163614422082901, -0.08874295651912689, -0.12644696235656738, 0.07587598264217377, -0.047256603837013245, 0.1202467828989029, -0.19284482300281525, -0.1092534065246582, 0.12139110267162323, -0.03322063758969307, 0.07695548236370087, 0.4232720136642456, -0.15114551782608032, 0.38411349058151245, -0.04406912252306938, 0.08840122818946838, 0.37544116377830505, 0.2227180302143097, -0.18514612317085266, -0.10099558532238007, -0.08412079513072968, -0.12406229972839355, 0.13021264970302582, -0.22561554610729218, -0.3341463506221771, -0.062124960124492645, 0.2737691402435303, -0.03864460811018944, 0.27069923281669617, 0.003980979323387146, -0.27244511246681213, 0.15827885270118713, 0.11002293229103088, 0.05412546172738075, 0.2948543429374695, 0.19788235425949097, -0.05751212313771248, -0.07533994317054749, -0.08208096772432327, 0.04687193036079407, 0.2616609036922455, 0.23143811523914337, 0.22471754252910614, 0.029017498716711998, 0.04852786287665367, 0.19244171679019928, -0.06182397902011871, -0.43683600425720215, -0.17832481861114502, 0.24929118156433105, -0.2860504686832428, 0.13886350393295288, -0.2293650060892105, -0.17995941638946533, -0.12498196959495544, -0.5852276086807251, 0.16042499244213104, -0.2572799623012543, -0.011480819433927536, 0.0810718834400177, -0.22216805815696716, 0.2791282832622528, 0.1503353714942932, 0.06761865317821503, 0.08542414754629135, -0.383465439081192, -0.12528151273727417, -0.3304895758628845, -0.2332397699356079, 0.09674040973186493, 0.2671474814414978, -0.283251017332077, 0.48682379722595215, -0.13610060513019562, 0.0613897368311882, -0.3297157287597656, -0.08869849890470505, 0.07270674407482147, -0.206918403506279, 0.24724026024341583, 0.39719176292419434, 0.2686823904514313, 0.04914035648107529, -0.1559620350599289, 0.3782169222831726, -0.1504448801279068, 0.05700866878032684, 0.23151680827140808, -0.1650499403476715, -0.2562617361545563, 0.08352406322956085, -0.15325286984443665, -0.08032234013080597, -0.19916220009326935, 0.15514442324638367, -0.0917975902557373, 0.005093451589345932, 0.38040855526924133, 0.18336184322834015, 0.19829373061656952, -0.13087016344070435, 0.0885978490114212, 0.13638943433761597, 0.05680396780371666, 0.25793901085853577, -0.04834351688623428, -0.16644056141376495, -0.04647394269704819, -0.13203716278076172, 0.06557131558656693, 0.25526362657546997, -0.18135946989059448, -0.09563696384429932, -0.2632159888744354, 0.1960792988538742, 0.09520275890827179, 0.34151262044906616, 0.21047323942184448, 0.2515539526939392, -0.10886288434267044, -0.026361029595136642, -0.04641188308596611, 0.027521096169948578, -0.24319425225257874, -0.11008290201425552, 0.23617833852767944, 0.44715172052383423, 0.14846362173557281, 0.5902130603790283, 0.09584331512451172, -0.2325410097837448, 0.3301248550415039, -0.12104686349630356, 0.30287304520606995, -0.12511661648750305, -0.3857043981552124, -0.11241558194160461, -0.18732932209968567, -0.021618202328681946, 0.23665426671504974, -0.25005394220352173, -0.006772294640541077, -0.16393893957138062, 0.019488636404275894, -0.25845757126808167, -0.21763239800930023, 0.3283659517765045, -0.539763331413269, 0.3856008052825928, -0.04820350557565689, 0.09250540286302567, -0.008074982091784477, -0.01799626648426056, 0.07920245826244354, -0.08503790199756622, 0.11700260639190674, -0.10172279179096222, -0.359770268201828, -0.6891679763793945, -0.2801053524017334, 0.3845319151878357, 0.21025407314300537, 0.7424201369285583, 0.24298347532749176, -0.18816107511520386, -0.04458451271057129, 0.0608866922557354, 0.5138818025588989, -0.23532922565937042, -0.15241830050945282, 0.01885053515434265, -0.12801793217658997, -0.3994271755218506, -0.35004258155822754, -0.40419116616249084, 0.3465169072151184, -0.059657592326402664, 0.06346161663532257, -0.19901838898658752, -0.08881203830242157, 0.22778655588626862, 0.281804621219635, 0.0809573233127594, -0.09592732787132263, -0.20971786975860596, -0.2745293974876404, -0.32740458846092224, 0.27784398198127747, 0.12812009453773499, -0.016552885994315147, -0.11296351253986359, -0.02266361005604267, -0.41199278831481934, 0.10610739141702652, 0.06457596272230148, 0.26785120368003845, 0.3020259439945221, -0.011930882930755615, 0.07679381966590881, -0.15416894853115082, 0.20689235627651215, 0.5043161511421204, 0.20818409323692322, -0.15782110393047333, -0.26468315720558167, -0.22595864534378052, -0.08119356632232666, 0.32503941655158997, 0.19970466196537018, -0.25325292348861694, 0.07583749294281006, -0.3410640060901642, 0.1392166018486023, 0.00048539694398641586, -0.2907891571521759, 0.4336834251880646, 0.07882356643676758, -0.7235928177833557, -0.48073774576187134, 0.3294735252857208, 0.22584302723407745, 0.029900528490543365, 0.40439629554748535, 0.24470919370651245, -0.21497978270053864, 0.6425031423568726, -0.07863923162221909, 0.5631107687950134, 0.09813421219587326, 0.2064136415719986, 0.3651917576789856, 0.12064646184444427, 0.3637576401233673, 0.41517174243927, 0.14076638221740723, -0.2682613730430603, -0.09184890985488892, 0.015930386260151863, -0.2523359954357147, 0.09833583235740662, 0.31512290239334106, -0.23800957202911377, 0.37857264280319214, -0.3420734405517578, 0.1399787962436676, 0.0829126238822937, 0.1436111330986023, 0.11594036221504211, -0.26507246494293213, -0.3994799256324768, 0.10134480893611908, -0.12151959538459778, -0.05867641791701317, -0.27454686164855957, 0.0761415883898735, 0.05707104131579399, -0.22509099543094635, -0.3649146556854248, 0.001025831326842308, -0.1748826801776886, 0.1334579885005951, 0.09135628491640091, -0.514306902885437, 0.12675829231739044, 0.1911579817533493, 0.3580009937286377, 0.09742454439401627, 0.04665956646203995, 0.012406479567289352, 0.31951433420181274, 0.14495569467544556, 0.12863627076148987, -0.18850497901439667, 0.6046899557113647, 0.11281367391347885, -0.042143769562244415, 0.1924719214439392, -0.1313241720199585, -0.1009042039513588, 0.021296605467796326, 0.27963709831237793, 0.22113166749477386, -0.30561986565589905, -0.35497936606407166, -0.10924467444419861, -0.03825227543711662, -0.1593240350484848, 0.1160898208618164, 0.029563114047050476, -0.1382851004600525, 0.720177173614502, -0.17008085548877716, -0.26264744997024536, -0.08468882739543915, -0.07808510214090347, 0.04201003909111023, -0.08852067589759827, 0.5980509519577026, 0.27939939498901367, -0.21104992926120758, -0.10425931215286255, -0.05330675467848778, 0.09343262761831284, -0.6413567662239075, 0.31893405318260193, 0.10796889662742615, 0.30186212062835693, 0.07080690562725067, 0.39162594079971313, 0.14745637774467468, 0.07631614804267883, -0.03576178848743439, -0.19966675341129303, 0.08776767551898956, 0.11780142039060593, 0.17522522807121277, 0.3697345554828644, 0.3987768888473511, 0.25408652424812317, 0.13396215438842773, 0.028880085796117783, -0.3144587278366089, -0.20075052976608276, -0.006682965904474258, 0.30842193961143494, -0.21299535036087036, 0.44495442509651184, 0.14664675295352936, -0.014975541271269321, 0.14051483571529388, 0.1595325469970703, 0.2172217071056366, -0.3041934669017792, -0.06369626522064209, 0.0988791212439537, 0.035018399357795715, -0.1595446914434433, 0.1459624320268631, -0.0819488987326622, -0.13202504813671112, -0.35047537088394165, 0.026865635067224503, 0.1756671965122223, -0.21726830303668976, 0.08486589789390564, 0.37898293137550354, 0.043331459164619446, 0.08295033872127533, 0.18435825407505035, -0.33097362518310547, -0.1360924392938614, 0.1287888139486313, 0.24376888573169708, -0.038756102323532104, -0.10929456353187561, -0.07918940484523773, 0.06911759078502655, 0.16126997768878937, 0.2346152365207672, 0.00023084133863449097, -0.4827355742454529, -0.08896122872829437, 0.0838177353143692, 0.5207581520080566, 0.24799393117427826, -0.09680788964033127, -0.14331044256687164, 0.3929222524166107, 0.20763926208019257, -0.31200018525123596, 0.03423764929175377, -0.07028022408485413, 0.024620456621050835, 0.16865967214107513, 0.3431810140609741, 0.044558484107255936, 0.0626802146434784, 0.016043446958065033, 0.11282550543546677, 0.09054241329431534, -0.24976034462451935, 0.17523708939552307, 0.2036016583442688, 0.21697579324245453, 0.0699094757437706, 0.18049472570419312, 0.07898426055908203, 0.17690172791481018, 0.09728431701660156, 0.14528775215148926, 0.3262118995189667, 0.347344309091568, -0.23631474375724792, 0.07794038206338882, -0.411718487739563, -0.1310107409954071, -0.17048439383506775, 0.12594416737556458, 0.1122385710477829, 0.22222357988357544, 0.29544442892074585, -0.30192598700523376, -0.23801419138908386, 0.12048415094614029, -0.03598905727267265, -0.14244195818901062, 0.12832984328269958, -0.2816077768802643, -0.05525832623243332, 0.2715890109539032, -0.33285707235336304, -0.14328713715076447, -0.14527396857738495, -0.025154195725917816, 0.12122443318367004, 0.06968418508768082, -0.5451204776763916, 0.15391460061073303, 0.019781701266765594, 0.2689996063709259, -0.20365244150161743, 0.04772813618183136, 0.11656568944454193, -0.055856406688690186, 0.05398034676909447, 0.575695812702179, 0.700167715549469, 0.570318877696991, -0.09843911230564117, 0.005459723062813282, -0.397266685962677, -0.06752362102270126, 0.02717066928744316, 0.10839501023292542, -0.16304023563861847, -0.06871722638607025, 0.4603589177131653, 0.27455779910087585, -0.0994911640882492, 0.07754279673099518, 0.10224619507789612, -0.0801527351140976, -0.2041221410036087, 0.14548420906066895, 0.013964544981718063, -0.23537930846214294, -0.12350283563137054, -0.04237159341573715, -0.2765774726867676, -0.01633428782224655, 0.25566378235816956, -0.10940533876419067, 0.011422760784626007, -0.08704996854066849, 0.10606342554092407, 0.16118821501731873, 0.31619831919670105, 0.49550575017929077, 0.40905895829200745, -0.31004559993743896, -0.06800929456949234, -0.43101152777671814, -0.14271879196166992, -0.214900404214859, -0.03405456990003586, -0.01519308052957058, 0.19123145937919617, -0.03327411413192749, 0.1407856047153473, 0.3787408471107483, 0.16159646213054657, -0.2973161041736603, 0.5314661264419556, -0.2696308195590973, -0.20306168496608734, -0.3404104709625244, -0.012163301929831505, -0.0920218676328659, -0.33353039622306824, 0.0005804076790809631, -0.1205182820558548, 0.10839111357927322, -0.13333217799663544, -0.09766766428947449, 0.4252810478210449, -0.06260792911052704, 0.35336706042289734, 0.16644854843616486, 0.3122965693473816, -0.004311911761760712, 0.04353625327348709, -0.2503775358200073, -0.13272708654403687, -0.2814517617225647, 0.2846840023994446, 0.03165721520781517, 0.26063138246536255, 0.0003032190725207329, -0.20103105902671814, -0.1776227205991745, 0.03905129060149193, -0.028021685779094696, 0.18122707307338715, -0.3175504505634308, 0.20301635563373566, -0.04583369567990303, -0.05749526619911194, -0.06392946094274521, 0.18043649196624756, -0.03480295091867447, 0.34867724776268005, -0.43113696575164795, -0.4612301290035248, 0.36048436164855957, -0.647658109664917, -0.46051886677742004, 0.16082988679409027, 0.06617555022239685, -0.11015217751264572, -0.1059102788567543, 0.046320609748363495, -0.03652321174740791, 0.28041356801986694, 0.04805843532085419, -0.23012666404247284, -0.022182095795869827, -0.07016190886497498, 0.2760496437549591, -0.0694093406200409, 0.32596123218536377, 0.011947833001613617, -0.27900826930999756, -0.1365758329629898, -0.28092867136001587]}
"""

# 使用 FAISS 构建 embeddings 列的向量索引
# FAISS 是 Facebook 开发的高效向量相似度搜索库，支持大规模向量快速检索
embeddings_dataset.add_faiss_index(column="embeddings")

# 保存数据集和 FAISS 索引
embeddings_dataset.save_to_disk("embeddings_dataset")
embeddings_dataset.get_index("embeddings").save("embeddings_index.faiss")
print("数据集和索引已保存")
"""
下次加载时使用：                                                                                                                                                                   
                                                                                                                                                                                     
  from datasets import load_from_disk                                                                                                                                                
                                                                                                                                                                                     
  embeddings_dataset = load_from_disk("embeddings_dataset")                                                                                                                          
  embeddings_dataset.load_faiss_index("embeddings", "embeddings_index.faiss") 
"""

# 定义查询问题
question = "How can I load a dataset offline?"
# 将查询问题转换为嵌入向量
question_embedding = get_embeddings([question]).cpu().detach().numpy()
print(question_embedding.shape)
"""
(1, 768)
"""

# 使用 FAISS 进行相似度搜索，找出与查询问题最相似的 5 条记录
# scores: 相似度分数数组
# samples: 匹配的样本数据（字典格式）
scores, samples = embeddings_dataset.get_nearest_examples(
    "embeddings", question_embedding, k=5
)

import pandas as pd

# 将搜索结果转换为 pandas DataFrame，便于排序和展示
samples_df = pd.DataFrame.from_dict(samples)
samples_df["scores"] = scores  # 添加相似度分数列
# 按相似度分数降序排序（分数越高表示越相似）
samples_df.sort_values("scores", ascending=False, inplace=True)

# 遍历并打印搜索结果
for _, row in samples_df.iterrows():
    print(f"COMMENT: {row.comments}")  # 评论内容
    print(f"SCORE: {row.scores}")  # 相似度分数
    print(f"TITLE: {row.title}")  # issue 标题
    print(f"URL: {row.html_url}")  # issue 链接
    print("=" * 50)  # 分隔线
    print()

"""
截取输出
==================================================

COMMENT: I can browse the google drive through google chrome. It's weird. I can download the dataset through google drive manually.
SCORE: 40.68833923339844
TITLE: (Load dataset failure) ConnectionError: Couldn’t reach https://raw.githubusercontent.com/huggingface/datasets/1.1.2/datasets/cnn_dailymail/cnn_dailymail.py
URL: https://github.com/huggingface/datasets/issues/759
==================================================

COMMENT: Hi ! `datasets` currently supports reading local files or files over HTTP. We may add support for other filesystems (cloud storages, hdfs...) at one point though :)
SCORE: 38.603851318359375
TITLE: Does datasets support load text from HDFS?
URL: https://github.com/huggingface/datasets/issues/3490
==================================================

这个模型使用点积相似度，分数越高表示越相似，但没有固定上限。                                                                                                                       
                                                                                                                                                                                     
  判断方法：                                                                                                                                                                         
                                                            
  1. 看分数分布 — 检索返回的 5 条结果中，如果最高分是 31，第二名是 25，差距明显，那 31 分算是比较好的匹配。                                                                          
  2. 看阈值经验 — 对于这个模型（multi-qa-mpnet-base-dot-v1），一般：
    - > 60-70：非常相关，高质量匹配                                                                                                                                                  
    - 30-50：中等相关，有一定语义相似                                                                                                                                                
    - < 20：相关性较弱                                                                                                                                                               
                                                                                                                                                                                     
  31 分属于中等水平，表示查询和结果有一定语义关联，但不是高度匹配。                                                                                                                  
                                                                                                                                                                                     
  如果想提高匹配精度，可以：                                                                                                                                                         
  - 增加检索数量 k=10 或更多，看分数分布                    
  - 设置阈值过滤，如只保留 scores > 40 的结果
"""
```

- 参考 ./src/llm-course/embedding-then-faiss-search.py脚本

---

#### 什么是 Embedding？

Embedding 是将文本转换为**高维向量**（数值数组）的过程，使得语义相似的文本在向量空间中距离更近。

```
"How to load dataset" → [0.12, -0.34, 0.56, ..., 0.78]  (768维向量)
"I want to download data" → [0.15, -0.30, 0.52, ..., 0.75]  (语义相似，向量相近)
"The weather is nice" → [0.89, 0.12, -0.45, ..., -0.23]  (语义不同，向量距离远)
```

**整体流程：**

```
原始文本 → Tokenizer编码 → 模型处理 → 提取嵌入向量 → FAISS索引 → 语义搜索
```

**关键步骤解析：**

1. **模型选择**：`sentence-transformers/multi-qa-mpnet-base-dot-v1` 是专门为问答检索优化的模型，能生成语义密集的嵌入向量。

2. **CLS 池化原理**：

```
输入文本: "How can I load a dataset offline?"
         ↓ Tokenizer
Token序列: [CLS] How can I load a dataset offline [SEP]
           ↓ 模型处理（BERT类模型）
Hidden States: 每个token都有一个768维向量

[CLS]位置的向量 → 作为整个句子的语义表示（index 0）
```

BERT 类模型在训练时，`[CLS]` token 被设计为聚合整个序列的语义信息，它作为整个句子的语义表示。

3. **FAISS 索引作用**：

```
┌─────────────────────────────────────────┐
│            FAISS 索引结构                │
├─────────────────────────────────────────┤
│  向量1: [0.12, -0.34, ...]              │
│  向量2: [0.15, -0.30, ...]              │
│  向量3: [0.89, 0.12, ...]               │
│  ...                                    │
├─────────────────────────────────────────┤
│  索引方式: 内积/余弦相似度               │
│  支持快速近似搜索（百万级向量毫秒级响应） │
└─────────────────────────────────────────┘
```

4. **相似度计算**：模型使用点积计算相似度，点积越大表示向量越接近、语义越相似。

```python
相似度分数 = query_embedding · document_embedding
```

5. **搜索过程**：

```
查询问题 → embedding → question_embedding
                              ↓
                    FAISS 索引中查找相似向量
                              ↓
                    返回: scores (相似度分数) + samples (匹配文本)
```

**各组件作用对照表：**

| 组件 | 作用 | 输入 | 输出 |
|------|------|------|------|
| Tokenizer | 文本→数字 | "How to..." | [101, 2129, 2000, ...] |
| Model | 数字→语义向量 | token IDs | hidden_states |
| CLS Pooling | 提取句子向量 | hidden_states | **Embedding** (768维向量) |
| FAISS | 向量检索索引 | Embeddings集合 | 快速搜索能力 |

---

#### 为什么是 768 维 Embedding？

768 维是由模型架构决定的：

```
BERT-base 模型架构：
- 12 层 Transformer
- 隐藏层维度 (hidden_size) = 768
- 12 个注意力头
- 约 110M 参数

每个 token 输出的向量维度 = hidden_size = 768
```

**不同模型的 embedding 维度：**

| 模型 | 维度 | 说明 |
|------|------|------|
| BERT-base | 768 | 常用中小模型 |
| BERT-large | 1024 | 更大模型，语义更丰富 |
| GPT-2 small | 768 | 与 BERT-base 相同 |
| GPT-3 | 12288 | 维度大幅增加 |
| sentence-transformers | 768 | 基于 BERT-base |

维度越高：
- 能编码更多语义信息
- 但计算成本和存储成本更高
- 可能出现冗余（很多维度接近 0）

---

#### Embedding 与 RAG 的联系

RAG（Retrieval-Augmented Generation，检索增强生成）的核心流程：

```
用户问题 → Embedding → 向量检索 → 获取相关文档 → 拼接到 Prompt → LLM生成回答
```

**Embedding 在 RAG 中的作用：**

```
┌─────────────────────────────────────────────────────────────┐
│                      RAG 架构                                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. 知识库构建阶段                                           │
│     文档 → Embedding → 存入向量数据库 (FAISS/Pinecone等)     │
│                                                             │
│  2. 查询阶段                                                 │
│     问题 → Embedding → 向量相似度搜索 → Top-K 相关文档        │
│                                                             │
│  3. 生成阶段                                                 │
│     问题 + 相关文档 → Prompt → LLM → 生成回答                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**示例流程：**

```
问题: "如何离线加载 HuggingFace 数据集？"
    ↓ Embedding (768维向量)
    ↓ FAISS 搜索
检索到文档: 
    - "使用 load_dataset() 函数加载本地文件..."
    - "支持 CSV、JSON、Parquet 等格式..."
    ↓ 拼接到 Prompt
Prompt: 
    "问题：如何离线加载 HuggingFace 数据集？
     参考资料：
     - 使用 load_dataset() 函数加载本地文件...
     - 支持 CSV、JSON、Parquet 等格式...
     
     请基于参考资料回答问题："
    ↓ LLM 生成
回答: "可以使用 load_dataset() 函数加载本地文件..."
```

**为什么要用 Embedding 而不是关键词搜索？**

| 对比项 | 关键词搜索 | Embedding 语义搜索 |
|--------|-----------|-------------------|
| "如何下载模型" | 匹配不到 "离线加载" | 能匹配（语义相似） |
| "报错了怎么办" | 匹配不到具体错误类型 | 能找到相关错误解决方案 |
| 精确度 | 精确匹配 | 语义理解 |
| 适用场景 | 明确关键词 | 自然语言问答 |

---

#### Agentic RAG（Agent 风格 RAG）

传统 RAG 是先检索再发给模型总结，现有新的方式是**让模型主动调用工具进行检索**，模型拿到检索结果后再总结。

**传统 RAG vs Agentic RAG 对比：**

```
┌─────────────────────────────────────────────────────────────┐
│  传统 RAG（被动检索）                                         │
├─────────────────────────────────────────────────────────────┤
│  用户问题 → 固定流程检索 → 拼接文档 → LLM生成                  │
│                                                             │
│  特点：检索是前置的、固定的、模型被动接收                       │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│  Agentic RAG（主动检索）                                      │
├─────────────────────────────────────────────────────────────┤
│  用户问题 → LLM分析 → LLM决定调用检索工具 →                   │
│  LLM拿到结果 → LLM分析是否需要再检索 → LLM总结回答             │
│                                                             │
│  特点：检索由模型主动决策、可迭代、可多轮                       │
└─────────────────────────────────────────────────────────────┘
```

**Agentic RAG 的优势：**

| 优势 | 说明 |
|------|------|
| 智能判断 | 模型自己判断是否需要检索（"你好"这类问题不需要检索） |
| 动态查询 | 模型可以优化查询词（用户问"A问题"，模型可能检索"A和 B"） |
| 迭代检索 | 第一次检索结果不满意，可以调整策略再检索 |
| 多源检索 | 可以调用多个不同的知识库工具 |
| 上下文理解 | 模型根据对话历史决定检索什么 |

**实现方式（Function Calling / Tool Use）：**

```python
# 定义检索工具
tools = [
    {
        "name": "search_knowledge_base",
        "description": "搜索知识库获取相关文档",
        "parameters": {
            "query": "搜索查询词",
            "top_k": "返回文档数量"
        }
    }
]

# Agentic RAG 流程示例
用户问题: "HuggingFace datasets 如何处理大数据？"
    ↓ 
LLM 分析: "这个问题需要检索知识库"
    ↓ 
LLM 调用工具: search_knowledge_base("datasets large data memory mapping", top_k=5)
    ↓ 
工具返回: 5条相关文档
    ↓ 
LLM 分析: "文档提到了 memory mapping，但不够详细，再检索一次"
    ↓ 
LLM 调用工具: search_knowledge_base("Apache Arrow lazy loading", top_k=3)
    ↓ 
工具返回: 3条补充文档
    ↓ 
LLM 总结: 综合所有文档生成回答
```

**Claude 实现 Agentic RAG 示例：**

```python
def agentic_rag(question):
    # 第一轮：模型决定是否检索
    response = claude.chat(
        messages=[{"role": "user", "content": question}],
        tools=[search_tool]
    )
    
    # 如果模型调用工具
    if response.stop_reason == "tool_use":
        tool_result = execute_search(response.tool_use_input)
        
        # 第二轮：模型拿到结果后继续处理
        response = claude.chat(
            messages=[
                {"role": "user", "content": question},
                {"role": "assistant", "tool_use": response.tool_use},
                {"role": "user", "tool_result": tool_result}
            ],
            tools=[search_tool]
        )
        
        # 模型可能再次调用工具（迭代检索）
        # 或者直接生成最终回答
        
    return response.content
```

**挑战与解决方案：**

| 挑战 | 解决方案 |
|------|----------|
| Token 消耗增加 | 控制检索文档长度、限制迭代次数 |
| 延迟增加 | 设置最大检索轮数（如 3轮） |
| 模型可能乱检索 | 在 tool description 中明确使用场景 |
| 检索结果无关时模型困惑 | 工具返回"无相关结果"标识 |

---

#### 检索文档过长超出上下文的解决方案

当检索的文档太大，超过模型的上下文长度限制时，有以下解决方案：

```
┌─────────────────────────────────────────────────────────────┐
│  问题：检索文档过长，超过模型上下文限制                         │
├─────────────────────────────────────────────────────────────┤
│  模型上下文限制：                                            │
│  - GPT-4: 8K / 32K tokens                                   │
│  - Claude: 100K / 200K tokens                               │
│  - 检索10篇文档 × 5000 tokens = 50K tokens → 可能超限        │
└─────────────────────────────────────────────────────────────┘
```

**1. 文档分块（Chunking）**

最常用的方法，在存储阶段就把长文档切成小块：

```
原始文档（10000 tokens）
    ↓ 分块
Chunk 1: 0-500 tokens    ┐
Chunk 2: 500-1000 tokens │ 每个块独立建立 embedding
Chunk 3: 1000-1500 tokens│
...                      ┘
    ↓ 检索时
只返回最相关的 3-5 个小块（总计 1500-2500 tokens）
```

分块策略对比：

| 策略 | 说明 | 适用场景 |
|------|------|----------|
| 固定长度 | 每 512 tokens 一块 | 通用场景 |
| 按段落 | 按自然段落分割 | 文章、博客 |
| 按语义 | 用模型检测语义边界 | 复杂文档 |
| 重叠分块 | 相邻块重叠 50-100 tokens | 防止信息断裂 |

**2. 限制检索数量（Top-K 控制）**

```python
# 限制返回文档数量和长度
results = faiss_search(query, top_k=3)  # 只取前3个
results = truncate_each(results, max_tokens=500)  # 每个最多500 tokens
```

**3. 重排序（Reranking）**

先用快速方法检索大量文档，再用精细模型重排序：

```
第一步：快速检索（Embedding + FAISS）
    检索 50 个文档（粗筛）
    ↓
第二步：精细重排序（Cross-Encoder 模型）
    对 50 个文档重新打分排序
    ↓
第三步：只保留 Top-5 最相关文档
    （精细模型判断更准确，减少无效文档）
```

Cross-Encoder vs Bi-Encoder：

```
Bi-Encoder（Embedding 检索）：
  Query → Embedding → 计算相似度 → 快但不精确

Cross-Encoder（重排序）：
  Query + Doc → 模型 → 精确打分 → 慢但准确
```

**4. 文档摘要压缩**

```python
# 检索后先压缩再喂给模型
def compress_documents(docs, query):
    # 用小模型对每个文档生成摘要（针对 query）
    summaries = []
    for doc in docs:
        prompt = f"问题：{query}\n文档：{doc}\n请提取与问题相关的关键信息（不超过200字）："
        summary = small_model.generate(prompt)
        summaries.append(summary)
    return summaries

# 摘要后总长度大幅减少
原始: 10 docs × 5000 tokens = 50K tokens
摘要后: 10 docs × 200 tokens = 2K tokens
```

**5. 分层检索（两阶段）**

```
第一层：检索文档摘要/标题
    摘要库（每个 100 tokens）→ 快速筛选出相关文档
    ↓
第二层：只对筛选出的文档检索全文
    减少需要处理的文档数量
```

**6. 选择长上下文模型**

| 模型 | 上下文长度 | 说明 |
|------|-----------|------|
| Claude 3 | 200K tokens | 约 150K 字英文，适合长文档 |
| GPT-4 Turbo | 128K tokens | 较长上下文 |
| Gemini 1.5 Pro | 1M tokens | 超长上下文 |

**7. Agentic RAG 的迭代策略**

```python
# 模型主动控制检索量
def agentic_rag_with_control(question):
    # 第一轮：小量检索试探
    docs = search(question, top_k=3, max_tokens_per_doc=500)
    
    # 模型判断是否足够
    if model_thinks_not_enough(docs):
        # 细化查询再检索
        new_query = model_refine_query(question, docs)
        more_docs = search(new_query, top_k=2)
        docs.extend(more_docs)
    
    # 模型生成回答
    return model.generate(question, docs)
```

**方案综合对比：**

| 方案 | 优点 | 缺点 |
|------|------|------|
| 文档分块 | 精度高、实现简单 | 可能丢失上下文 |
| Top-K 控制 | 简单直接 | 可能漏掉重要信息 |
| 重排序 | 质量高 | 多一次模型调用 |
| 摘要压缩 | 大幅减少长度 | 损失细节、增加延迟 |
| 分层检索 | 效率高 | 需要维护摘要库 |
| 长上下文模型 | 最简单 | 成本高、延迟高 |
| Agentic 迭代 | 灵活智能 | 实现复杂 |

---