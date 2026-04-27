from datasets import load_dataset

# 由于 tsv 仅仅是 csv 的一个变体，可以用加载 csv 文件的 load_dataset()函数并指定分隔符，来加载这些文件
data_files = {"train": "./download/drugsComTrain_raw.tsv", "test": "./download/drugsComTest_raw.tsv"}
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
