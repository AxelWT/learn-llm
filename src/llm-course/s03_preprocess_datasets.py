from datasets import load_dataset
from transformers import AutoTokenizer

# 加载的映射（索引），数据不进内存
raw_datasets = load_dataset("glue", "mrpc")
print(raw_datasets)
"""
DatasetDict({
    train: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx'],
        num_rows: 3668
    })
    validation: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx'],
        num_rows: 408
    })
    test: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx'],
        num_rows: 1725
    })
})
"""
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# 输入句子数据进行分词，生成模型可以读懂的数字数据
ids1 = tokenizer("This is the first sentence.", "This is the second one.")
print(ids1)
"""
{'input_ids': [101, 2023, 2003, 1996, 2034, 6251, 1012, 102, 2023, 2003, 1996, 2117, 2028, 1012, 102], 
'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1], 
'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]}
"""

# 把数字转换回分词文字
t1 = tokenizer.convert_ids_to_tokens(ids1["input_ids"])
# 多句话都在同一个 input_ids列表中了，token_type_ids的值用来区分每一句话（有些模型有token_type_ids，取决于模型预训练时是否用过）
print(t1)
"""
['[CLS]', 'this', 'is', 'the', 'first', 'sentence', '.', '[SEP]', 'this', 'is', 'the', 'second', 'one', '.', '[SEP]']
"""


# 预处理训练数据集

# 首先raw_datasets = load_dataset("glue", "mrpc")方式下载的数据集会保存在本地磁盘，以 Apache Arrow 格式
# 以这种方式将磁盘中的数据集所有数据加载到内存中，可能内存会不够用
# tokenized_dataset = tokenizer(raw_datasets["train"]["sentence1"], raw_datasets["train"]["sentence2"], padding=True, truncation=True)

def tokenize_function(example):
    # 之前直接 tokenizer(raw_datasets["train"]["sentence1"]) 报错，是因为 Arrow 类型不被识别。
    # 而 .map() 方法内部会自动处理类型转换，把 Arrow 数据转成 Python 原生类型再传给你的函数，所以 example["sentence1"] 实际上已经是 list[str]，tokenizer 可以正常处理。
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


# 将数据集分成多个 batch（默认每批 1000条），对每个 batch 调用tokenize_function，将返回的 tokenization 结果合并到原始数据集中得到新的数据集
tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
print(tokenized_datasets)
print(tokenized_datasets.cache_files)
"""
DatasetDict({
    train: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx', 'input_ids', 'token_type_ids', 'attention_mask'],
        num_rows: 3668
    })
    validation: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx', 'input_ids', 'token_type_ids', 'attention_mask'],
        num_rows: 408
    })
    test: Dataset({
        features: ['sentence1', 'sentence2', 'label', 'idx', 'input_ids', 'token_type_ids', 'attention_mask'],
        num_rows: 1725
    })
})
{'train': [{'filename': '/Users/<account>/.cache/huggingface/datasets/glue/mrpc/0.0.0/bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c/cache-9c65154dc04d15c8.arrow'}], 
'validation': [{'filename': '/Users/<account>/.cache/huggingface/datasets/glue/mrpc/0.0.0/bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c/cache-c476afed73355cef.arrow'}], 
'test': [{'filename': '/Users/<account>/.cache/huggingface/datasets/glue/mrpc/0.0.0/bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c/cache-49ddf8863e6bf95c.arrow'}]}

"""

# 取预处理后的数据集的前 8 行数据，并去除idx,sentence1,sentence2键
samples = tokenized_datasets["train"][:8]
samples = {k: v for k, v in samples.items() if k not in ["idx", "sentence1", "sentence2"]}
print([len(x) for x in samples["input_ids"]])
"""
结果如下，长度不统一
[50, 59, 47, 67, 59, 50, 62, 32]
"""

# 动态填充
# 因为每个句子长度不同，需要填充到统一长度，该工具会在一个 batch 内找到最长的那条，把其他短的补 0（padding token）到相同长度，这样比预先固定长度更省内存
from transformers import DataCollatorWithPadding

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 进行动态填充，长度统一
batch2 = data_collator(samples)
print({k: v.shape for k, v in batch2.items()})
"""
{'input_ids': torch.Size([8, 67]), 
'token_type_ids': torch.Size([8, 67]), 
'attention_mask': torch.Size([8, 67]), 
'labels': torch.Size([8])}
"""