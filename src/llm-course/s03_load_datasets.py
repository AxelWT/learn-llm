# 加载 GLUE 基准测试中的 MRPC（微软研究释义语料库）数据集
# MRPC 是一个句子对分类任务，判断两个句子是否语义相同（ paraphrase 或 not paraphrase）
from datasets import load_dataset

# load_dataset() 从 HuggingFace Hub 下载并加载数据集
# "glue" 是基准测试名称，"mrpc" 是具体子任务
raw_datasets = load_dataset("glue", "mrpc")
print(raw_datasets)  # 打印数据集结构，包含 train、validation、test 三个子集

# 查看训练集中的第 15 条样本
# 每条样本包含：sentence1、sentence2、label（0=不相同，1=相同）、idx
print(raw_datasets["train"][15])

# 查看验证集中的第 87 条样本
print(raw_datasets["validation"][87])

#结果
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

label == 0 表示两句非同义，label == 1 表示两句同义
{'sentence1': 'Rudder was most recently senior vice president for the Developer & Platform Evangelism Business .', 'sentence2': 'Senior Vice President Eric Rudder , formerly head of the Developer and Platform Evangelism unit , will lead the new entity .', 'label': 0, 'idx': 16}
{'sentence1': 'However , EPA officials would not confirm the 20 percent figure .', 'sentence2': 'Only in the past few weeks have officials settled on the 20 percent figure .', 'label': 0, 'idx': 812}
"""