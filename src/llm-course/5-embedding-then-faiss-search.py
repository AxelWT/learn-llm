"""
GitHub Issues 数据集处理脚本
功能：从 HuggingFace Hub 加载 GitHub issues 数据，筛选并转换为可用于嵌入检索的文本格式
"""

from datasets import load_dataset

# 从 HuggingFace Hub 加载 lewtun/github-issues 数据集，使用 train 分割
issues_dataset = load_dataset("lewtun/github-issues", split="train")

# 筛选数据集：
# - 只保留 issues（排除 pull requests）
# - 只保留有评论的 issues
issues_dataset = issues_dataset.filter(
    lambda x: (x["is_pull_request"] == False and len(x["comments"]) > 0)
)

print(issues_dataset)

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

# 将 comments 列"展开"（explode）：
# 原来每行可能有多条评论（列表形式），展开后每条评论变成单独一行
# ignore_index=True 表示重置索引
comments_df = df.explode("comments", ignore_index=True)
comments_df.head(4)  # 显示前4行（用于调试）

from datasets import Dataset

# 将 pandas DataFrame 转换回 HuggingFace Dataset 格式
comments_dataset = Dataset.from_pandas(comments_df)
print(comments_dataset)

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

# 加载 sentence-transformers 模型，用于生成文本嵌入向量
# multi-qa-mpnet-base-dot-v1 是一个专门针对问答检索任务优化的模型
from transformers import AutoTokenizer, AutoModel

model_ckpt = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)  # 加载分词器
model = AutoModel.from_pretrained(model_ckpt)  # 加载预训练模型


def cls_pooling(model_output):
    """
    CLS 池化：从模型输出中提取 [CLS] token 的嵌入向量
    [CLS] token 位于序列的第一个位置（index 0），通常用于表示整个序列的语义
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

# 对整个数据集计算嵌入向量
# 每条文本生成一个 768 维的嵌入向量（mpnet-base 模型的输出维度）
# .detach().cpu().numpy()[0] 用于将 PyTorch tensor 转换为 numpy 数组
embeddings_dataset = comments_dataset.map(
    lambda x: {"embeddings": get_embeddings(x["text"]).detach().cpu().numpy()[0]}
)

print(embeddings_dataset)

# 使用 FAISS 构建 embeddings 列的向量索引
# FAISS 是 Facebook 开发的高效向量相似度搜索库，支持大规模向量快速检索
embeddings_dataset.add_faiss_index(column="embeddings")

# 定义查询问题
question = "How can I load a dataset offline?"
# 将查询问题转换为嵌入向量
question_embedding = get_embeddings([question]).cpu().detach().numpy()
print(question_embedding.shape)

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
