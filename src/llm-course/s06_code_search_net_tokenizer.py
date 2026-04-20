"""
CodeSearchNet Tokenizer Training Script
训练一个专门用于Python代码的tokenizer
"""

# ==================== 第一部分：加载数据集 ====================
print("\n" + "=" * 60)
print("第一部分：加载 CodeSearchNet 数据集")
print("=" * 60 + "\n")

from datasets import load_dataset

# 加载CodeSearchNet数据集的Python子集
# 该数据集包含大量Python函数代码，适合训练代码tokenizer
raw_datasets = load_dataset("code_search_net", "python")

# 打印数据集结构信息
print("数据集结构：")
print(raw_datasets)

# 打印训练集详细信息
print("\n训练集详情：")
print(raw_datasets["train"])

# 打印一个示例函数，查看数据格式
print("\n示例函数（索引123456）：")
print(raw_datasets["train"][123456]["whole_func_string"])

# ==================== 第二部分：准备训练语料 ====================
print("\n" + "=" * 60)
print("第二部分：准备训练语料（Training Corpus）")
print("=" * 60 + "\n")


def get_training_corpus():
    """
    生成器函数：分批提取训练语料

    tokenizer训练需要迭代器形式的数据，
    每次返回一批文本样本，避免一次性加载全部数据到内存。

    Args:
        无参数，使用全局的raw_datasets

    Yields:
        list: 包含1000个函数字符串的批次
    """
    dataset = raw_datasets["train"]
    # 每1000个样本为一批，逐步遍历整个训练集
    for start_idx in range(0, len(dataset), 1000):
        samples = dataset[start_idx: start_idx + 1000]
        # 提取完整函数字符串作为训练文本
        yield samples["whole_func_string"]


# 创建训练语料迭代器
training_corpus = get_training_corpus()
print("训练语料迭代器已创建，将分批提供函数代码样本")

# ==================== 第三部分：加载旧tokenizer ====================
print("\n" + "=" * 60)
print("第三部分：加载基础 GPT-2 Tokenizer")
print("=" * 60 + "\n")

from transformers import AutoTokenizer

# 加载预训练的GPT-2 tokenizer作为基础
# GPT-2 tokenizer主要针对自然语言，对代码的分词效果不佳
old_tokenizer = AutoTokenizer.from_pretrained("gpt2")
print("已加载 GPT-2 tokenizer 作为基础模型")

# 定义一个Python函数示例用于对比测试
example = '''def add_numbers(a, b):
    """Add the two numbers `a` and `b`."""
    return a + b'''

print("\n测试代码示例：")
print(example)

# 使用旧tokenizer进行分词
tokens = old_tokenizer.tokenize(example)
print("\n使用 GPT-2 tokenizer 分词结果：")
print(tokens)
print(f"分词数量: {len(tokens)}")

# ==================== 第四部分：训练新tokenizer ====================
print("\n" + "=" * 60)
print("第四部分：训练新的代码专用 Tokenizer")
print("=" * 60 + "\n")

# 从旧tokenizer训练新版本，词汇表大小设为52000
# 训练过程中会学习Python代码的特殊模式（如缩进、函数名、操作符等）
print("开始训练新tokenizer（词汇表大小: 52000）...")
tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, 52000)
print("训练完成！")

# 使用新tokenizer对相同示例进行分词，对比效果
tokens = tokenizer.tokenize(example)
print("\n使用新训练的 tokenizer 分词结果：")
print(tokens)
print(f"分词数量: {len(tokens)}")

# 对比说明：新tokenizer能更好地识别代码结构
# 例如：函数名、缩进空格、特殊字符等会被更合理地分割


# ==================== 第五部分：保存tokenizer ====================
print("\n" + "=" * 60)
print("第五部分：保存和上传 Tokenizer")
print("=" * 60 + "\n")

# 保存tokenizer到本地目录
tokenizer.save_pretrained("code-search-net-tokenizer")
print("Tokenizer 已保存到本地目录: code-search-net-tokenizer")

# 上传tokenizer到Hugging Face Hub
# 需要已登录Hugging Face账号（huggingface-cli login）
tokenizer.push_to_hub("code-search-net-tokenizer")
print("Tokenizer 已上传到 Hugging Face Hub")

print("\n" + "=" * 60)
print("全部完成！")
print("=" * 60 + "\n")
