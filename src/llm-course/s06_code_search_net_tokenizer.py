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
"""
DatasetDict({
    train: Dataset({
        features: ['repository_name', 'func_path_in_repository', 'func_name', 'whole_func_string', 'language', 'func_code_string', 'func_code_tokens', 'func_documentation_string', 'func_documentation_tokens', 'split_name', 'func_code_url'],
        num_rows: 412178
    })
    test: Dataset({
        features: ['repository_name', 'func_path_in_repository', 'func_name', 'whole_func_string', 'language', 'func_code_string', 'func_code_tokens', 'func_documentation_string', 'func_documentation_tokens', 'split_name', 'func_code_url'],
        num_rows: 22176
    })
    validation: Dataset({
        features: ['repository_name', 'func_path_in_repository', 'func_name', 'whole_func_string', 'language', 'func_code_string', 'func_code_tokens', 'func_documentation_string', 'func_documentation_tokens', 'split_name', 'func_code_url'],
        num_rows: 23107
    })
})
"""

# 打印一个示例函数，查看数据格式
print("\n示例函数（索引123456）：")
print(raw_datasets["train"][123456]["whole_func_string"])
"""
def oauth_token_create(self, data, **kwargs):
        api_path = "/api/v2/oauth/tokens.json"
        return self.call(api_path, method="POST", data=data, **kwargs)
"""

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

# 打印模型的 embedding 层参数
from transformers import GPT2LMHeadModel

model = GPT2LMHeadModel.from_pretrained('gpt2')
print(model.config.vocab_size)  # 模型使用的词汇表大小
print(model.config.n_embd)  # 每个 token 用多少维的向量表示
"""
50257
768
"""

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

# 使用旧tokenizer进行分词
tokens = old_tokenizer.tokenize(example)
print("\n使用 GPT-2 tokenizer 分词结果：")
print(tokens)
print(f"分词数量: {len(tokens)}")
"""
['def', 'Ġadd', '_', 'n', 'umbers', '(', 'a', ',', 'Ġb', '):', ..., Ġ`', 'b', '`', '."', '""', 'Ċ', 'Ġ', 'Ġ', 'Ġ', 'Ġreturn', 'Ġa', 'Ġ+', 'Ġb']
分词数量: 36
"""

# ==================== 第四部分：训练新tokenizer ====================
print("\n" + "=" * 60)
print("第四部分：训练新的代码专用 Tokenizer")
print("=" * 60 + "\n")

# 从旧tokenizer训练新版本，词汇表大小设为52000
# 训练过程中会学习Python代码的特殊模式（如缩进、函数名、操作符等）
# GPT-2 基础词汇表大小: GPT-2 原始 tokenizer 的词汇表大小是 50257 个 token
# 52000 比原始多出约 1743 个新 token 位置，这些额外位置用于学习 Python 代码特有的模式（如缩进     、操作符 ->、函数命名风格等）
print("开始训练新tokenizer（词汇表大小: 52000）...")  # 自定义的
# train_new_from_iterator 只是在统计"哪些字符组合出现最频繁"，然后把高频组合合并成新 token。全程是纯数学统计，没有任何神经网络参与。
tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, 52000)
print("训练完成！")

# 使用新tokenizer对相同示例进行分词，对比效果
tokens = tokenizer.tokenize(example)
print("\n使用新训练的 tokenizer 分词结果：")
print(tokens)
print(f"分词数量: {len(tokens)}")
"""
['def', 'Ġadd', '_', 'numbers', '(', 'a', ',', 'Ġb', '):', 'ĊĠĠĠ', 'Ġ"', ....]
分词数量: 27
"""

# 对比说明：新tokenizer能更好地识别代码结构
# 例如：函数名、缩进空格、特殊字符等会被更合理地分割


# ==================== 第五部分：保存tokenizer ====================
print("\n" + "=" * 60)
print("第五部分：保存和上传 Tokenizer")
print("=" * 60 + "\n")

# 保存tokenizer到本地目录
tokenizer.save_pretrained("code-search-net-tokenizer")
print("Tokenizer 已保存到本地目录: code-search-net-tokenizer")

# 对比新旧词汇表
old_vocab = old_tokenizer.get_vocab()
new_vocab = tokenizer.get_vocab()

# 检查是否有差异
same_tokens = old_vocab == new_vocab
print(f"词汇表是否完全相同: {same_tokens}")
"""
词汇表是否完全相同: False

相对于旧的 tokenizer 变动了 62503 个 token（其中30380个旧token被替换，32123个新token被加入）。

如果你想要保留全部旧 vocab 只新增 1743 个，需要设置不同的参数：                                                                                                                    
  tokenizer = old_tokenizer.train_new_from_iterator(                                                                                                                                 
      training_corpus,                                                                                                                                                               
      vocab_size=52000,                                                                                                                                                              
      new_tokens=None  # 不强制替换旧token                                                                                                                                           
  )                                                                                                                                                                                                                                                                                                                                                                  
  但这通常不是最优做法——因为保留大量代码中不常用的自然语言 token 会浪费词汇表空间。
"""

# 上传tokenizer到Hugging Face Hub
# 需要已登录Hugging Face账号（huggingface-cli login）
tokenizer.push_to_hub("code-search-net-tokenizer")
print("Tokenizer 已上传到 Hugging Face Hub")

print("\n" + "=" * 60)
print("全部完成！")
print("=" * 60 + "\n")

"""
新的数据有几十万条，可以把词汇表扩充到几十万的大小吗?                                                                                                                              
                                                                                                                                                                                     
⏺ 可以，但不建议。原因如下：                                                                                                                                                                                                                                                                                                                                   
  词汇表大小的权衡                                                                                                                                                                   
  ┌─────────────────┬───────────────────────────┬───────────────────────────────────────────────────────────────┐                                                                    
  │   词汇表大小      │           优点            │                             缺点                              │
  ├─────────────────┼───────────────────────────┼───────────────────────────────────────────────────────────────┤                                                                    
  │ 大 (几十万)      │ 每个词/token 都有唯一表示    │ Embedding 层参数爆炸、内存占用大、推理慢、很多 token 出现稀少.        │
  ├─────────────────┼───────────────────────────┼───────────────────────────────────────────────────────────────┤                                                                    
  │ 适中 (5万-10万)  │ 平衡表示能力和效率           │ 主流做法，经验验证有效                                           │                                                                    
  ├─────────────────┼───────────────────────────┼───────────────────────────────────────────────────────────────┤                                                                    
  │ 小 (<5万)       │ 参数少、速度快               │ 频繁拆分，序列过长                                               │                                                                    
  └─────────────────┴───────────────────────────┴───────────────────────────────────────────────────────────────┘                                                                    
                                                            
  具体计算                                                                                                                                                                           
                                                            
  假设模型 hidden_size = 768（类似 GPT-2 small）：                                                                                                                                   
                                                            
  ┌────────────┬──────────────────────┬──────────┐                                                                                                                                   
  │ vocab_size │   Embedding 参数量   │ 相对大小 │          
  ├────────────┼──────────────────────┼──────────┤                                                                                                                                   
  │ 52,000     │ 52,000 × 768 = 40M   │ 基准     │          
  ├────────────┼──────────────────────┼──────────┤                                                                                                                                   
  │ 100,000    │ 100,000 × 768 = 77M  │ ~2倍     │                                                                                                                                   
  ├────────────┼──────────────────────┼──────────┤                                                                                                                                   
  │ 300,000    │ 300,000 × 768 = 230M │ ~6倍     │                                                                                                                                   
  ├────────────┼──────────────────────┼──────────┤                                                                                                                                   
  │ 500,000    │ 500,000 × 768 = 384M │ ~10倍    │          
  └────────────┴──────────────────────┴──────────┘                                                                                                                                   
                                                            
  词汇表从 5万 扩到 50万，仅 Embedding 层就增加 340M 参数。                                                                                                                          
                                          
  数据量 vs 词汇表大小                                                                                                                                                               
                                                            
  - 数据量 主要影响 tokenizer 学习合并规则的质量                                                                                                                                     
  - 词汇表大小 决定最终保留多少 token
                                                                                                                                                                                     
  即使有 40万条代码，真正高频出现的 token 模式远少于这个数字。大部分词会被拆分成子词（subword），而不是每个函数名都单独成为一个 token。                                              
                                                                                                                                                                                     
  建议范围                                                                                                                                                                           
                                                            
  对于代码 tokenizer：                                                                                                                                                               
  - 50,000 - 100,000 是合理范围                             
  - 超过 100,000 收益递减，成本显著上升                                                                                                                                              
                                                            
  如果想扩充，可以尝试：                                                                                                                                                             
  tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, 100000)  # 适度扩充                                                                                             
                                                                                                                                                                                     
  但几十万（300k+）的词汇表大小在实际中很少使用，除非有特殊需求（如多语言覆盖）。 
"""

"""
  ┌────────────────────────────────────────────────────────────┐                                                                                                                     
  │  tokenizer 变化 → 模型 embedding 层必须重新训练                │                                                                                                                     
  ├────────────────────────────────────────────────────────────┤                                                                                                                     
  │                                                            │                                                                                                                     
  │  GPT-2 原模型:                                              │                                                                                                                     
  │  embedding 层形状 = [50257 × 768]                           │                                                                                                                     
  │  每个 token id 对应一个 768 维向量                            │                                                                                                                      
  │                                                            │                                                                                                                     
  │  新 tokenizer:                                             │                                                                                                                     
  │  vocab_size = 52000                                        │                                                                                                                     
  │  token id 映射完全不同（同个词可能变成不同 id）                  │                                                                                                                     
  │                                                            │
  │  问题:                                                     │
  │  1. 形状不匹配 → embedding 层需要扩展到 [52000 × 768]          │
  │  2. id 映射变了 → 原 embedding 向量无法直接复用                │
  │     （"def" 在旧 tokenizer id=100，新 tokenizer id=200）     │
  │                                                            │
  │  解决方案:                                                  │
  │  - 从头训练新模型                                            │
  │  - 或保留原模型权重，初始化新 embedding，然后微调                │
  │                                                            │
  └────────────────────────────────────────────────────────────┘

  简单说：tokenizer 和模型 embedding 层是绑定的。vocab 变了，embedding 必须变，否则模型输出就是乱码。
"""
