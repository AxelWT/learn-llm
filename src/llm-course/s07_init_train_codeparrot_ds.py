# ========================================
# CodeParrot 代码生成模型训练脚本
# 使用 GPT-2 模型在 Python 代码数据集上进行语言模型训练
# 目标：训练一个能够生成 Python 数据科学代码的模型
# ========================================

print("=" * 60)
print("步骤1: 加载 CodeParrot 代码数据集（流式加载）")
print("=" * 60)

from datasets import load_dataset

split = "train"  # 可选 "valid"
# 定义过滤关键词：只保留包含这些库的代码样本
filters = ["pandas", "sklearn", "matplotlib", "seaborn"]

print(f"数据分割: {split}")
print(f"过滤关键词: {filters}")

# 流式加载数据集（避免一次性加载大量数据到内存）
data = load_dataset(f"transformersbook/codeparrot-{split}", split=split, streaming=True)
print("数据集流式加载完成！")


# ========================================
# 步骤2: 定义过滤函数
# ========================================

print("\n" + "=" * 60)
print("步骤2: 定义数据过滤函数")
print("=" * 60)

def any_keyword_in_string(string, keywords):
    """检查字符串中是否包含任意关键词"""
    for keyword in keywords:
        if keyword in string:
            return True
    return False

def filter_streaming_dataset(dataset, filters):
    """
    过滤流式数据集：只保留包含指定关键词的样本
    用于筛选数据科学相关的代码（pandas, sklearn等）
    """
    from collections import defaultdict
    from tqdm import tqdm
    from datasets import Dataset

    filtered_dict = defaultdict(list)
    total = 0
    for sample in tqdm(iter(dataset), desc="过滤数据"):
        total += 1
        if any_keyword_in_string(sample["content"], filters):
            for k, v in sample.items():
                filtered_dict[k].append(v)
    print(f"{len(filtered_dict['content']) / total:.2%} of data after filtering.")
    return Dataset.from_dict(filtered_dict)

print("过滤函数定义完成！")
print(f"说明: 将筛选包含 {filters} 关键词的代码样本")


# ========================================
# 步骤3: 执行数据过滤
# ========================================

print("\n" + "=" * 60)
print("步骤3: 执行数据过滤")
print("=" * 60)

filtered_data = filter_streaming_dataset(data, filters)
print(f"过滤后样本数: {len(filtered_data)}")


# ========================================
# 步骤4: 加载预处理好的数据集
# ========================================

print("\n" + "=" * 60)
print("步骤4: 加载预处理好的训练和验证数据集")
print("=" * 60)

from datasets import load_dataset, DatasetDict

# 直接加载预处理好的数据科学代码数据集
ds_train = load_dataset("huggingface-course/codeparrot-ds-train", split="train")
ds_valid = load_dataset("huggingface-course/codeparrot-ds-valid", split="validation")

print(f"训练集加载完成: {ds_train.num_rows} 样本")
print(f"验证集加载完成: {ds_valid.num_rows} 样本")


# ========================================
# 步骤5: 创建数据集字典并展示样本
# ========================================

print("\n" + "=" * 60)
print("步骤5: 创建数据集字典并展示样本")
print("=" * 60)

raw_datasets = DatasetDict(
    {
        "train": ds_train,  # .shuffle().select(range(50000)),  # 可选：限制样本数
        "valid": ds_valid,  # .shuffle().select(range(500))
    }
)

print("数据集结构:")
print(raw_datasets)

# 展示第一个样本的内容
print("\n第一个样本内容预览:")
for key in raw_datasets["train"][0]:
    content = raw_datasets["train"][0][key]
    if isinstance(content, str):
        print(f"{key.upper()}: {content[:200]}...")
    else:
        print(f"{key.upper()}: {content}")


# ========================================
# 步骤6: 加载代码专用分词器
# ========================================

print("\n" + "=" * 60)
print("步骤6: 加载代码专用分词器")
print("=" * 60)

from transformers import AutoTokenizer

context_length = 128  # 上下文长度（每个样本的最大token数）
tokenizer = AutoTokenizer.from_pretrained("huggingface-course/code-search-net-tokenizer")

print(f"分词器: huggingface-course/code-search-net-tokenizer")
print(f"上下文长度: {context_length}")
print(f"词汇表大小: {len(tokenizer)}")


# ========================================
# 步骤7: 测试分词器的长文本处理
# ========================================

print("\n" + "=" * 60)
print("步骤7: 测试分词器的长文本截断和分块")
print("=" * 60)

# 分词并处理超出最大长度的情况
outputs = tokenizer(
    raw_datasets["train"][:2]["content"],
    truncation=True,                    # 截断超出长度的文本
    max_length=context_length,          # 最大长度
    return_overflowing_tokens=True,      # 返回溢出的tokens（长文本会被分成多个块）
    return_length=True,                  # 返回每个块的长度
)

print(f"原始样本数: 2")
print(f"分词后的块数: {len(outputs['input_ids'])}")
print(f"各块长度: {outputs['length']}")
print(f"块到样本的映射: {outputs['overflow_to_sample_mapping']}")

# 解释: overflow_to_sample_mapping 表示每个块来自哪个原始样本


# ========================================
# 步骤8: 定义批量分词函数
# ========================================

print("\n" + "=" * 60)
print("步骤8: 定义批量分词函数")
print("=" * 60)

def tokenize(element):
    """
    批量分词函数：
    - 将代码文本分词
    - 截断到最大长度
    - 处理溢出tokens（长文本分成多个块）
    - 只保留完整长度的块（用于训练）
    """
    outputs = tokenizer(
        element["content"],
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=True,
        return_length=True,
    )
    input_batch = []
    # 只保留长度等于 context_length 的块（确保训练数据完整）
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        if length == context_length:
            input_batch.append(input_ids)
    return {"input_ids": input_batch}

print("分词函数定义完成！")
print(f"说明: 只保留长度={context_length}的完整块，丢弃不完整的块")


# ========================================
# 步骤9: 对整个数据集进行分词
# ========================================

print("\n" + "=" * 60)
print("步骤9: 对整个数据集进行分词预处理")
print("=" * 60)

tokenized_datasets = raw_datasets.map(
    tokenize,
    batched=True,                                    # 批量处理
    remove_columns=raw_datasets["train"].column_names  # 移除原始列
)

print("分词后的数据集:")
print(tokenized_datasets)
print(f"训练样本数: {len(tokenized_datasets['train'])}")
print(f"验证样本数: {len(tokenized_datasets['valid'])}")


# ========================================
# 步骤10: 配置 GPT-2 模型
# ========================================

print("\n" + "=" * 60)
print("步骤10: 配置 GPT-2 语言模型")
print("=" * 60)

from transformers import AutoTokenizer, GPT2LMHeadModel, AutoConfig

config = AutoConfig.from_pretrained(
    "gpt2",
    vocab_size=len(tokenizer),           # 使用分词器的词汇表大小
    n_ctx=context_length,                # 上下文长度
    bos_token_id=tokenizer.bos_token_id, # 开始token ID
    eos_token_id=tokenizer.eos_token_id, # 结束token ID
)

print("GPT-2 模型配置:")
print(f"  - 词汇表大小: {config.vocab_size}")
print(f"  - 上下文长度: {config.n_ctx}")
print(f"  - 层数: {config.n_layer}")
print(f"  - 注意力头数: {config.n_head}")
print(f"  - 嵌入维度: {config.n_embd}")


# ========================================
# 步骤11: 初始化模型并查看参数量
# ========================================

print("\n" + "=" * 60)
print("步骤11: 初始化 GPT-2 模型并查看参数量")
print("=" * 60)

model = GPT2LMHeadModel(config)
model_size = sum(t.numel() for t in model.parameters())
print(f"GPT-2 模型参数量: {model_size / 1000 ** 2:.1f}M")
print(f"说明: 这是一个小型 GPT-2 模型，用于代码生成任务")


# ========================================
# 步骤12: 创建数据整理器
# ========================================

print("\n" + "=" * 60)
print("步骤12: 创建语言模型数据整理器")
print("=" * 60)

from transformers import DataCollatorForLanguageModeling

# 设置 pad_token（GPT-2 原始没有 pad_token，使用 eos_token 替代）
tokenizer.pad_token = tokenizer.eos_token

# 创建数据整理器（因果语言模型，不使用 MLM）
data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

print("数据整理器创建完成！")
print(f"  - 使用因果语言模型（Causal LM）")
print(f"  - MLM=False（不使用掩码语言模型）")
print(f"  - Pad token = EOS token")

# 测试数据整理器
out = data_collator([tokenized_datasets["train"][i] for i in range(5)])
print("\n数据整理器测试（5个样本）:")
for key in out:
    print(f"  {key} shape: {out[key].shape}")


# ========================================
# 步骤13: 配置训练参数
# ========================================

print("\n" + "=" * 60)
print("步骤13: 配置训练参数")
print("=" * 60)

from transformers import Trainer, TrainingArguments

args = TrainingArguments(
    output_dir="codeparrot-ds",              # 输出目录
    per_device_train_batch_size=32,          # 训练批次大小
    per_device_eval_batch_size=32,           # 评估批次大小
    eval_strategy="steps",             # 每隔一定步数评估
    eval_steps=5_000,                        # 每5000步评估一次
    logging_steps=5_000,                     # 每5000步记录日志
    gradient_accumulation_steps=8,           # 梯度累积步数（有效batch=32*8=256）
    num_train_epochs=1,                      # 训练轮数
    weight_decay=0.1,                        # 权重衰减
    warmup_steps=1_000,                      # 预热步数
    lr_scheduler_type="cosine",              # 学习率调度器类型
    learning_rate=5e-4,                      # 学习率
    save_steps=5_000,                        # 每5000步保存
    fp16=False,                              # 使用混合精度训练
    push_to_hub=True,                        # 推送到 Hugging Face Hub
)

print("训练参数配置:")
print(f"  - 输出目录: {args.output_dir}")
print(f"  - 训练批次: {args.per_device_train_batch_size}")
print(f"  - 有效批次: {args.per_device_train_batch_size * args.gradient_accumulation_steps} (含梯度累积)")
print(f"  - 训练轮数: {args.num_train_epochs}")
print(f"  - 学习率: {args.learning_rate}")
print(f"  - 学习率调度: {args.lr_scheduler_type}")
print(f"  - 预热步数: {args.warmup_steps}")
print(f"  - 混合精度: {args.fp16}")


# ========================================
# 步骤14: 创建训练器
# ========================================

print("\n" + "=" * 60)
print("步骤14: 创建 Trainer 训练器")
print("=" * 60)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=args,
    data_collator=data_collator,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["valid"],
)

print("训练器创建完成！")
print(f"  - 训练样本数: {len(tokenized_datasets['train'])}")
print(f"  - 验证样本数: {len(tokenized_datasets['valid'])}")


# ========================================
# 步骤15: 开始训练
# ========================================

print("\n" + "=" * 60)
print("步骤15: 开始训练代码生成模型")
print("=" * 60)
print("注意: 训练可能需要数小时，取决于硬件配置")
print("=" * 60)

trainer.train()
print("\n训练完成！")


# ========================================
# 步骤16: 推送模型到 Hub
# ========================================

print("\n" + "=" * 60)
print("步骤16: 推送模型到 Hugging Face Hub")
print("=" * 60)

trainer.push_to_hub()
print("模型已推送到 Hub！")


# ========================================
# 步骤17: 创建推理管道并测试
# ========================================

print("\n" + "=" * 60)
print("步骤17: 创建推理管道并生成代码示例")
print("=" * 60)

import torch
from transformers import pipeline

# 检测可用设备
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"使用设备: {device}")

# 创建文本生成管道
pipe = pipeline(
    "text-generation",
    model="huggingface-course/codeparrot-ds",
    device=device
)

print("推理管道创建完成！")

# 测试代码生成
prompt = "import pandas as pd\ndf = pd."
print(f"\n输入提示: {prompt}")

generated = pipe(prompt, max_length=50, num_return_sequences=1)
print("\n生成的代码:")
print(generated[0]['generated_text'])


# ========================================
# 总结
# ========================================

print("\n" + "=" * 60)
print("脚本执行总结")
print("=" * 60)

print("""
本脚本训练一个代码生成模型，完整流程如下:

1-3. 数据加载与过滤: 流式加载 CodeParrot 数据集，筛选数据科学代码
4-5. 数据集准备: 加载预处理好数据集，创建 DatasetDict
6-7. 分词器: 加载代码专用分词器，测试长文本分块
8-9. 分词预处理: 批量分词，只保留完整块
10-11. 模型配置: 初始化 GPT-2 语言模型（~124M参数）
12. 数据整理器: 创建因果语言模型整理器
13-14. 训练配置: 设置批次、学习率、混合精度等
15-16. 训练与推送: 执行训练，推送到 Hub
17. 推理测试: 创建管道，生成代码示例

关键技术点:
- 流式数据加载: 处理大规模数据集不爆内存
- 梯度累积: 小显存也能用大有效批次（32*8=256）
- 混合精度(fp16): 加速训练，减少显存占用
- Cosine学习率调度: 平滑的学习率衰减
""")
print("=" * 60)
print("所有步骤完成！")
print("=" * 60)