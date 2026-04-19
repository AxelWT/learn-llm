# ========================================
# CodeParrot 代码生成模型训练脚本（使用 Accelerate）
# 使用 GPT-2 模型在 Python 代码数据集上进行语言模型训练
# 目标：训练一个能够生成 Python 数据科学代码的模型
# 特点：使用自定义关键词加权损失函数，强调数据科学关键词的学习
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
    truncation=True,  # 截断超出长度的文本
    max_length=context_length,  # 最大长度
    return_overflowing_tokens=True,  # 返回溢出的tokens（长文本会被分成多个块）
    return_length=True,  # 返回每个块的长度
)

print(f"原始样本数: 2")
print(f"分词后的块数: {len(outputs['input_ids'])}")
print(f"各块长度: {outputs['length']}")
print(f"块到样本的映射: {outputs['overflow_to_sample_mapping']}")

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

# ========================================
# 步骤9: 对整个数据集进行分词
# ========================================

print("\n" + "=" * 60)
print("步骤9: 对整个数据集进行分词预处理")
print("=" * 60)

tokenized_datasets = raw_datasets.map(
    tokenize,
    batched=True,  # 批量处理
    remove_columns=raw_datasets["train"].column_names  # 移除原始列
)

print("分词后的数据集:")
print(tokenized_datasets)

# ========================================
# 步骤10: 配置 GPT-2 模型
# ========================================

print("\n" + "=" * 60)
print("步骤10: 配置 GPT-2 语言模型")
print("=" * 60)

from transformers import GPT2LMHeadModel, AutoConfig

config = AutoConfig.from_pretrained(
    "gpt2",
    vocab_size=len(tokenizer),  # 使用分词器的词汇表大小
    n_ctx=context_length,  # 上下文长度
    bos_token_id=tokenizer.bos_token_id,  # 开始token ID
    eos_token_id=tokenizer.eos_token_id,  # 结束token ID
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

# 测试数据整理器
out = data_collator([tokenized_datasets["train"][i] for i in range(5)])
print("\n数据整理器测试（5个样本）:")
for key in out:
    print(f"  {key} shape: {out[key].shape}")

# ========================================
# 步骤13: 提取关键词 token IDs（用于加权损失）
# ========================================

print("\n" + "=" * 60)
print("步骤13: 提取数据科学关键词的 token IDs")
print("=" * 60)

# 定义要强调的关键词（数据科学常用函数/库名）
# 这些关键词在损失计算中会被加权，使模型更关注它们
keytoken_ids = []
for keyword in [
    "plt",  # matplotlib 绘图
    "pd",  # pandas
    "sk",  # sklearn
    "fit",  # 模型训练
    "predict",  # 预测
    " plt",  # 带空格的 plt
    " pd",  # 带空格的 pd
    " sk",  # 带空格的 sk
    " fit",  # 带空格的 fit
    " predict",  # 带空格的 predict
]:
    ids = tokenizer([keyword]).input_ids[0]
    if len(ids) == 1:
        keytoken_ids.append(ids[0])
    else:
        print(f"关键词 '{keyword}' 不是单个 token，跳过")

print(f"提取到的关键 token IDs: {keytoken_ids}")
print(f"关键词数量: {len(keytoken_ids)}")

# ========================================
# 步骤14: 定义关键词加权损失函数
# ========================================

print("\n" + "=" * 60)
print("步骤14: 定义关键词加权损失函数")
print("=" * 60)

from torch.nn import CrossEntropyLoss
import torch


def keytoken_weighted_loss(inputs, logits, keytoken_ids, alpha=1.0):
    """
    关键词加权损失函数：
    - 对包含关键 token 的样本给予更高的权重
    - alpha: 权重放大系数

    计算流程：
    1. 左移 tokens（预测下一个 token）
    2. 计算每个位置的交叉熵损失
    3. 统计样本中关键 token 的数量
    4. 根据关键 token 数量调整权重
    5. 计算加权平均损失
    """
    # 左移 tokens: position n predicts token n+1
    shift_labels = inputs[..., 1:].contiguous()
    shift_logits = logits[..., :-1, :].contiguous()

    # 计算每一个 token 的损失（不进行汇总）
    # BUG修复: reduce=False 已废弃，使用 reduction='none'
    loss_fct = CrossEntropyLoss(reduction='none')
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

    # 对于每个样本重新调整大小并平均
    loss_per_sample = loss.view(shift_logits.size(0), shift_logits.size(1)).mean(axis=1)

    # 计算每个样本中关键 token 的出现次数
    weights = torch.stack([(inputs == kt).float() for kt in keytoken_ids]).sum(
        axis=[0, 2]
    )

    # 缩放权重：包含关键 token 的样本权重更高
    weights = alpha * (1.0 + weights)

    # 计算加权平均损失
    weighted_loss = (loss_per_sample * weights).mean()
    return weighted_loss


print("关键词加权损失函数定义完成！")

# ========================================
# 步骤15: 创建 DataLoader
# ========================================

print("\n" + "=" * 60)
print("步骤15: 创建训练和验证 DataLoader")
print("=" * 60)

from torch.utils.data.dataloader import DataLoader

# 设置数据格式为 PyTorch tensor
tokenized_datasets.set_format("torch")

# 创建 DataLoader（注意：这里使用 data_collator 整理数据）
train_dataloader = DataLoader(
    tokenized_datasets["train"],
    batch_size=32,
    shuffle=True,
    collate_fn=data_collator  # 使用数据整理器
)
eval_dataloader = DataLoader(
    tokenized_datasets["valid"],
    batch_size=32,
    collate_fn=data_collator  # 使用数据整理器
)

print(f"训练 DataLoader 批次数: {len(train_dataloader)}")
print(f"验证 DataLoader 批次数: {len(eval_dataloader)}")

# ========================================
# 步骤16: 定义参数分组函数（用于差异化学习率）
# ========================================

print("\n" + "=" * 60)
print("步骤16: 定义参数分组函数（差异化学习率）")
print("=" * 60)


# BUG修复: 原脚本缺少此函数定义
def get_grouped_params(model, no_decay=["bias", "LayerNorm.weight"]):
    """
    将模型参数分组：
    - 带衰减的参数：权重矩阵等
    - 不带衰减的参数：bias、LayerNorm 等

    这样可以对不同类型的参数使用不同的权重衰减策略
    """
    params_with_wd, params_without_wd = [], []
    for n, p in model.named_parameters():
        if any(nd in n for nd in no_decay):
            params_without_wd.append(p)
        else:
            params_with_wd.append(p)

    return [
        {"params": params_with_wd, "weight_decay": 0.1},
        {"params": params_without_wd, "weight_decay": 0.0},
    ]


print("参数分组函数定义完成！")

# ========================================
# 步骤17: 初始化 Accelerator 和优化器
# ========================================

print("\n" + "=" * 60)
print("步骤17: 初始化 Accelerator 和优化器")
print("=" * 60)

from accelerate import Accelerator
from torch.optim import AdamW

# 初始化 Accelerator（支持多 GPU、混合精度）
import torch

if torch.backends.mps.is_available():
    # Mac MPS 设备：禁用混合精度，使用 bf16 或 no
    accelerator = Accelerator(mixed_precision="no")
    print("检测到 MPS 设备，禁用混合精度（fp16 在 MPS 上不稳定）")
else:
    # CUDA 设备：可以使用 fp16
    accelerator = Accelerator(mixed_precision="fp16")
    print("使用 fp16 混合精度")

# 初始化优化器（使用参数分组）
optimizer = AdamW(get_grouped_params(model), lr=5e-4)

print(f"Accelerator 设备: {accelerator.device}")
print(f"混合精度模式: fp16")

# ========================================
# 步骤18: 准备模型和数据加载器
# ========================================

print("\n" + "=" * 60)
print("步骤18: 使用 Accelerator 准备训练组件")
print("=" * 60)

model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)

print("模型、优化器和 DataLoader 已准备好！")

# ========================================
# 步骤19: 创建学习率调度器
# ========================================

print("\n" + "=" * 60)
print("步骤19: 创建学习率调度器")
print("=" * 60)

# BUG修复: 导入 get_scheduler
from transformers import get_scheduler

num_train_epochs = 1
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    name="linear",
    optimizer=optimizer,
    num_warmup_steps=1_000,
    num_training_steps=num_training_steps,
)

print(f"总训练步数: {num_training_steps}")
print(f"预热步数: 1_000")
print(f"调度器类型: linear")

# ========================================
# 步骤20: 定义评估函数
# ========================================

print("\n" + "=" * 60)
print("步骤20: 定义评估函数")
print("=" * 60)


def evaluate():
    """
    评估函数：计算验证集的损失和困惑度

    困惑度 = exp(loss)，表示模型对下一个 token 的不确定性
    困惑度越低，模型预测越准确
    """
    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            # 注意：batch 已经通过 data_collator 整理，包含 input_ids 和 labels
            outputs = model(batch["input_ids"], labels=batch["input_ids"])
        losses.append(accelerator.gather(outputs.loss))

    loss = torch.mean(torch.cat(losses))
    try:
        perplexity = torch.exp(loss)
    except OverflowError:
        perplexity = float("inf")

    model.train()  # 恢复训练模式
    return loss.item(), perplexity.item()


print("评估函数定义完成！")

# ========================================
# 步骤21: 配置模型保存（使用 HuggingFace Hub 新 API）
# ========================================

print("\n" + "=" * 60)
print("步骤21: 配置模型保存路径")
print("=" * 60)

from huggingface_hub import create_repo, get_full_repo_name

model_name = "codeparrot-ds-accelerate"
output_dir = "codeparrot-ds-accelerate"

try:
    repo_name = get_full_repo_name(model_name)
    print(f"Hub 仓库名: {repo_name}")

    # BUG修复: Repository 已废弃，使用 create_repo
    create_repo(repo_name, exist_ok=True)
    print("Hub 仓库创建成功！")
except Exception as e:
    print(f"Hub 配置失败（可继续本地训练）: {e}")

# ========================================
# 步骤22: 开始训练循环
# ========================================

print("\n" + "=" * 60)
print("步骤22: 开始训练循环")
print("=" * 60)
print("注意: 训练可能需要数小时，取决于硬件配置")
print("=" * 60)

# BUG修复: 使用标准 tqdm，而非 tqdm.notebook（仅限 Jupyter）
from tqdm import tqdm as std_tqdm

# 训练参数
gradient_accumulation_steps = 8  # 梯度累积步数
eval_steps = 5_000  # 每5000步评估一次
# BUG修复: 定义 samples_per_step
batch_size = 32
samples_per_step = batch_size * gradient_accumulation_steps

# 初始化训练状态
completed_steps = 0
model.train()

# 训练循环
for epoch in range(num_train_epochs):
    print(f"\n开始 Epoch {epoch + 1}/{num_train_epochs}")

    for step, batch in std_tqdm(
            enumerate(train_dataloader, start=1),
            total=num_training_steps,
            desc="训练进度"
    ):
        # 前向传播
        logits = model(batch["input_ids"]).logits

        # 计算关键词加权损失
        loss = keytoken_weighted_loss(batch["input_ids"], logits, keytoken_ids)

        # 每100步打印训练状态
        if step % 100 == 0:
            accelerator.print(
                {
                    "samples": step * samples_per_step,
                    "steps": completed_steps,
                    "loss/train": loss.item() * gradient_accumulation_steps,
                }
            )

        # 梯度累积：损失除以累积步数
        loss = loss / gradient_accumulation_steps
        accelerator.backward(loss)

        # 每累积步数更新一次参数
        if step % gradient_accumulation_steps == 0:
            # 梯度裁剪（防止梯度爆炸）
            accelerator.clip_grad_norm_(model.parameters(), 1.0)

            # 更新参数
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            completed_steps += 1

        # 定期评估和保存
        if (step % (eval_steps * gradient_accumulation_steps)) == 0:
            eval_loss, perplexity = evaluate()
            accelerator.print(f"\n评估结果 - Loss: {eval_loss:.4f}, Perplexity: {perplexity:.2f}")

            # 等待所有进程同步
            accelerator.wait_for_everyone()

            # 保存模型
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)

            if accelerator.is_main_process:
                tokenizer.save_pretrained(output_dir)
                print(f"模型已保存到 {output_dir}")

print("\n训练完成！")

# ========================================
# 步骤23: 最终评估和总结
# ========================================

print("\n" + "=" * 60)
print("步骤23: 最终评估")
print("=" * 60)

final_loss, final_perplexity = evaluate()
print(f"最终验证 Loss: {final_loss:.4f}")
print(f"最终困惑度: {final_perplexity:.2f}")

# ========================================
# 总结
# ========================================

print("\n" + "=" * 60)
print("脚本执行总结")
print("=" * 60)

print("""
本脚本使用 Accelerate 库训练代码生成模型，特点如下:

关键技术:
1. 自定义损失函数: 关键词加权损失，强调数据科学关键词
2. 梯度累积: 有效批次 = 32 * 8 = 256
3. 混合精度: fp16 加速训练
4. 参数分组: 对 bias/LayerNorm 不使用权重衰减
5. 学习率调度: linear 调度器 + warmup

Bug修复说明:
- CrossEntropyLoss: reduce=False → reduction='none'
- 导入 get_scheduler: 从 transformers 导入
- 定义 get_grouped_params: 参数分组函数
- 定义 samples_per_step: 用于日志打印
- DataLoader: 使用 data_collator 整理数据
- Repository → create_repo: 使用新 API
- tqdm.notebook → tqdm: 支持非 Jupyter 环境
- 模型不重复初始化: 使用已创建的模型
""")
print("=" * 60)
print("所有步骤完成！")
print("=" * 60)
