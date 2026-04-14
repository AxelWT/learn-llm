# 在不使用 Trainer 类的情况下实现一样的训练步骤和效果

from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 以上是数据准备阶段

# 删除模型不需要的列
tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
# 将列名 label 重命名为 labels，因为模型默认的输入是 labels
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
# 设置数据集的格式，使其返回 pytorch 张量而不是列表
tokenized_datasets.set_format("torch")
print(tokenized_datasets["train"].column_names)

from torch.utils.data import DataLoader

# 定义训练数据加载器
train_dataloader = DataLoader(
    tokenized_datasets["train"], shuffle=True, batch_size=8, collate_fn=data_collator
)
# 第一个返回的是 batch 数量（注意每 batch 是 8 条数据），第二个返回的是总样本数量3668条，训练完所有 459个 batch 算作一次 epoch
print(len(train_dataloader), len(tokenized_datasets["train"]))
# 定义评估数据加载器
eval_dataloader = DataLoader(
    tokenized_datasets["validation"], batch_size=8, collate_fn=data_collator
)
# 快速验证数据处理中有没有错误，可以检验其中的一个 batch
for batch in train_dataloader:
    print({k: v.shape for k, v in batch.items()})
    break

from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
# 传入一个 batch 的数据到模型中测试一下，batch 中有 labels 时，transformer 模型都将返回这个 batch 的 loss
for batch in train_dataloader:
    outputs = model(**batch)
    print(outputs.loss, outputs.logits.shape)
    print(outputs)
    break

# 训练过程
#   1 epoch:
#   ├── batch 1: 8 样本   → 前向传播 → 计算损失 → 反向传播 → 更新权重
#   ├── batch 2: 8 样本   → 前向传播 → 计算损失 → 反向传播 → 更新权重
#   ├── ...
#   ├── batch 458: 8 样本 → 前向传播 → 计算损失 → 反向传播 → 更新权重
#   └── batch 459: 4 样本 → 前向传播 → 计算损失 → 反向传播 → 更新权重
#                       ↑
#                 这就是 1 个 epoch 完成

# 一个 Batch 是一起训练的
#
#   所有样本同时进行前向传播、计算损失、反向传播。
#
#   训练过程
#   一个 batch (8条数据):
#
#   前向传播:
#   ┌─────────────────────────────────────┐
#   │ 数据1 ─┐                              │
#   │ 数据2 ─┤                              │
#   │ 数据3 ─┼──→ 模型 ──→ 8个输出 ──→ 1个loss │
#   │ ...   ─┤      (并行计算)              │
#   │ 数据8 ─┘                              │
#   └─────────────────────────────────────┘
#               ↓
#   反向传播: 更新一次参数
#
#   代码验证
#
#   for batch in train_dataloader:
#       # batch 包含 8 条数据，形状 [8, 81]
#       outputs = model(**batch)  # 8条数据一起前向传播
#
#       print(outputs.logits.shape)  # torch.Size([8, 2]) - 8条数据的预测结果
#       print(outputs.loss)          # 一个标量 - 8条数据的平均损失
#
#       loss.backward()              # 基于这个平均损失反向传播
#       optimizer.step()             # 更新一次参数
#
#   对比两种方式
#
#   ┌────────────┬───────────────────────┬────────────────────────────┐
#   │    方式     │     参数更新次数        │            特点            │
#   ├────────────┼───────────────────────┼────────────────────────────┤
#   │ Batch 训练  │ 3668/8 = 459 次/epoch │ 并行计算，速度快，梯度稳定     │
#   ├────────────┼───────────────────────┼────────────────────────────┤
#   │ 逐条训练    │ 3668 次/epoch          │ 串行计算，慢，梯度波动大       │
#   └────────────┴───────────────────────┴────────────────────────────┘
#
#   为什么一起训练
#
#   # 损失是 batch 内所有样本的平均
#   loss = (loss_样本1 + loss_样本2 + ... + loss_样本8) / 8
#
#   # 优点:
#   # 1. GPU 并行计算，速度快
#   # 2. 平均梯度更稳定，训练更平滑
#   # 3. 充分利用 GPU 显存带宽
#
#   总结
#
#   ┌──────────┬────────────────────────────────────┐
#   │   概念   │                说明                │
#   ├──────────┼────────────────────────────────────┤
#   │ Batch 内 │ 8 条数据同时计算，产生一个平均损失 │
#   ├──────────┼────────────────────────────────────┤
#   │ 参数更新 │ 每个 batch 结束后更新一次          │
#   ├──────────┼────────────────────────────────────┤
#   │ 1 epoch  │ 参数更新 459 次（= batch 数量）    │
#   └──────────┴────────────────────────────────────┘


# 优化器（使用 AdamW）和学习率调度器
#   AdamW vs Adam：
#   - Adam：经典自适应学习率优化器
#   - AdamW：带权重衰减的 Adam，正则化效果更好，是 BERT 训练的标准选择
from torch.optim import AdamW

# model.parameters()是模型所有可训练参数，lr=5e-5学习率 0.00005（BERT 常用的小学习率）
optimizer = AdamW(model.parameters(), lr=5e-5)

# 学习率调度器
from transformers import get_scheduler

num_epoches = 3
# 训练步数 = epoch 数量 * batch 批次数（一个 batch 全丢进去训练一次）= 3 * 459 = 1377【总训练步数 1377 整个训练过程参数更新 1377 次】
num_training_steps = num_epoches * len(train_dataloader)
print(num_training_steps)

# "linear" # 调度策略
# 学习率变化曲线（linear策略）：
#
# 学习率
# │
# 5e-5 ─────┐
#           │ \
#           │  \
#           │   \
#           │    \
#           │     \
#           │      \
#      ─────└──────└────→ 训练步数
#           0
#                 1377
#
# 对比不同调度策略：
#
# ┌────────────────────────┬──────────────┬──────────────────────┐
# │          策略           │   曲线形状    │         特点          │
# ├────────────────────────┼──────────────┼──────────────────────┤
# │ "linear"               │ 线性下降到0    │ BERT常用，简单有效     │
# ├────────────────────────┼──────────────┼──────────────────────┤
# │ "cosine"               │ 余弦曲线下降   │ 平滑，后期下降慢        │
# ├────────────────────────┼──────────────┼──────────────────────┤
# │ "cosine_with_restarts" │ 余弦重启      │ 多次周期，适合长训练     │
# ├────────────────────────┼──────────────┼──────────────────────┤
# │ "constant"             │ 保持不变      │ 学习率始终5e-5         │
# └────────────────────────┴──────────────┴──────────────────────┘

# num_warmup_steps=0 # 预热步数，设置为 0 则不使用预热，预热的作用： 训练初期学习率从小逐渐增大，避免一开始大学习率破坏预训练权重。
# num_training_steps=num_training_steps  # 总训练步数
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps
)
# 总结
#
# ┌────────────────────┬──────────────────────────────────────┐
# │        组件         │                 作用                  │
# ├────────────────────┼──────────────────────────────────────┤
# │ AdamW              │ 优化器，决定如何更新参数                 │
# ├────────────────────┼──────────────────────────────────────┤
# │ num_training_steps │ 总训练步数 = epoch × batch数           │
# ├────────────────────┼──────────────────────────────────────┤
# │ lr_scheduler       │ 学习率调度器，让学习率随训练逐渐降低       │
# └────────────────────┴──────────────────────────────────────┘
#
# 为什么需要学习率调度？
# - 训练初期：需要较大学习率快速学习
# - 训练后期：需要小学习率精细调整，避免震荡

# 访问 GPU 设置，目的是加快训练（不过我这台机器目前没有 cuda😭）
# import torch
# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# model.to(device)

# 训练循环♻️

# 使用 tqdm 库，在训练步骤数上添加一个进度条
from tqdm.auto import tqdm

progress_bar = tqdm(range(num_training_steps))
# 设置训练模式
# ┌───────────────┬────────────────────────────────────────┐
# │     模式       │                  作用                  │
# ├───────────────┼────────────────────────────────────────┤
# │ model.train() │ 训练模式，启用Dropout、BatchNorm更新      │
# ├───────────────┼────────────────────────────────────────┤
# │ model.eval()  │ 评估模式，关闭Dropout、固定BatchNorm      │
# └───────────────┴────────────────────────────────────────┘
model.train()

for epoch in range(num_epoches):
    for batch in train_dataloader:
        # 使用 GPU 加速，因为本机配置问题，暂不用
        # batch = {k: v.to(device) for k, v in batch.items()}
        # 前向传播
        # **是 Python 的字典解包语法，将字典的键值对展开为函数的关键字参数
        outputs = model(**batch)
        # 输出一个 batch 的平均损失值
        loss = outputs.loss
        # 反向传播，计算梯度（累加到现有梯度）
        loss.backward()
        # 用梯度更新参数
        optimizer.step()
        # 更新学习率
        lr_scheduler.step()
        # 清零梯度，准备下一轮
        optimizer.zero_grad()
        # 训练完一个 batch 更新进度条
        progress_bar.update(1)

# 上述训练循环不会告诉我们任何关于模型目前的状态，我们需要为此添加一个评估循环
# 评估循环
# evaluate 库，这是 Hugging Face 开发的评估库（原名 datasets.metrics），专门用于计算机器学习模型的各种评估指标。
import evaluate, torch

# "glue" │ 基准测试名称（GLUE 是一个包含多个 NLP 任务的评测基准）
# "mrpc" │ 具体任务名称（MRPC = Microsoft Research Paraphrase Corpus）
# 加载 GLUE 基准测试中的 MRPC 任务的评估指标。MRPC 是微软研究院的同义改写语料库，用于判断两个句子是否表达相同意思。指标会计算准确率和 F1 分数。
metric = evaluate.load("glue", "mrpc")
# 设置模型的训练模式为评估模式
model.eval()
for batch in eval_dataloader:
    # batch = {k: v.to(device) for k, v in batch.items()}
    # 遍历数据批次。torch.no_grad() 禁用梯度计算，节省内存并加速推理。
    with torch.no_grad():
        outputs = model(**batch)

    logits = outputs.logits
    # 提取模型的原始输出，然后用 argmax 取最后一维的最大值索引，得到预测类别（0 或 1）。【dim=-1 就是对每一行找最大值的列索引】
    predictions = torch.argmax(logits, dim=-1)
    # 将预测结果和真实标签累积到指标对象中。
    metric.add_batch(predictions=predictions, references=batch["labels"])

# 打印评估结果
print(metric.compute())
# 微调模型保存
model.save_pretrained("./test-trainer")
tokenizer.save_pretrained("./test-trainer")

# 显式清理资源，避免 Python 退出时崩溃
import gc

del model, optimizer, lr_scheduler
del train_dataloader, eval_dataloader
del tokenized_datasets, raw_datasets

gc.collect()
gc.collect()
print("训练完成，资源已清理")

# # 评估示例数据：
# labels = torch.tensor([1, 0])
#
# 完整示例：
#
# batch = {
#     "input_ids": torch.tensor([
#         [101, 2023, 2003, ...],
#         # 样本0的输入token
#         [101, 2054, 2856, ...]
#         # 样本1的输入token
#     ]),
#     # shape: [2, seq_len]
#
#     "attention_mask": torch.tensor([
#         [1, 1, 1, ...],
#         [1, 1, 0, ...]
#     ]),
#     # shape: [2, seq_len]
#
#     "labels": torch.tensor([1, 0])
#     # shape: [2]
# }
# #              ↑  ↑
# #              │  └── 样本1的真实标签：0（不是同义改写）
# #              └───── 样本0的真实标签：1（是同义改写）
#
# predictions
# 和
# labels
# 的对应关系：
#
# predictions = torch.tensor([1,
#                             0])  # 模型预测
# labels = torch.tensor([1,
#                        0])  # 真实标签
#
# # 对比：
# # 样本0：预测1，真实1 → 正确 ✓
# # 样本1：预测0，真实0 → 正确 ✓
#
# # 这批样本全部预测正确
#
# 另一个例子（有错误预测）：
#
# predictions = torch.tensor([1, 0, 1,
#                             0])  # 模型预测
# labels = torch.tensor([1, 1, 1,
#                        0])  # 真实标签
#
# # 对比：
# # 样本0：预测1，真实1 → 正确 ✓
# # 样本1：预测0，真实1 → 错误 ✗
# # 样本2：预测1，真实1 → 正确 ✓
# # 样本3：预测0，真实0 → 正确 ✓
#
# MRPC
# 标签含义：
# - 0 = 两个句子不是同义改写
# - 1 = 两个句子是同义改写
