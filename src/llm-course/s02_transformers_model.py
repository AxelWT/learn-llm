import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
sequences = ["I've been waiting for a HuggingFace course my whole life.", "So have I!"]

# tokenizer()函数内部自动做了填充（保证不同句子分词后的 token 列表长度一致）
# tokenizer()函数内部自动做了截断（保证输入不会超过模型可以接受的最大 token 数量）
# tokenizer()函数内部自动做了张量转换，先把句子分词，再把分词映射为数字，再把数字转换为 pytorch tensor，因为模型只接受张量 tensor 输入
tokens = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
print(tokens)
output = model(**tokens)
print(output)

"""
结果：
{'input_ids': 
tensor([[  101,  1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662, 12172, 2607,  2026,  2878,  2166,  1012,   102],
        [  101,  2061,  2031,  1045,   999,   102,     0,     0,     0,     0, 0,     0,     0,     0,     0,     0],
        [  101,  7592,  2088,   102,     0,     0,     0,     0,     0,     0, 0,     0,     0,     0,     0,     0]]), 
'token_type_ids': 
tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]), 
'attention_mask': 
tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])}

SequenceClassifierOutput(loss=None, 
logits=tensor([[-1.5607,  1.6123],
        [-3.6183,  3.9137],
        [-3.9943,  4.3083]], 
        grad_fn=<AddmmBackward0>), hidden_states=None, attentions=None)
"""
