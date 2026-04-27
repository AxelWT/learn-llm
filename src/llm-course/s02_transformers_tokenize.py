from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
sequence = "Using a Transformer network is simple"
tokens = tokenizer.tokenize(sequence)
ids = tokenizer.convert_tokens_to_ids(tokens)
decoded_string = tokenizer.decode(ids)
print(tokens)
print(ids)
print(decoded_string)
print(f"词汇表大小: {len(tokenizer.vocab)}")
print(f"词汇表内容示例: {list(tokenizer.vocab.keys())[:20]}")

"""
结果：
['Using', 'a', 'Trans', '##former', 'network', 'is', 'simple']
[7993, 170, 13809, 23763, 2443, 1110, 3014]
Using a Transformer network is simple
词汇表大小: 28996
词汇表内容示例: ['##up', 'applying', '##ample', 'replaced', 'jumper', 'destroyed', 'convened', 'iPhone', 'therefore', '##master', '##本', 'Bosnia', '##utile', 'spaced', '##roon', 'tensions', 'brows', '##ogan', 'altitude', 'Karnataka']
"""
