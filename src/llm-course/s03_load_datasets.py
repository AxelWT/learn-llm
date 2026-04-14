from datasets import load_dataset
from transformers import AutoTokenizer

raw_datasets = load_dataset("glue", "mrpc")
print(raw_datasets)

print(raw_datasets["train"][15])
print(raw_datasets["validation"][87])