### 阅读笔记

#### 总结

#### 5.DATASETS 库
- **如果我的数据集不在 hub 上怎么办？ 用datasets工具进行加载，比较简单**
- 支持几种常见的数据格式

| 类型参数 | 加载的指令 |
|-----------|------------|
| CSV & TSV | `load_dataset("csv", data_files="my_file.csv")` |
| Text files | `load_dataset("text", data_files="my_file.txt")` |
| JSON & JSON Lines | `load_dataset("json", data_files="my_file.jsonl")` |
| Pickled DataFrames | `load_dataset("pandas", data_files="my_dataframe.pkl")` |

```Python
from dataclasses import field

from datasets import load_dataset

# 加载本地数据集

# 加载单个文件，测试 ok 的
# squad_it_dataset = load_dataset("json", data_files="./download/SQuAD_it-train.json", field="data")
#
# print(squad_it_dataset)
# print("加载完成")

# 加载多个文件
# data_files = {"train": "./download/SQuAD_it-train.json", "test": "./download/SQuAD_it-test.json"}
# squad_it_dataset = load_dataset("json", data_files=data_files, field="data")
# print(squad_it_dataset)
# print("加载完成")

# 从压缩包加载多个本地文件
data_files = {"train": "./download/SQuAD_it-train.json.gz", "test": "./download/SQuAD_it-test.json.gz"}
squad_it_dataset = load_dataset("json", data_files=data_files, field="data")
print(squad_it_dataset)

# 加载远程数据集
url = "https://github.com/crux82/squad-it/raw/master/"
data_files = {
    "train": url + "SQuAD_it-train.json.gz",
    "test": url + "SQuAD_it-test.json.gz",
}
squad_it_dataset = load_dataset("json", data_files=data_files, field="data")
```

- **是时候来学一下切片了**


#### 6.TOKENIZERS 库

#### 7.主要的 NLP 任务

#### 8.如何寻求帮助

#### 9.构建并分享你的模型