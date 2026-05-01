"""
使用示例：python deepseek_tokenizer.py "hello, world"

不支持中文的检测，因为该版本的词表 tokenizer.json中不包含中文词元（token）
一般情况下模型中 token 和字数的换算比例大致如下：

1 个英文字符 ≈ 0.3 个 token。
1 个中文字符 ≈ 0.6 个 token。
"""
import sys
import transformers

chat_tokenizer_dir = "./"

tokenizer = transformers.AutoTokenizer.from_pretrained(
    chat_tokenizer_dir, trust_remote_code=True
)

text = sys.argv[1]
result = tokenizer.encode(text)
print(f"token counts: {len(result)}")
