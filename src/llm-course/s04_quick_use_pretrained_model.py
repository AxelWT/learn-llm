# 导入 pipeline 函数，这是 Hugging Face 提供的高层 API，用于快速加载预训练模型
from transformers import pipeline

# - 创建一个"填空"类型的 pipeline
# - model="camembert-base" 指定使用 CamemBERT 模型，这是一个专门针对法语训练的 BERT 模型
camembert_fill_mask = pipeline("fill-mask", model="camembert-base")
# 输入法语句子，其中 <mask> 是需要预测的词位
results = camembert_fill_mask("Le camembert est <mask> :)")
# 输出预测结果，通常返回一个列表，包含多个候选词及其概率
print(results)

# 还可以使用模型架构实例化 checkpoint，建议使用 Auto*类，因为它们在设计时不依赖模型架构
from transformers import AutoTokenizer, AutoModelForMaskedLM
tokenizer = AutoTokenizer.from_pretrained("camembert-base")
model = AutoModelForMaskedLM.from_pretrained("camembert-base")