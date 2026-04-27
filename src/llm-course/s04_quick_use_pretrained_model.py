# 导入 pipeline 函数，这是 Hugging Face 提供的高层 API，用于快速加载预训练模型
from transformers import pipeline

# - 创建一个"填空"类型的 pipeline
# - model="camembert-base" 指定使用 CamemBERT 模型，这是一个专门针对法语训练的 BERT 模型
camembert_fill_mask = pipeline("fill-mask", model="camembert-base")
# 输入法语句子，其中 <mask> 是需要预测的词位
results = camembert_fill_mask("Le camembert est <mask> :)")
# 输出预测结果，通常返回一个列表，包含多个候选词及其概率
print(results)
"""
[{'score': 0.49091655015945435, 'token': 7200, 'token_str': 'délicieux', 'sequence': 'Le camembert est délicieux :)'}, 
{'score': 0.10557064414024353, 'token': 2183, 'token_str': 'excellent', 'sequence': 'Le camembert est excellent :)'}, 
{'score': 0.03453359007835388, 'token': 26202, 'token_str': 'succulent', 'sequence': 'Le camembert est succulent :)'}, 
{'score': 0.03303172439336777, 'token': 528, 'token_str': 'meilleur', 'sequence': 'Le camembert est meilleur :)'}, 
{'score': 0.030076900497078896, 'token': 1654, 'token_str': 'parfait', 'sequence': 'Le camembert est parfait :)'}]

"""

# 还可以使用模型架构实例化 checkpoint，建议使用 Auto*类，因为它们在设计时不依赖模型架构
from transformers import AutoTokenizer, AutoModelForMaskedLM
tokenizer = AutoTokenizer.from_pretrained("camembert-base")
model = AutoModelForMaskedLM.from_pretrained("camembert-base")