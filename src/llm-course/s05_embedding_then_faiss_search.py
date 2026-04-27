"""
GitHub Issues 数据集处理脚本
功能：从 HuggingFace Hub 加载 GitHub issues 数据，筛选并转换为可用于嵌入检索的文本格式
"""

from datasets import load_dataset

# 从 HuggingFace Hub 加载 lewtun/github-issues 数据集，使用 train（字段）分割
issues_dataset = load_dataset("axelloo/github-issues", split="train")

# 筛选数据集：
# - 只保留 issues（排除 pull requests）- pull_request 字段为 None 表示是 issue
# - 只保留有评论的 issues
issues_dataset = issues_dataset.filter(
    lambda x: (x["pull_request"] is None and len(x["comments"]) > 0)
)

print(issues_dataset)
"""
Dataset({
    features: ['url', 'repository_url', 'labels_url', 'comments_url', 'events_url', 'html_url', 'id', 'node_id', 'number', 'title', 'user', 'labels', 'state', 'locked', 'assignees', 'milestone', 'comments', 'created_at', 'updated_at', 'closed_at', 'author_association', 'type', 'active_lock_reason', 'draft', 'pull_request', 'body', 'closed_by', 'reactions', 'timeline_url', 'performed_via_github_app', 'state_reason', 'sub_issues_summary', 'issue_dependencies_summary', 'pinned_comment'],
    num_rows: 247
})
"""

# 获取所有列名，确定需要保留和删除的列
columns = issues_dataset.column_names
columns_to_keep = ["title", "body", "html_url", "comments"]
# 使用对称差集计算需要删除的列（即不在 columns_to_keep 中的列）
columns_to_remove = set(columns_to_keep).symmetric_difference(columns)
issues_dataset = issues_dataset.remove_columns(columns_to_remove)

# 将数据集转换为 pandas DataFrame 格式以便处理
issues_dataset.set_format("pandas")
df = issues_dataset[:]  # 获取整个数据集

# 打印第一条记录的评论列表（用于调试查看数据结构）
print(df["comments"][0].tolist())
"""
['`datasets` 4.8.4 is out and includes a fix :)']
"""

# 将 comments 列"展开"（explode）：
# 原来每行可能有多条评论（列表形式），展开后每条评论变成单独一行
# ignore_index=True 表示重置索引
comments_df = df.explode("comments", ignore_index=True)
# 显示前4行（用于调试）
print(comments_df.head(4))
"""
                                            html_url  ...                                               body
0  https://github.com/huggingface/datasets/issues...  ...  ### Describe the bug\n\nFor PyTorch 2.11 + tor...
1  https://github.com/huggingface/datasets/issues...  ...  The `.batch()` method currently assumes the in...
2  https://github.com/huggingface/datasets/issues...  ...  The `.batch()` method currently assumes the in...
3  https://github.com/huggingface/datasets/issues...  ...  ### Describe the bug\n\nFor PyTorch 2.11 + tor...

[4 rows x 4 columns]
"""

from datasets import Dataset

# 将 pandas DataFrame 转换回 HuggingFace Dataset 格式
comments_dataset = Dataset.from_pandas(comments_df)
print(comments_dataset)
"""
Dataset({
    features: ['html_url', 'title', 'comments', 'body'],
    num_rows: 851
})
"""
# 计算每条评论的词数，并筛选出词数大于15的评论
# 注意：这里假设 comments 字段是字符串类型
comments_dataset = comments_dataset.map(
    lambda x: {"comment_length": len(x["comments"].split())}
).filter(lambda x: x["comment_length"] > 15)


def concatenate_text(examples):
    """
    将 title、body 和 comments 拼接成一个完整的文本字段
    用于后续的嵌入检索或文本处理
    """
    title = examples["title"] or ""
    body = examples["body"] or ""
    comments = examples["comments"] or ""
    return {
        "text": title + " \n " + body + " \n " + comments
    }


# 对数据集应用文本拼接函数
comments_dataset = comments_dataset.map(concatenate_text)
print(comments_dataset)
print(comments_dataset[0])
"""
Dataset({
    features: ['html_url', 'title', 'comments', 'body', 'comment_length', 'text'],
    num_rows: 616
})
{'html_url': 'https://github.com/huggingface/datasets/issues/8075', 
'title': '`.batch()` error on formatted datasets', 
'comments': 'Hi ! Good catch :) Since table-formatted iterable datasets have an Arrow path (i.e. they have `.iter_arrow()`) I guess a simple fix would be to make `.batch()` make the dataset arrow-formatted and use a `batch_arrow_fn` in that case (instead of the current `_batch_fn` that expects dictionaries). It would also be better performance-wise. How does that sound ?', 
'body': 'The `.batch()` method currently assumes the input (batch) is always a dictionary, which causes errors when it isn\'t. This can happen with formatted datasets, since formats like `"pyarrow"`, `"pandas"` (only affects `IterableDataset`), and `"polars"` return tables/dataframes instead of dictionaries.\n\nFor example:\n```python\nfrom datasets import IterableDataset, Dataset\nlist(IterableDataset.from_dict({"a": [1, 2, 3, 4]}).with_format("pyarrow").batch(2))\n# AttributeError: \'pyarrow.lib.Table\' object has no attribute \'items\'\n```\n\nIdeally, the result should be the same whether the format is applied before or after batching, i.e., the following should hold for all the format types:\n```python\nassert list(IterableDataset.from_dict({"a": [1, 2, 3, 4]}).with_format(format_type).batch(2)) == list(IterableDataset.from_dict({"a": [1, 2, 3, 4]}).batch(2).with_format(format_type))\nassert list(Dataset.from_dict({"a": [1, 2, 3, 4]}).with_format(format_type).batch(2)) == list(Dataset.from_dict({"a": [1, 2, 3, 4]}).batch(2).with_format(format_type))\n```', 
'comment_length': 57, 
'text': '`.batch()` error on formatted datasets \n The `.batch()` method currently assumes the input (batch) is always a dictionary, which causes errors when it isn\'t. This can happen with formatted datasets, since formats like `"pyarrow"`, `"pandas"` (only affects `IterableDataset`), and `"polars"` return tables/dataframes instead of dictionaries.\n\nFor example:\n```python\nfrom datasets import IterableDataset, Dataset\nlist(IterableDataset.from_dict({"a": [1, 2, 3, 4]}).with_format("pyarrow").batch(2))\n# AttributeError: \'pyarrow.lib.Table\' object has no attribute \'items\'\n```\n\nIdeally, the result should be the same whether the format is applied before or after batching, i.e., the following should hold for all the format types:\n```python\nassert list(IterableDataset.from_dict({"a": [1, 2, 3, 4]}).with_format(format_type).batch(2)) == list(IterableDataset.from_dict({"a": [1, 2, 3, 4]}).batch(2).with_format(format_type))\nassert list(Dataset.from_dict({"a": [1, 2, 3, 4]}).with_format(format_type).batch(2)) == list(Dataset.from_dict({"a": [1, 2, 3, 4]}).batch(2).with_format(format_type))\n``` \n Hi ! Good catch :) Since table-formatted iterable datasets have an Arrow path (i.e. they have `.iter_arrow()`) I guess a simple fix would be to make `.batch()` make the dataset arrow-formatted and use a `batch_arrow_fn` in that case (instead of the current `_batch_fn` that expects dictionaries). It would also be better performance-wise. How does that sound ?'}

"""

# 加载 sentence-transformers 模型，用于生成文本嵌入向量
# multi-qa-mpnet-base-dot-v1 是一个专门针对问答检索任务优化的模型
from transformers import AutoTokenizer, AutoModel

model_ckpt = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)  # 加载分词器
model = AutoModel.from_pretrained(model_ckpt)  # 加载预训练模型


# 实现 CLS 池化，用于从 BERT 类模型的输出中提取句子级别的嵌入向量。
def cls_pooling(model_output):
    """
    1. model_output.last_hidden_state — 模型的隐藏层输出，形状为 (batch_size, seq_length, hidden_dim)。例如输入 2 个句子、每个 10 个 token，输出形状就是 (2, 10, 768)。
    2. [:, 0] — 取所有样本的第 0 个位置（即 [CLS] token）。结果是 (batch_size, hidden_dim)。
    为什么用 [CLS]？
    BERT 类模型在输入序列开头会添加一个特殊的 [CLS] token。经过 Transformer 的自注意力机制，[CLS] 会聚合整个序列的信息，因此它的嵌入向量常被用作整段文本的语义表示。

    举例：
    假设输入文本 "How can I load a dataset?" 经 tokenizer 后变成：
    [CLS] How can I load a dataset [SEP]
     0   1   2  3  4    5       6     7
    last_hidden_state[:, 0] 就提取位置 0 的向量，代表整个问题的语义，用于后续的相似度检索。
    """
    return model_output.last_hidden_state[:, 0]


def get_embeddings(text_list):
    """
    将文本列表转换为嵌入向量
    参数：text_list - 文本字符串列表
    返回：嵌入向量矩阵
    """
    # 使用 tokenizer 对文本进行编码
    # padding=True: 对短文本进行填充，使所有序列长度一致
    # truncation=True: 对超长文本进行截断
    # return_tensors="pt": 返回 PyTorch tensor 格式
    encoded_input = tokenizer(
        text_list, padding=True, truncation=True, return_tensors="pt"
    )
    # 将编码输入转换为字典格式（去除 batch 索引）
    encoded_input = {k: v for k, v in encoded_input.items()}
    # 将编码输入传入模型，获取输出
    model_output = model(**encoded_input)
    # 使用 CLS 池化提取嵌入向量
    return cls_pooling(model_output)


# 测试：对第一条文本生成嵌入向量，打印向量维度
embedding = get_embeddings(comments_dataset["text"][0])
print(embedding.shape)
"""
torch.Size([1, 768])
"""

# 对整个数据集计算嵌入向量
# 每条文本生成一个 768 维的嵌入向量（mpnet-base 模型的输出维度）
# .detach().cpu().numpy()[0] 用于将 PyTorch tensor 转换为 numpy 数组
embeddings_dataset = comments_dataset.map(
    lambda x: {"embeddings": get_embeddings(x["text"]).detach().cpu().numpy()[0]}
)
print(embeddings_dataset)
print(embeddings_dataset[0])
"""
Dataset({
    features: ['html_url', 'title', 'comments', 'body', 'comment_length', 'text', 'embeddings'],
    num_rows: 616
})
{'html_url': 'https://github.com/huggingface/datasets/issues/8075', 
'title': '`.batch()` error on formatted datasets', 
'comments': 'Hi ! Good catch :) Since table-formatted iterable datasets have an Arrow path (i.e. they have `.iter_arrow()`) I guess a simple fix would be to make `.batch()` make the dataset arrow-formatted and use a `batch_arrow_fn` in that case (instead of the current `_batch_fn` that expects dictionaries). It would also be better performance-wise. How does that sound ?', 
'body': 'The `.batch()` method currently assumes the input (batch) is always a dictionary, which causes errors when it isn\'t. This can happen with formatted datasets, since formats like `"pyarrow"`, `"pandas"` (only affects `IterableDataset`), and `"polars"` return tables/dataframes instead of dictionaries.\n\nFor example:\n```python\nfrom datasets import IterableDataset, Dataset\nlist(IterableDataset.from_dict({"a": [1, 2, 3, 4]}).with_format("pyarrow").batch(2))\n# AttributeError: \'pyarrow.lib.Table\' object has no attribute \'items\'\n```\n\nIdeally, the result should be the same whether the format is applied before or after batching, i.e., the following should hold for all the format types:\n```python\nassert list(IterableDataset.from_dict({"a": [1, 2, 3, 4]}).with_format(format_type).batch(2)) == list(IterableDataset.from_dict({"a": [1, 2, 3, 4]}).batch(2).with_format(format_type))\nassert list(Dataset.from_dict({"a": [1, 2, 3, 4]}).with_format(format_type).batch(2)) == list(Dataset.from_dict({"a": [1, 2, 3, 4]}).batch(2).with_format(format_type))\n```', 
'comment_length': 57, 
'text': '`.batch()` error on formatted datasets \n The `.batch()` method currently assumes the input (batch) is always a dictionary, which causes errors when it isn\'t. This can happen with formatted datasets, since formats like `"pyarrow"`, `"pandas"` (only affects `IterableDataset`), and `"polars"` return tables/dataframes instead of dictionaries.\n\nFor example:\n```python\nfrom datasets import IterableDataset, Dataset\nlist(IterableDataset.from_dict({"a": [1, 2, 3, 4]}).with_format("pyarrow").batch(2))\n# AttributeError: \'pyarrow.lib.Table\' object has no attribute \'items\'\n```\n\nIdeally, the result should be the same whether the format is applied before or after batching, i.e., the following should hold for all the format types:\n```python\nassert list(IterableDataset.from_dict({"a": [1, 2, 3, 4]}).with_format(format_type).batch(2)) == list(IterableDataset.from_dict({"a": [1, 2, 3, 4]}).batch(2).with_format(format_type))\nassert list(Dataset.from_dict({"a": [1, 2, 3, 4]}).with_format(format_type).batch(2)) == list(Dataset.from_dict({"a": [1, 2, 3, 4]}).batch(2).with_format(format_type))\n``` \n Hi ! Good catch :) Since table-formatted iterable datasets have an Arrow path (i.e. they have `.iter_arrow()`) I guess a simple fix would be to make `.batch()` make the dataset arrow-formatted and use a `batch_arrow_fn` in that case (instead of the current `_batch_fn` that expects dictionaries). It would also be better performance-wise. How does that sound ?',
'embeddings': [-0.40911683440208435, -0.1059914156794548, -0.05522923171520233, -0.0641660988330841, 0.11291451752185822, 0.08336920291185379, 0.463390976190567, 0.3956606388092041, -0.426876962184906, -0.07155115902423859, -0.06572770327329636, 0.47573480010032654, -0.19835688173770905, 0.11820483952760696, -0.1148185282945633, -0.1328001320362091, 0.08152212202548981, 0.15055111050605774, -0.07466396689414978, 0.03878754377365112, -0.19033733010292053, 0.08133885264396667, -0.4599868655204773, 0.22942528128623962, -0.21314433217048645, -0.2881758511066437, -0.12671679258346558, -0.1859939992427826, 0.0871286541223526, -0.7704533934593201, 0.3828081488609314, 0.12954838573932648, 0.3740334212779999, 0.6622698903083801, -0.00011272120173089206, -0.046194158494472504, 0.1864146739244461, -0.05348096787929535, -0.4844150245189667, -0.06087496876716614, -0.23521947860717773, -0.18493197858333588, 0.2199983447790146, -0.14592893421649933, -0.05526606738567352, -0.5972678661346436, -0.24288831651210785, -0.1606457531452179, 0.0378539115190506, 0.12746672332286835, 0.2100406140089035, 0.32998132705688477, -0.06503552198410034, 0.026027729734778404, 0.15730485320091248, 0.21658283472061157, -0.025716383010149002, 0.4210568368434906, 0.3901309072971344, -0.09501868486404419, 0.1241636574268341, 0.18036866188049316, -0.3338664770126343, 0.20260807871818542, -0.06366817653179169, 0.05272982642054558, 0.013951964676380157, -0.1316254734992981, -0.07426539063453674, 0.2758042812347412, 0.4868970513343811, -0.403902530670166, -0.5257353186607361, -0.3883194923400879, -0.030763130635023117, -0.3132793605327606, -0.20147162675857544, 0.17869015038013458, -0.051441874355077744, 0.014630984514951706, -0.07274423539638519, 0.189640611410141, -0.19236978888511658, -0.019875280559062958, -0.09078405052423477, 0.3942931592464447, -0.0760517418384552, 0.17314490675926208, -0.08642107248306274, -0.08095592260360718, 0.19773051142692566, -0.17343057692050934, -0.3543521463871002, 0.13075105845928192, -0.21657197177410126, -0.11495479941368103, 0.0751824676990509, -0.09402791410684586, 0.11524442583322525, 0.17278125882148743, -0.08004692196846008, -0.1250341832637787, 0.08936438709497452, 0.010875500738620758, 0.3309323489665985, 0.30324333906173706, 0.24935847520828247, 0.26325905323028564, 0.14833098649978638, 0.12230972945690155, -0.036081112921237946, -0.01875821128487587, 0.15182167291641235, -0.41670936346054077, 0.3561733365058899, 0.05621141567826271, 0.33688586950302124, -0.10796955227851868, -0.2942178547382355, -0.08334088325500488, -0.2224130928516388, -0.25148722529411316, 0.027613844722509384, 0.21299588680267334, 0.029774893075227737, 0.1607840657234192, -0.10163012146949768, 0.310607373714447, 0.0777539610862732, 0.039710793644189835, -0.07049227505922318, 0.06991730630397797, -0.22135759890079498, -0.23978058993816376, 0.13247384130954742, -0.3666727542877197, 0.021909480914473534, 0.14092865586280823, -0.030365586280822754, 0.08047424256801605, -0.08656249940395355, -0.2426605522632599, 0.44843384623527527, 0.10931888967752457, 0.0181749127805233, 0.2889542877674103, 0.15471233427524567, -0.05855479836463928, -0.1528550684452057, 0.15041112899780273, -0.12792238593101501, -0.1970234364271164, -0.12976720929145813, 0.1871112585067749, -0.19612595438957214, -0.12540996074676514, -0.260233610868454, 0.046286724507808685, 0.07848338037729263, -0.20184342563152313, 0.05439405515789986, -0.3447802662849426, 0.1798258125782013, -0.38337963819503784, 0.2408025562763214, 0.32196617126464844, -0.6573533415794373, 0.14890262484550476, 0.3685605823993683, -0.10237351059913635, 0.3370114862918854, 0.2689688801765442, -0.12471570819616318, 0.516861081123352, -0.28704091906547546, 0.012729249894618988, -0.08588264137506485, -0.20366843044757843, -0.2628403306007385, 0.2163202464580536, 0.208762988448143, 0.37994441390037537, -0.0020478665828704834, -0.04133594408631325, 0.16595767438411713, -0.2725694477558136, 0.11481555551290512, 0.42455917596817017, -0.20413823425769806, 0.0962647870182991, -0.08622147142887115, 0.04533044993877411, 0.2136079967021942, -0.04319401830434799, 0.077810138463974, 0.10014522075653076, -0.08352603763341904, -0.2792647182941437, 0.20784991979599, -0.32493242621421814, -0.04465719312429428, -0.24377663433551788, 0.36748170852661133, 0.09334681183099747, 0.16207553446292877, -0.1633572280406952, -0.3386428952217102, 0.14716538786888123, -0.12948977947235107, -0.013376109302043915, -0.3402308225631714, -0.26823925971984863, -0.04463218152523041, 0.2982430160045624, -0.3973882496356964, -0.17660671472549438, 0.15491606295108795, 0.107688307762146, 0.27986645698547363, 0.05588323995471001, -0.22562530636787415, -0.15298819541931152, -0.02753128856420517, -0.00020723987836390734, 0.07146941125392914, -0.05063937231898308, -0.02691287361085415, -0.2508769631385803, -0.10000093281269073, 0.28035426139831543, 0.05793580412864685, -0.15736331045627594, -0.10361672192811966, 0.44240665435791016, -0.163614422082901, -0.08874295651912689, -0.12644696235656738, 0.07587598264217377, -0.047256603837013245, 0.1202467828989029, -0.19284482300281525, -0.1092534065246582, 0.12139110267162323, -0.03322063758969307, 0.07695548236370087, 0.4232720136642456, -0.15114551782608032, 0.38411349058151245, -0.04406912252306938, 0.08840122818946838, 0.37544116377830505, 0.2227180302143097, -0.18514612317085266, -0.10099558532238007, -0.08412079513072968, -0.12406229972839355, 0.13021264970302582, -0.22561554610729218, -0.3341463506221771, -0.062124960124492645, 0.2737691402435303, -0.03864460811018944, 0.27069923281669617, 0.003980979323387146, -0.27244511246681213, 0.15827885270118713, 0.11002293229103088, 0.05412546172738075, 0.2948543429374695, 0.19788235425949097, -0.05751212313771248, -0.07533994317054749, -0.08208096772432327, 0.04687193036079407, 0.2616609036922455, 0.23143811523914337, 0.22471754252910614, 0.029017498716711998, 0.04852786287665367, 0.19244171679019928, -0.06182397902011871, -0.43683600425720215, -0.17832481861114502, 0.24929118156433105, -0.2860504686832428, 0.13886350393295288, -0.2293650060892105, -0.17995941638946533, -0.12498196959495544, -0.5852276086807251, 0.16042499244213104, -0.2572799623012543, -0.011480819433927536, 0.0810718834400177, -0.22216805815696716, 0.2791282832622528, 0.1503353714942932, 0.06761865317821503, 0.08542414754629135, -0.383465439081192, -0.12528151273727417, -0.3304895758628845, -0.2332397699356079, 0.09674040973186493, 0.2671474814414978, -0.283251017332077, 0.48682379722595215, -0.13610060513019562, 0.0613897368311882, -0.3297157287597656, -0.08869849890470505, 0.07270674407482147, -0.206918403506279, 0.24724026024341583, 0.39719176292419434, 0.2686823904514313, 0.04914035648107529, -0.1559620350599289, 0.3782169222831726, -0.1504448801279068, 0.05700866878032684, 0.23151680827140808, -0.1650499403476715, -0.2562617361545563, 0.08352406322956085, -0.15325286984443665, -0.08032234013080597, -0.19916220009326935, 0.15514442324638367, -0.0917975902557373, 0.005093451589345932, 0.38040855526924133, 0.18336184322834015, 0.19829373061656952, -0.13087016344070435, 0.0885978490114212, 0.13638943433761597, 0.05680396780371666, 0.25793901085853577, -0.04834351688623428, -0.16644056141376495, -0.04647394269704819, -0.13203716278076172, 0.06557131558656693, 0.25526362657546997, -0.18135946989059448, -0.09563696384429932, -0.2632159888744354, 0.1960792988538742, 0.09520275890827179, 0.34151262044906616, 0.21047323942184448, 0.2515539526939392, -0.10886288434267044, -0.026361029595136642, -0.04641188308596611, 0.027521096169948578, -0.24319425225257874, -0.11008290201425552, 0.23617833852767944, 0.44715172052383423, 0.14846362173557281, 0.5902130603790283, 0.09584331512451172, -0.2325410097837448, 0.3301248550415039, -0.12104686349630356, 0.30287304520606995, -0.12511661648750305, -0.3857043981552124, -0.11241558194160461, -0.18732932209968567, -0.021618202328681946, 0.23665426671504974, -0.25005394220352173, -0.006772294640541077, -0.16393893957138062, 0.019488636404275894, -0.25845757126808167, -0.21763239800930023, 0.3283659517765045, -0.539763331413269, 0.3856008052825928, -0.04820350557565689, 0.09250540286302567, -0.008074982091784477, -0.01799626648426056, 0.07920245826244354, -0.08503790199756622, 0.11700260639190674, -0.10172279179096222, -0.359770268201828, -0.6891679763793945, -0.2801053524017334, 0.3845319151878357, 0.21025407314300537, 0.7424201369285583, 0.24298347532749176, -0.18816107511520386, -0.04458451271057129, 0.0608866922557354, 0.5138818025588989, -0.23532922565937042, -0.15241830050945282, 0.01885053515434265, -0.12801793217658997, -0.3994271755218506, -0.35004258155822754, -0.40419116616249084, 0.3465169072151184, -0.059657592326402664, 0.06346161663532257, -0.19901838898658752, -0.08881203830242157, 0.22778655588626862, 0.281804621219635, 0.0809573233127594, -0.09592732787132263, -0.20971786975860596, -0.2745293974876404, -0.32740458846092224, 0.27784398198127747, 0.12812009453773499, -0.016552885994315147, -0.11296351253986359, -0.02266361005604267, -0.41199278831481934, 0.10610739141702652, 0.06457596272230148, 0.26785120368003845, 0.3020259439945221, -0.011930882930755615, 0.07679381966590881, -0.15416894853115082, 0.20689235627651215, 0.5043161511421204, 0.20818409323692322, -0.15782110393047333, -0.26468315720558167, -0.22595864534378052, -0.08119356632232666, 0.32503941655158997, 0.19970466196537018, -0.25325292348861694, 0.07583749294281006, -0.3410640060901642, 0.1392166018486023, 0.00048539694398641586, -0.2907891571521759, 0.4336834251880646, 0.07882356643676758, -0.7235928177833557, -0.48073774576187134, 0.3294735252857208, 0.22584302723407745, 0.029900528490543365, 0.40439629554748535, 0.24470919370651245, -0.21497978270053864, 0.6425031423568726, -0.07863923162221909, 0.5631107687950134, 0.09813421219587326, 0.2064136415719986, 0.3651917576789856, 0.12064646184444427, 0.3637576401233673, 0.41517174243927, 0.14076638221740723, -0.2682613730430603, -0.09184890985488892, 0.015930386260151863, -0.2523359954357147, 0.09833583235740662, 0.31512290239334106, -0.23800957202911377, 0.37857264280319214, -0.3420734405517578, 0.1399787962436676, 0.0829126238822937, 0.1436111330986023, 0.11594036221504211, -0.26507246494293213, -0.3994799256324768, 0.10134480893611908, -0.12151959538459778, -0.05867641791701317, -0.27454686164855957, 0.0761415883898735, 0.05707104131579399, -0.22509099543094635, -0.3649146556854248, 0.001025831326842308, -0.1748826801776886, 0.1334579885005951, 0.09135628491640091, -0.514306902885437, 0.12675829231739044, 0.1911579817533493, 0.3580009937286377, 0.09742454439401627, 0.04665956646203995, 0.012406479567289352, 0.31951433420181274, 0.14495569467544556, 0.12863627076148987, -0.18850497901439667, 0.6046899557113647, 0.11281367391347885, -0.042143769562244415, 0.1924719214439392, -0.1313241720199585, -0.1009042039513588, 0.021296605467796326, 0.27963709831237793, 0.22113166749477386, -0.30561986565589905, -0.35497936606407166, -0.10924467444419861, -0.03825227543711662, -0.1593240350484848, 0.1160898208618164, 0.029563114047050476, -0.1382851004600525, 0.720177173614502, -0.17008085548877716, -0.26264744997024536, -0.08468882739543915, -0.07808510214090347, 0.04201003909111023, -0.08852067589759827, 0.5980509519577026, 0.27939939498901367, -0.21104992926120758, -0.10425931215286255, -0.05330675467848778, 0.09343262761831284, -0.6413567662239075, 0.31893405318260193, 0.10796889662742615, 0.30186212062835693, 0.07080690562725067, 0.39162594079971313, 0.14745637774467468, 0.07631614804267883, -0.03576178848743439, -0.19966675341129303, 0.08776767551898956, 0.11780142039060593, 0.17522522807121277, 0.3697345554828644, 0.3987768888473511, 0.25408652424812317, 0.13396215438842773, 0.028880085796117783, -0.3144587278366089, -0.20075052976608276, -0.006682965904474258, 0.30842193961143494, -0.21299535036087036, 0.44495442509651184, 0.14664675295352936, -0.014975541271269321, 0.14051483571529388, 0.1595325469970703, 0.2172217071056366, -0.3041934669017792, -0.06369626522064209, 0.0988791212439537, 0.035018399357795715, -0.1595446914434433, 0.1459624320268631, -0.0819488987326622, -0.13202504813671112, -0.35047537088394165, 0.026865635067224503, 0.1756671965122223, -0.21726830303668976, 0.08486589789390564, 0.37898293137550354, 0.043331459164619446, 0.08295033872127533, 0.18435825407505035, -0.33097362518310547, -0.1360924392938614, 0.1287888139486313, 0.24376888573169708, -0.038756102323532104, -0.10929456353187561, -0.07918940484523773, 0.06911759078502655, 0.16126997768878937, 0.2346152365207672, 0.00023084133863449097, -0.4827355742454529, -0.08896122872829437, 0.0838177353143692, 0.5207581520080566, 0.24799393117427826, -0.09680788964033127, -0.14331044256687164, 0.3929222524166107, 0.20763926208019257, -0.31200018525123596, 0.03423764929175377, -0.07028022408485413, 0.024620456621050835, 0.16865967214107513, 0.3431810140609741, 0.044558484107255936, 0.0626802146434784, 0.016043446958065033, 0.11282550543546677, 0.09054241329431534, -0.24976034462451935, 0.17523708939552307, 0.2036016583442688, 0.21697579324245453, 0.0699094757437706, 0.18049472570419312, 0.07898426055908203, 0.17690172791481018, 0.09728431701660156, 0.14528775215148926, 0.3262118995189667, 0.347344309091568, -0.23631474375724792, 0.07794038206338882, -0.411718487739563, -0.1310107409954071, -0.17048439383506775, 0.12594416737556458, 0.1122385710477829, 0.22222357988357544, 0.29544442892074585, -0.30192598700523376, -0.23801419138908386, 0.12048415094614029, -0.03598905727267265, -0.14244195818901062, 0.12832984328269958, -0.2816077768802643, -0.05525832623243332, 0.2715890109539032, -0.33285707235336304, -0.14328713715076447, -0.14527396857738495, -0.025154195725917816, 0.12122443318367004, 0.06968418508768082, -0.5451204776763916, 0.15391460061073303, 0.019781701266765594, 0.2689996063709259, -0.20365244150161743, 0.04772813618183136, 0.11656568944454193, -0.055856406688690186, 0.05398034676909447, 0.575695812702179, 0.700167715549469, 0.570318877696991, -0.09843911230564117, 0.005459723062813282, -0.397266685962677, -0.06752362102270126, 0.02717066928744316, 0.10839501023292542, -0.16304023563861847, -0.06871722638607025, 0.4603589177131653, 0.27455779910087585, -0.0994911640882492, 0.07754279673099518, 0.10224619507789612, -0.0801527351140976, -0.2041221410036087, 0.14548420906066895, 0.013964544981718063, -0.23537930846214294, -0.12350283563137054, -0.04237159341573715, -0.2765774726867676, -0.01633428782224655, 0.25566378235816956, -0.10940533876419067, 0.011422760784626007, -0.08704996854066849, 0.10606342554092407, 0.16118821501731873, 0.31619831919670105, 0.49550575017929077, 0.40905895829200745, -0.31004559993743896, -0.06800929456949234, -0.43101152777671814, -0.14271879196166992, -0.214900404214859, -0.03405456990003586, -0.01519308052957058, 0.19123145937919617, -0.03327411413192749, 0.1407856047153473, 0.3787408471107483, 0.16159646213054657, -0.2973161041736603, 0.5314661264419556, -0.2696308195590973, -0.20306168496608734, -0.3404104709625244, -0.012163301929831505, -0.0920218676328659, -0.33353039622306824, 0.0005804076790809631, -0.1205182820558548, 0.10839111357927322, -0.13333217799663544, -0.09766766428947449, 0.4252810478210449, -0.06260792911052704, 0.35336706042289734, 0.16644854843616486, 0.3122965693473816, -0.004311911761760712, 0.04353625327348709, -0.2503775358200073, -0.13272708654403687, -0.2814517617225647, 0.2846840023994446, 0.03165721520781517, 0.26063138246536255, 0.0003032190725207329, -0.20103105902671814, -0.1776227205991745, 0.03905129060149193, -0.028021685779094696, 0.18122707307338715, -0.3175504505634308, 0.20301635563373566, -0.04583369567990303, -0.05749526619911194, -0.06392946094274521, 0.18043649196624756, -0.03480295091867447, 0.34867724776268005, -0.43113696575164795, -0.4612301290035248, 0.36048436164855957, -0.647658109664917, -0.46051886677742004, 0.16082988679409027, 0.06617555022239685, -0.11015217751264572, -0.1059102788567543, 0.046320609748363495, -0.03652321174740791, 0.28041356801986694, 0.04805843532085419, -0.23012666404247284, -0.022182095795869827, -0.07016190886497498, 0.2760496437549591, -0.0694093406200409, 0.32596123218536377, 0.011947833001613617, -0.27900826930999756, -0.1365758329629898, -0.28092867136001587]}
"""

# 使用 FAISS 构建 embeddings 列的向量索引
# FAISS 是 Facebook 开发的高效向量相似度搜索库，支持大规模向量快速检索
embeddings_dataset.add_faiss_index(column="embeddings")

# 保存数据集和 FAISS 索引
embeddings_dataset.save_to_disk("embeddings_dataset")
embeddings_dataset.get_index("embeddings").save("embeddings_index.faiss")
print("数据集和索引已保存")
"""
下次加载时使用：                                                                                                                                                                   

  from datasets import load_from_disk                                                                                                                                                

  embeddings_dataset = load_from_disk("embeddings_dataset")                                                                                                                          
  embeddings_dataset.load_faiss_index("embeddings", "embeddings_index.faiss") 
"""

# 定义查询问题
question = "How can I load a dataset offline?"
# 将查询问题转换为嵌入向量
question_embedding = get_embeddings([question]).cpu().detach().numpy()
print(question_embedding.shape)
"""
(1, 768)
"""

# 使用 FAISS 进行相似度搜索，找出与查询问题最相似的 5 条记录
# scores: 相似度分数数组
# samples: 匹配的样本数据（字典格式）
scores, samples = embeddings_dataset.get_nearest_examples(
    "embeddings", question_embedding, k=5
)

import pandas as pd

# 将搜索结果转换为 pandas DataFrame，便于排序和展示
samples_df = pd.DataFrame.from_dict(samples)
samples_df["scores"] = scores  # 添加相似度分数列
# 按相似度分数降序排序（分数越高表示越相似）
samples_df.sort_values("scores", ascending=False, inplace=True)

# 遍历并打印搜索结果
for _, row in samples_df.iterrows():
    print(f"COMMENT: {row.comments}")  # 评论内容
    print(f"SCORE: {row.scores}")  # 相似度分数
    print(f"TITLE: {row.title}")  # issue 标题
    print(f"URL: {row.html_url}")  # issue 链接
    print("=" * 50)  # 分隔线
    print()

"""
截取输出
==================================================

COMMENT: I can browse the google drive through google chrome. It's weird. I can download the dataset through google drive manually.
SCORE: 40.68833923339844
TITLE: (Load dataset failure) ConnectionError: Couldn’t reach https://raw.githubusercontent.com/huggingface/datasets/1.1.2/datasets/cnn_dailymail/cnn_dailymail.py
URL: https://github.com/huggingface/datasets/issues/759
==================================================

COMMENT: Hi ! `datasets` currently supports reading local files or files over HTTP. We may add support for other filesystems (cloud storages, hdfs...) at one point though :)
SCORE: 38.603851318359375
TITLE: Does datasets support load text from HDFS?
URL: https://github.com/huggingface/datasets/issues/3490
==================================================

这个模型使用点积相似度，分数越高表示越相似，但没有固定上限。                                                                                                                       

  判断方法：                                                                                                                                                                         

  1. 看分数分布 — 检索返回的 5 条结果中，如果最高分是 31，第二名是 25，差距明显，那 31 分算是比较好的匹配。                                                                          
  2. 看阈值经验 — 对于这个模型（multi-qa-mpnet-base-dot-v1），一般：
    - > 60-70：非常相关，高质量匹配                                                                                                                                                  
    - 30-50：中等相关，有一定语义相似                                                                                                                                                
    - < 20：相关性较弱                                                                                                                                                               

  31 分属于中等水平，表示查询和结果有一定语义关联，但不是高度匹配。                                                                                                                  

  如果想提高匹配精度，可以：                                                                                                                                                         
  - 增加检索数量 k=10 或更多，看分数分布                    
  - 设置阈值过滤，如只保留 scores > 40 的结果
"""
