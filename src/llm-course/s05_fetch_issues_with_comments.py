import pandas as pd
from datasets import Dataset
import requests
from dotenv import load_dotenv
import os

# 加载 .env 文件，override=True 覆盖已有环境变量
load_dotenv(override=True)
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
headers = {
    "Accept": "application/vnd.github+json",
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "X-GitHub-Api-Version": "2026-03-10",
}

# pandas 能更好地处理混合类型，lines=True读 取 jsonl 格式文件，默认lines=False 读取整个文件一个大 JSON 数组
df = pd.read_json("datasets-issues.jsonl", lines=True)

# 转为 datasets
issues_dataset = Dataset.from_pandas(df)
print(issues_dataset)

# 根据 GitHub 官方文档提示，列表中包含 issue 和 pull request，鉴别两者区别并 filter 掉 pull request
sample = issues_dataset.shuffle(seed=666).select(range(3))

# 打印出 URL 和 pull 请求
for url, pr in zip(sample["html_url"], sample["pull_request"]):
    print(f">> URL: {url}")
    print(f">> Pull request: {pr}\n")

# 新增一个字段，用于区分 issue和 pull request 数据
issues_dataset = issues_dataset.map(
    lambda x: {"is_pull_request": False if x["pull_request"] is None else True}
)
print(issues_dataset)


def get_comments(issue_number):
    url_comment = f"https://api.github.com/repos/huggingface/datasets/issues/{issue_number}/comments"
    response = requests.get(url_comment, headers=headers)
    return [r["body"] for r in response.json()]


# 测试我们的函数是否按预期工作
print(get_comments(2792))

# 新增一个字段，issue 的评论列表
issues_with_comments_dataset = issues_dataset.map(
    lambda x: {"comments": get_comments(x["number"])}
)

issues_with_comments_dataset.push_to_hub(repo_id="<acount>/<repo_id>",
                                         commit_message="huggingface/datasets repository issues and comments")

## 创建新的数据集仓库（推荐用网页创建更直观）
# 或用 CLI：
# 终端用 huggingface 客户端运行： hf repo create your - dataset - name - -type dataset  创建数据集hub仓库
# 直接推送到 Hub
#   dataset.push_to_hub(
#       repo_id="你的用户名/your-dataset-name",
#       commit_message="Add dataset"
#   )
