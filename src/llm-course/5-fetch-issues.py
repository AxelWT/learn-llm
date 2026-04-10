# from rich import print
import requests
import os
from dotenv import load_dotenv
from pathlib import Path
import math
import time
import pandas as pd
from tqdm import tqdm

# 加载 .env 文件，override=True 覆盖已有环境变量
load_dotenv(override=True)
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

# 参考 GitHub 官方的 API 文档进行设置 https://docs.github.com/en/rest/issues/issues?apiVersion=2026-03-10
headers = {
    "Accept": "application/vnd.github+json",
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "X-GitHub-Api-Version": "2026-03-10",
}


def fetch_issues(
        owner="huggingface",
        repo="datasets",
        # 要获取的 issue 总数
        num_issues=10_000,
        # GitHub API 速率限制阈值
        rate_limit=5_000,
        # issue 数据保存路径
        issues_path=Path("."),
):
    if not issues_path.is_dir():
        issues_path.mkdir(parents=True, exist_ok=True)

    batch = []
    all_issues = []
    # 每页返回的 issue 的数量
    per_page = 100
    num_pages = math.ceil(num_issues / per_page)
    base_url = "https://api.github.com/repos"

    for page in tqdm(range(num_pages)):
        # 使用 state=all 进行查询来获取 open 和 closed 的 issue
        query = f"issues?page={page}&per_page={per_page}&state=all"
        issues = requests.get(f"{base_url}/{owner}/{repo}/{query}", headers=headers)
        # extend 是把列表拆开每个元素都加入，append 是加入一个元素
        batch.extend(issues.json())

        if len(batch) > rate_limit and len(all_issues) < num_issues:
            all_issues.extend(batch)
            # 重置 batch
            batch = []
            print(f"Reached GitHub rate limit. Sleeping for one hour ...")
            # 设置了 token 官方的 rate limit 是 5000，可以注释掉这行代码
            # time.sleep(60 * 60 + 1)

    all_issues.extend(batch)
    df = pd.DataFrame.from_records(all_issues)
    df.to_json(f"{issues_path}/{repo}-issues.jsonl", orient="records", lines=True)
    print(f"Downloaded all the issues for {repo}! Dataset stored at {issues_path}/{repo}-issues.jsonl")


if __name__ == "__main__":
    fetch_issues()
