import asyncio
import aiohttp
from tqdm.asyncio import tqdm_asyncio
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv(override=True)
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

headers = {
    "Accept": "application/vnd.github+json",
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "X-GitHub-Api-Version": "2026-03-10",
}


async def get_comments_async(session, issue_number, headers, max_retries=3):
    """异步获取单个 issue 的评论，带重试机制"""
    url = f"https://api.github.com/repos/huggingface/datasets/issues/{issue_number}/comments"

    for attempt in range(max_retries):
        try:
            async with session.get(url, headers=headers) as response:
                remaining = response.headers.get("X-RateLimit-Remaining", "5000")
                if int(remaining) <= 10:
                    print(f"Rate limit low ({remaining}), sleeping 1 hour...")
                    await asyncio.sleep(3600)

                if response.status == 200:
                    data = await response.json()
                    return issue_number, [r["body"] for r in data]
                elif response.status == 404:
                    return issue_number, []  # Issue 没有评论
                else:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** attempt)  # 指数退避
                    continue
        except Exception as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
            else:
                print(f"Error fetching issue {issue_number} after {max_retries} retries: {e}")
                return issue_number, []

    return issue_number, []


async def fetch_all_comments_async(issue_numbers, headers, concurrency=5):
    """异步并发获取所有 issue 评论，使用信号量控制并发"""
    comments_dict = {}
    semaphore = asyncio.Semaphore(concurrency)
    connector = aiohttp.TCPConnector(limit=concurrency, ssl=False)
    timeout = aiohttp.ClientTimeout(total=60)

    async def limited_get(session, issue_number):
        async with semaphore:
            result = await get_comments_async(session, issue_number, headers)
            # 请求之间加间隔，避免触发 GitHub 二级速率限制
            await asyncio.sleep(0.5)
            return result

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = [limited_get(session, num) for num in issue_numbers]
        results = await tqdm_asyncio.gather(*tasks, desc="Fetching comments")

        for issue_num, comments in results:
            comments_dict[issue_num] = comments

    return comments_dict


async def main():
    # 加载已有的 issues 数据集
    df = pd.read_json("datasets-issues.jsonl", lines=True)
    from datasets import Dataset
    issues_dataset = Dataset.from_pandas(df)

    issue_numbers = issues_dataset["number"]
    print(f"Total issues to fetch comments: {len(issue_numbers)}")

    # 异步获取所有评论
    comments_dict = await fetch_all_comments_async(issue_numbers, headers, concurrency=5)

    # 添加到 dataset
    issues_with_comments_dataset = issues_dataset.map(
        lambda x: {"comments": comments_dict.get(x["number"], [])}
    )

    print(issues_with_comments_dataset)
    print(f"Sample issue comments: {issues_with_comments_dataset[0]['comments']}")

    # 保存一份到本地，防止网络断开，上传不了时数据丢失
    issues_with_comments_dataset.save_to_disk("issues-with-comments-dataset")
    # 保存到 hub
    issues_with_comments_dataset.push_to_hub(repo_id="<account>/<repo-id>",
                                             commit_message="huggingface/datasets repository issues and comments")

    return issues_with_comments_dataset


if __name__ == "__main__":
    asyncio.run(main())
