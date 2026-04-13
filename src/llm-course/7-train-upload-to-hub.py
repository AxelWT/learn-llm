"""
单独上传本地模型到 Hugging Face Hub

用途：当训练脚本的网络上传失败时，可使用此脚本单独上传已保存的本地模型

使用方法：
    python 7-train-upload-to-hub.py

或指定参数：
    python 7-train-upload-to-hub.py --model-dir bert-finetuned-ner --repo-name your-username/bert-finetuned-ner
"""

import argparse
from huggingface_hub import HfApi
from transformers import AutoModelForTokenClassification, AutoTokenizer


def upload_model(model_dir: str, repo_name: str, commit_message: str = "upload model"):
    """
    上传本地模型到 Hugging Face Hub

    参数:
        model_dir: 本地模型目录
        repo_name: Hub 上的仓库名（格式：username/repo-name）
        commit_message: 提交信息
    """
    api = HfApi()

    # 创建仓库（如果不存在）
    api.create_repo(repo_id=repo_name, repo_type="model", exist_ok=True)
    print(f"仓库已准备好: {repo_name}")

    # 上传整个模型目录（排除 checkpoint 目录）
    api.upload_folder(
        folder_path=model_dir,
        repo_id=repo_name,
        repo_type="model",
        commit_message=commit_message,
        ignore_patterns=["checkpoint-*"],
    )
    print(f"模型已成功上传到: https://huggingface.co/{repo_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="上传本地模型到 Hugging Face Hub")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="bert-finetuned-ner",
        help="本地模型目录路径"
    )
    parser.add_argument(
        "--repo-name",
        type=str,
        default=None,
        help="Hub 仓库名（格式：username/repo-name），默认从模型配置读取"
    )
    parser.add_argument(
        "--commit-message",
        type=str,
        default="upload trained model",
        help="提交信息"
    )

    args = parser.parse_args()

    # 如果未指定 repo_name，尝试从模型配置获取
    if args.repo_name is None:
        try:
            model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
            # 从模型配置获取 Hub 名称（如果有）
            name_or_path = model.config._name_or_path
            # 检查是否是有效的 Hub repo_id 格式（包含 /）
            if "/" in name_or_path:
                args.repo_name = name_or_path
            else:
                # 获取当前用户名并构造完整的 repo_id
                api = HfApi()
                username = api.whoami()["name"]
                args.repo_name = f"{username}/{name_or_path}"
            print(f"使用仓库名: {args.repo_name}")
        except Exception as e:
            print(f"无法从模型配置获取仓库名: {e}")
            print("请指定 --repo-name 参数，例如：--repo-name your-username/bert-finetuned-ner")
            exit(1)

    upload_model(args.model_dir, args.repo_name, args.commit_message)
