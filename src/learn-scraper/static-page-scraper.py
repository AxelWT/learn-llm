"""
静态网页爬虫模板
支持：重试机制、代理、User-Agent轮换、请求延迟、日志记录
"""

import time
import random
import logging
import os
import re
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

import requests
from bs4 import BeautifulSoup

# ============== 输出目录 ==============
OUTPUT_DIR = "output"


# ============== 配置 ==============
@dataclass
class CrawlerConfig:
    """爬虫配置"""
    base_url: str = ""
    timeout: int = 10
    max_retries: int = 3
    retry_delay: float = 1.0
    request_delay: float = 1.0  # 请求间隔，避免被封
    use_proxy: bool = False
    proxies: Optional[List[str]] = None

    # User-Agent池
    user_agents: tuple = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    )


# ============== 日志配置 ==============
def setup_logging(log_file: Optional[str] = None, level: int = logging.INFO):
    """配置日志"""
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    handlers = [logging.StreamHandler()]

    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format=log_format,
        handlers=handlers
    )


# ============== 爬虫核心 ==============
class StaticCrawler:
    """静态网页爬虫"""

    def __init__(self, config: Optional[CrawlerConfig] = None):
        self.config = config or CrawlerConfig()
        self.session = requests.Session()
        self.logger = logging.getLogger(self.__class__.__name__)

    def _get_headers(self) -> Dict[str, str]:
        """生成请求头"""
        return {
            "User-Agent": random.choice(self.config.user_agents),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        }

    def _get_proxies(self) -> Optional[Dict[str, str]]:
        """获取代理配置"""
        if not self.config.use_proxy or not self.config.proxies:
            return None

        proxy = random.choice(self.config.proxies)
        return {
            "http": proxy,
            "https": proxy,
        }

    def fetch(self, url: str, **kwargs) -> Optional[requests.Response]:
        """
        发送HTTP请求

        Args:
            url: 目标URL
            **kwargs: requests额外参数

        Returns:
            Response对象或None
        """
        headers = self._get_headers()
        headers.update(kwargs.pop("headers", {}))
        proxies = self._get_proxies()

        for attempt in range(1, self.config.max_retries + 1):
            try:
                self.logger.info(f"请求: {url} (尝试 {attempt}/{self.config.max_retries})")

                response = self.session.get(
                    url,
                    headers=headers,
                    proxies=proxies,
                    timeout=self.config.timeout,
                    **kwargs
                )
                response.raise_for_status()

                self.logger.info(f"成功: {url} - 状态码 {response.status_code}")
                return response

            except requests.RequestException as e:
                self.logger.warning(f"失败: {url} - {e}")

                if attempt < self.config.max_retries:
                    time.sleep(self.config.retry_delay * attempt)

        self.logger.error(f"最终失败: {url}")
        return None

    def parse(self, html: str, parser: str = "html.parser") -> BeautifulSoup:
        """
        解析HTML

        Args:
            html: HTML文本
            parser: 解析器 (lxml, html.parser, html5lib)

        Returns:
            BeautifulSoup对象
        """
        return BeautifulSoup(html, parser)

    def crawl(self, url: str) -> Optional[BeautifulSoup]:
        """
        爬取并解析页面（便捷方法）

        Args:
            url: 目标URL

        Returns:
            BeautifulSoup对象或None
        """
        response = self.fetch(url)
        if response:
            return self.parse(response.text)
        return None


def sanitize_filename(name: str) -> str:
    """清理文件名，移除非法字符"""
    # 移除非法字符
    name = re.sub(r'[<>:"/\\|?*\n\r\t]', '', name)
    # 替换空格为下划线
    name = name.strip().replace(' ', '_')
    # 限制长度
    if len(name) > 100:
        name = name[:100]
    # 如果为空，用时间戳
    if not name:
        name = f"page_{int(time.time())}"
    return name


def save_results(soup: BeautifulSoup, html_content: str, url: str):
    """
    保存爬取结果

    Args:
        soup: BeautifulSoup对象
        html_content: 原始HTML
        url: 原始URL
    """
    # 创建输出目录
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 获取标题作为文件名
    title = soup.find("title")
    title_text = title.text.strip() if title else ""
    filename = sanitize_filename(title_text)

    # 保存HTML文件
    html_path = os.path.join(OUTPUT_DIR, f"{filename}.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(f"<!-- 来源: {url} -->\n")
        f.write(f"<!-- 爬取时间: {time.strftime('%Y-%m-%d %H:%M:%S')} -->\n")
        f.write(html_content)
    print(f"HTML已保存: {html_path}")

    # 提取文本内容并保存TXT
    text_content = []
    text_content.append(f"来源URL: {url}")
    text_content.append(f"爬取时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    text_content.append(f"页面标题: {title_text}")
    text_content.append("=" * 50)

    # 提取标题
    for h in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
        text = h.text.strip()
        if text:
            text_content.append(f"\n[{h.name}] {text}")

    # 提取段落
    text_content.append("\n【正文内容】")
    text_content.append("-" * 30)
    for p in soup.find_all("p"):
        text = p.text.strip()
        if text:
            text_content.append(text)

    # 提取链接列表
    links = soup.find_all("a", href=True)
    if links:
        text_content.append("\n【链接列表】")
        text_content.append("-" * 30)
        for link in links:
            text = link.text.strip() or "[无文本]"
            href = link["href"]
            text_content.append(f"{text} -> {href}")

    # 保存TXT文件
    txt_path = os.path.join(OUTPUT_DIR, f"{filename}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(text_content))
    print(f"TXT已保存: {txt_path}")


def display_page_info(soup: BeautifulSoup):
    """展示页面信息"""
    print("\n" + "=" * 50)
    print("【页面信息】")
    print("=" * 50)

    # 标题
    title = soup.find("title")
    print(f"\n标题: {title.text.strip() if title else '无标题'}")

    # meta描述
    meta_desc = soup.find("meta", attrs={"name": "description"})
    if meta_desc and meta_desc.get("content"):
        print(f"描述: {meta_desc['content'][:100]}...")

    # 链接统计
    links = soup.find_all("a", href=True)
    print(f"\n链接总数: {len(links)}")

    # 图片统计
    images = soup.find_all("img")
    print(f"图片总数: {len(images)}")


def main():
    """主程序：交互式爬取，支持多个网址"""
    setup_logging()

    config = CrawlerConfig(timeout=15, max_retries=3)
    crawler = StaticCrawler(config)

    print("\n" + "=" * 50)
    print("静态网页爬虫 - 支持批量爬取")
    print("=" * 50)

    while True:
        print("\n请输入要爬取的网址（每行一个，输入空行开始爬取，单独输入 q 退出）:")

        lines = []
        while True:
            line = input().strip()
            if line.lower() == "q":
                print("再见!")
                return
            if line == "":
                break
            lines.append(line)

        if not lines:
            print("请输入有效的网址")
            continue

        # 解析多个网址
        urls = [line for line in lines if line]

        # 自动补全协议
        urls = [
            u if u.startswith(("http://", "https://")) else "https://" + u
            for u in urls
        ]

        print(f"\n共 {len(urls)} 个网址，开始爬取...")

        for i, url in enumerate(urls, 1):
            print(f"\n[{i}/{len(urls)}] 正在爬取: {url}")

            response = crawler.fetch(url)
            if response:
                soup = crawler.parse(response.text)
                display_page_info(soup)
                save_results(soup, response.text, url)
            else:
                print(f"爬取失败: {url}")

            # 多个网址时添加延迟
            if i < len(urls):
                time.sleep(config.request_delay)

        print(f"\n全部完成！结果保存在 {OUTPUT_DIR} 目录")


# ============== 入口 ==============
if __name__ == "__main__":
    main()
