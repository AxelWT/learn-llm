"""
智能爬虫 - 自动判断页面类型并选择合适的爬取方式
静态页面：使用 requests + BeautifulSoup
动态页面：使用 Playwright，支持手动登录
"""

import os
import re
import time
import logging
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass

import requests
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright

# ============== 配置 ==============
OUTPUT_DIR = "output"


@dataclass
class CrawlerConfig:
    """爬虫配置"""
    timeout: int = 10
    max_retries: int = 3
    retry_delay: float = 1.0
    request_delay: float = 1.0

    user_agents: tuple = (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
    )


# ============== 日志 ==============
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )


# ============== 页面类型检测 ==============
class PageTypeDetector:
    """页面类型检测器"""

    FRAMEWORK_SIGNATURES = {
        "React": [r'data-reactroot', r'react-dom', r'__REACT_DEVTOOLS'],
        "Vue": [r'data-v-[a-f0-9]+', r'__vue__', r'Vue\.'],
        "Angular": [r'ng-version', r'ng-app', r'angular\.module'],
        "Next.js": [r'__NEXT_DATA__', r'_next/static'],
        "SPA": [r'window\.__INITIAL_STATE__', r'bundle\..*\.js', r'main\..*\.js'],
    }

    DYNAMIC_SIGNATURES = [
        r'document\.write', r'innerHTML', r'fetch\(',
        r'axios\.', r'XMLHttpRequest', r'\.ajax\(',
    ]

    # 登录页面特征
    LOGIN_SIGNATURES = [
        r'login', r'signin', r'sign-in', r'登录', r'登陆',
        r'auth', r'authenticate', r'password', r'密码',
        r'<form.*action.*login', r'type="password"',
        r'请登录', r'请先登录', r'需要登录',
    ]

    def __init__(self, timeout: int = 10):
        self.timeout = timeout

    def fetch_html(self, url: str) -> Optional[str]:
        """获取原始HTML"""
        try:
            headers = {"User-Agent": "Mozilla/5.0 Chrome/120.0.0.0"}
            response = requests.get(url, headers=headers, timeout=self.timeout)
            return response.text
        except Exception:
            return None

    def check_login_page(self, html: str) -> Tuple[bool, List[str]]:
        """检测是否为登录页面"""
        soup = BeautifulSoup(html, "html.parser")
        reasons = []

        text_content = html.lower()

        # 检查登录关键词
        for pattern in self.LOGIN_SIGNATURES:
            if re.search(pattern, text_content, re.IGNORECASE):
                reasons.append(f"发现登录特征: {pattern}")

        # 检查是否有密码输入框
        password_inputs = soup.find_all("input", type="password")
        if password_inputs:
            reasons.append("存在密码输入框")

        # 检查标题包含登录关键词
        title = soup.find("title")
        if title:
            title_text = title.text.lower()
            if any(kw in title_text for kw in ["login", "登录", "signin", "sign in", "auth"]):
                reasons.append(f"标题包含登录关键词: {title.text}")

        # 检查登录表单
        login_forms = soup.find_all("form")
        for form in login_forms:
            form_text = str(form).lower()
            if any(kw in form_text for kw in ["login", "登录", "signin", "password", "auth"]):
                reasons.append("存在登录表单")
                break

        is_login = len(reasons) >= 2
        return is_login, reasons

    def detect(self, url: str) -> Tuple[bool, float, Optional[str], bool, List[str]]:
        """
        检测页面类型

        Returns:
            (is_dynamic, confidence, framework, is_login_page, login_reasons)
        """
        html = self.fetch_html(url)
        if not html:
            return False, 0.0, None, False, ["无法获取页面"]

        confidence = 0.0
        framework = None

        # 检查是否为登录页面
        is_login, login_reasons = self.check_login_page(html)

        # 检查JS框架
        for fw, patterns in self.FRAMEWORK_SIGNATURES.items():
            for pattern in patterns:
                if re.search(pattern, html, re.IGNORECASE):
                    framework = fw
                    confidence += 0.7
                    break

        # 检查动态脚本
        for pattern in self.DYNAMIC_SIGNATURES:
            if re.search(pattern, html):
                confidence += 0.2

        # 检查内容结构
        soup = BeautifulSoup(html, "html.parser")
        body = soup.find("body")
        if body:
            body_text = body.get_text(strip=True)
            if len(body_text) < 50:
                confidence += 0.8

        # 检查空div比例
        divs = soup.find_all("div")
        empty_divs = [d for d in divs if not d.get_text(strip=True)]
        if divs and len(empty_divs) / len(divs) > 0.5:
            confidence += 0.5

        is_dynamic = confidence >= 0.5
        return is_dynamic, min(confidence, 1.0), framework, is_login, login_reasons


# ============== 静态页面爬虫 ==============
class StaticCrawler:
    """静态网页爬虫"""

    def __init__(self, config: Optional[CrawlerConfig] = None):
        self.config = config or CrawlerConfig()
        self.session = requests.Session()
        self.logger = logging.getLogger(self.__class__.__name__)

    def _get_headers(self) -> Dict[str, str]:
        import random
        return {
            "User-Agent": random.choice(self.config.user_agents),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        }

    def fetch(self, url: str) -> Optional[requests.Response]:
        """发送HTTP请求"""
        import random

        for attempt in range(1, self.config.max_retries + 1):
            try:
                self.logger.info(f"请求: {url} (尝试 {attempt})")
                response = self.session.get(
                    url,
                    headers=self._get_headers(),
                    timeout=self.config.timeout
                )
                response.raise_for_status()
                self.logger.info(f"成功: {url}")
                return response
            except requests.RequestException as e:
                self.logger.warning(f"失败: {e}")
                if attempt < self.config.max_retries:
                    time.sleep(self.config.retry_delay * attempt)
        return None

    def crawl(self, url: str) -> Tuple[Optional[str], Optional[str]]:
        """爬取并返回(html, title)"""
        response = self.fetch(url)
        if response:
            soup = BeautifulSoup(response.text, "html.parser")
            title = soup.find("title")
            title_text = title.text.strip() if title else ""
            return response.text, title_text
        return None, None


# ============== 动态页面爬虫 ==============
class DynamicCrawler:
    """动态网页爬虫（Playwright）"""

    def crawl(self, url: str, headless: bool = False) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """
        爬取动态页面

        Args:
            url: 目标URL
            headless: 是否无头模式

        Returns:
            (html, text, title)
        """
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=headless)
            context = browser.new_context()
            page = context.new_page()

            try:
                print(f"正在访问: {url}")
                page.goto(url, wait_until="networkidle")

                if not headless:
                    # 有头模式：等待用户确认
                    print("\n" + "=" * 50)
                    print("浏览器已打开:")
                    print("  - 如果需要登录，请在浏览器中手动登录")
                    print("  - 登录完成后，按回车继续爬取")
                    print("=" * 50)
                    input("\n按回车键继续...")

                    print("\n正在爬取页面内容，请稍候...")
                    page.goto(url, wait_until="networkidle")
                    time.sleep(2)
                    print("页面加载完成，提取数据...")
                else:
                    print("等待页面渲染...")
                    time.sleep(3)
                    print("提取数据...")

                html = page.content()
                text = page.evaluate("() => document.body.innerText")
                title = page.title()
                print("数据提取完成！")

                return html, text, title

            except Exception as e:
                print(f"爬取失败: {e}")
                return None, None, None

            finally:
                browser.close()


# ============== 文件保存 ==============
def sanitize_filename(name: str) -> str:
    """清理文件名"""
    name = re.sub(r'[<>:"/\\|?*\n\r\t]', '', name)
    name = name.strip().replace(' ', '_')
    if len(name) > 100:
        name = name[:100]
    return name or f"page_{int(time.time())}"


def ensure_output_dir():
    """确保输出目录存在"""
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)


def save_static_result(html: str, title: str, url: str):
    """保存静态页面结果"""
    ensure_output_dir()
    filename = sanitize_filename(title)

    # 保存HTML
    html_path = os.path.join(OUTPUT_DIR, f"{filename}.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(f"<!-- 来源: {url} -->\n")
        f.write(html)
    print(f"HTML已保存: {html_path}")

    # 保存TXT
    soup = BeautifulSoup(html, "html.parser")
    text_content = []
    text_content.append(f"来源URL: {url}")
    text_content.append(f"页面标题: {title}")
    text_content.append("=" * 50)

    for h in soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
        if h.text.strip():
            text_content.append(f"\n[{h.name}] {h.text.strip()}")

    text_content.append("\n【正文】")
    for p in soup.find_all("p"):
        if p.text.strip():
            text_content.append(p.text.strip())

    links = soup.find_all("a", href=True)
    if links:
        text_content.append("\n【链接】")
        for link in links[:20]:
            text_content.append(f"{link.text.strip()[:30]} -> {link['href']}")

    txt_path = os.path.join(OUTPUT_DIR, f"{filename}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(text_content))
    print(f"TXT已保存: {txt_path}")


def save_dynamic_result(html: str, text: str, title: str, url: str):
    """保存动态页面结果"""
    ensure_output_dir()
    filename = sanitize_filename(title)

    # 保存HTML
    html_path = os.path.join(OUTPUT_DIR, f"{filename}.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(f"<!-- 来源: {url} -->\n")
        f.write(html)
    print(f"HTML已保存: {html_path}")

    # 保存TXT
    txt_path = os.path.join(OUTPUT_DIR, f"{filename}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"来源URL: {url}\n")
        f.write(f"页面标题: {title}\n")
        f.write("=" * 50 + "\n")
        f.write(text)
    print(f"TXT已保存: {txt_path}")


# ============== 主程序 ==============
def get_urls_input() -> List[str]:
    """获取用户输入的URL列表"""
    print("\n请输入要爬取的网址（每行一个，空行开始爬取，q退出）:")
    lines = []
    while True:
        line = input().strip()
        if line.lower() == "q":
            return None
        if line == "":
            break
        lines.append(line)

    urls = []
    for line in lines:
        if not line.startswith(("http://", "https://")):
            line = "https://" + line
        urls.append(line)

    return urls


def get_headless_mode() -> bool:
    """获取运行模式"""
    print("\n选择运行模式:")
    print("  1 - 有头模式（显示浏览器，可手动登录）")
    print("  2 - 无头模式（后台运行）")
    choice = input("请选择 (1/2，默认1): ").strip()
    return choice == "2"


def get_crawler_choice(is_dynamic: bool, is_login: bool, login_reasons: List[str]) -> str:
    """
    让用户确认或选择爬取方式

    Returns:
        'static', 'dynamic', 或 'skip'
    """
    print("\n请选择爬取方式:")
    print("  1 - 静态爬取")
    print("  2 - 动态爬取")
    print("  3 - 跳过此页面")

    # 自动检测建议
    if is_login:
        print(f"\n检测到登录页面特征:")
        for reason in login_reasons[:3]:
            print(f"    - {reason}")
        print("  建议: 选择动态爬取，手动登录")
        default = "2"
    elif is_dynamic:
        print("  建议: 检测为动态页面，建议动态爬取")
        default = "2"
    else:
        print("  建议: 检测为静态页面，建议静态爬取")
        default = "1"

    choice = input(f"\n请选择 (1/2/3，默认{default}): ").strip()

    if choice == "3":
        return "skip"
    elif choice == "2":
        return "dynamic"
    elif choice == "1":
        return "static"
    else:
        return "dynamic" if default == "2" else "static"


def main():
    """主程序"""
    setup_logging()

    detector = PageTypeDetector(timeout=15)
    static_crawler = StaticCrawler(CrawlerConfig(timeout=15))
    dynamic_crawler = DynamicCrawler()

    headless = None  # 运行模式，首次动态爬取时询问

    print("=" * 50)
    print("智能爬虫 - 自动判断页面类型")
    print("=" * 50)

    while True:
        urls = get_urls_input()
        if urls is None:
            print("再见!")
            break

        if not urls:
            print("未输入有效网址")
            continue

        print(f"\n共 {len(urls)} 个网址，开始处理...")

        for i, url in enumerate(urls, 1):
            print(f"\n{'=' * 50}")
            print(f"[{i}/{len(urls)}] {url}")

            # 检测页面类型
            print("\n检测页面特征...")
            is_dynamic, confidence, framework, is_login, login_reasons = detector.detect(url)

            # 显示检测结果
            if is_login:
                print("检测到登录页面特征！")
            else:
                page_type = "动态页面" if is_dynamic else "静态页面"
                print(f"检测结果: {page_type} (置信度 {confidence:.0%}%)")

            if framework:
                print(f"框架: {framework}")

            # 让用户确认或手动选择
            crawler_type = get_crawler_choice(is_dynamic, is_login, login_reasons)

            if crawler_type == "skip":
                print("跳过此页面")
                continue

            elif crawler_type == "dynamic":
                # 首次使用动态爬虫时询问运行模式
                if headless is None:
                    headless = get_headless_mode()

                # 使用动态爬虫
                html, text, title = dynamic_crawler.crawl(url, headless=headless)

                if html:
                    save_dynamic_result(html, text, title or "", url)
                    # 检查结果是否为登录页面
                    if "登录" in text or "login" in text.lower() or "请先登录" in text or "signin" in text.lower():
                        print("\n爬取结果仍包含登录提示！")
                        print("  1 - 重新爬取（手动登录）")
                        print("  2 - 保存当前结果并继续")
                        retry_choice = input("请选择 (1/2，默认1): ").strip()

                        if retry_choice != "2":
                            # 重新爬取，强制有头模式手动登录
                            print("\n重新打开浏览器，请在浏览器中手动登录...")
                            html, text, title = dynamic_crawler.crawl(url, headless=False)
                            if html:
                                save_dynamic_result(html, text, title or "", url)
                            else:
                                print("重新爬取失败")
                else:
                    print("爬取失败")

            elif crawler_type == "static":
                # 使用静态爬虫
                html, title = static_crawler.crawl(url)

                if html:
                    save_static_result(html, title or "", url)
                else:
                    print("爬取失败")

            # 延迟
            if i < len(urls):
                time.sleep(1)

        print(f"\n全部完成！结果保存在 {OUTPUT_DIR}/ 目录")


if __name__ == "__main__":
    main()
