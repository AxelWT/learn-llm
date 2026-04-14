"""
动态网页爬取脚本 - 使用 Playwright
支持交互式输入多个 URL 进行爬取

使用前请先配置 cookies.json
"""

from playwright.sync_api import sync_playwright
import json
import time
from pathlib import Path
import re


def parse_cookie_string(cookie_str: str, domain: str) -> list:
    """
    解析 HTTP Cookie 字符串格式，转换为 Playwright 所需的 cookie 列表

    支持格式: "name1=value1; name2=value2; name3=value3"
    """
    cookies = []
    for item in cookie_str.split(";"):
        item = item.strip()
        if "=" in item:
            name, value = item.split("=", 1)
            cookies.append({
                "name": name.strip(),
                "value": value.strip(),
                "domain": domain,
                "path": "/"
            })
    return cookies


def load_cookies(cookie_file: str = "cookies.json", default_domain: str = None) -> list:
    """
    加载 cookies 文件，支持两种格式：
    1. 标准 Playwright 格式：[{name, value, domain, path}, ...]
    2. Cookie 字符串格式：[{name: "Cookie", value: "name1=val1; name2=val2", ...}]
    """
    with open(cookie_file, "r") as f:
        data = json.load(f)

    cookies = []
    for item in data:
        item_domain = item.get("domain", default_domain)
        # 如果是 Cookie 字符串格式（name="Cookie"，value 是拼接的字符串）
        if item.get("name") == "Cookie" and item.get("value"):
            if item_domain:
                parsed = parse_cookie_string(item["value"], item_domain)
                cookies.extend(parsed)
        else:
            # 标准 Playwright 格式，直接使用
            cookie = {
                "name": item.get("name"),
                "value": item.get("value"),
                "domain": item_domain,
                "path": item.get("path", "/")
            }
            if cookie["name"] and cookie["value"] and cookie["domain"]:
                cookies.append(cookie)

    return cookies


def sanitize_filename(name: str) -> str:
    """清理文件名，移除不安全字符"""
    # 移除或替换不安全字符
    name = re.sub(r'[<>:"/\\|?*\n\r\t]', '_', name)
    # 移除前后空格
    name = name.strip()
    # 限制长度
    if len(name) > 100:
        name = name[:100]
    # 如果为空，返回默认值
    return name or "untitled"


def get_page_title(page) -> str:
    """从页面提取标题"""
    # 优先尝试 <title> 标签
    title = page.title()
    if title:
        return sanitize_filename(title)

    # 尝试 <h1> 标签
    try:
        h1 = page.locator("h1").first.text_content()
        if h1:
            return sanitize_filename(h1)
    except:
        pass

    # 尝试 meta og:title
    try:
        og_title = page.locator('meta[property="og:title"]').get_attribute("content")
        if og_title:
            return sanitize_filename(og_title)
    except:
        pass

    return "untitled"


def get_filename_from_url(url: str) -> str:
    """从 URL 生成安全的文件名（作为备选）"""
    parsed = urlparse(url)
    path = parsed.path.strip("/").replace("/", "_")
    if not path:
        path = parsed.netloc.replace(".", "_")
    return sanitize_filename(path or "page")


def scrape_page(page, url: str, wait_time: int = 3) -> tuple:
    """
    爬取单个页面内容

    Args:
        page: Playwright page 对象
        url: 目标页面 URL
        wait_time: 等待动态内容的时间（秒）

    Returns:
        (html, text, title) 页面 HTML、纯文本内容和标题
    """
    print(f"正在访问: {url}")
    page.goto(url, wait_until="networkidle")

    print("等待页面渲染...")
    time.sleep(wait_time)

    try:
        page.wait_for_selector("body", timeout=10000)
    except Exception as e:
        print(f"等待超时: {e}")

    print("提取页面内容...")
    html = page.content()
    text = page.evaluate("() => document.body.innerText")
    title = get_page_title(page)
    print(f"页面标题: {title}")

    return html, text, title


def save_content(html: str, text: str, filename: str, output_dir: str = "output"):
    """保存爬取的内容"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # 保存 HTML
    html_file = output_path / f"{filename}.html"
    with open(html_file, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"HTML 已保存到: {html_file}")

    # 保存纯文本
    text_file = output_path / f"{filename}.txt"
    with open(text_file, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"文本已保存到: {text_file}")


def input_urls() -> list:
    """交互式输入 URL"""
    print("\n请输入要爬取的 URL（每行一个，输入空行结束）：")
    urls = []
    while True:
        url = input().strip()
        if not url:
            break
        if not url.startswith("http"):
            url = "https://" + url
        urls.append(url)
    return urls


def main():
    COOKIE_FILE = "cookies.json"

    # 检查 cookies 文件
    if not Path(COOKIE_FILE).exists():
        print(f"提示: 未找到 {COOKIE_FILE}，将无 cookie 模式运行")
        cookies = []
    else:
        try:
            cookies = load_cookies(COOKIE_FILE)
            print(f"已加载 {len(cookies)} 个 cookies")
        except Exception as e:
            print(f"加载 cookies 失败: {e}")
            cookies = []

    # 交互式输入 URL
    urls = input_urls()

    if not urls:
        print("未输入任何 URL，退出")
        return

    # 选择模式
    print("\n选择运行模式：")
    print("  1. 有头模式（显示浏览器，方便调试）")
    print("  2. 无头模式（后台运行，速度快）")
    mode = input("请输入选项 (1/2，默认 1): ").strip()
    headless = mode == "2"

    # 爬取多个页面
    print(f"\n开始爬取 {len(urls)} 个页面...")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context()

        if cookies:
            context.add_cookies(cookies)

        page = context.new_page()

        for i, url in enumerate(urls, 1):
            print(f"\n[{i}/{len(urls)}] 处理: {url}")
            try:
                html, text, title = scrape_page(page, url)
                # 使用标题作为文件名，如果标题为空则用 URL
                filename = title if title != "untitled" else get_filename_from_url(url)
                save_content(html, text, filename)
                print(f"文本预览: {text[:200]}...")
            except Exception as e:
                print(f"爬取失败: {e}")

        browser.close()

    print(f"\n全部完成！结果保存在 output/ 目录")


if __name__ == "__main__":
    main()
