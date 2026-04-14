"""
监控包装器 - 外部注入方式（单文件版本）

通过命令行参数指定目标脚本和函数，监控其网络请求。
支持代理配置，所有请求可通过指定代理服务器发送。

用法：
    python monitor_wrapper.py script.py
    python monitor_wrapper.py script.py -f function_name
    python monitor_wrapper.py script.py -p http://127.0.0.1:7890
    python monitor_wrapper.py script.py -f func -p socks5://proxy:1080

示例：
    python monitor_wrapper.py download_script.py
    python monitor_wrapper.py download_script.py -f download_example_files
    python monitor_wrapper.py download_script.py -p http://127.0.0.1:7890
    python monitor_wrapper.py my_script.py -f main -p socks5://user:pass@proxy:1080
"""

import requests
from functools import wraps
import time
import threading
import argparse
import importlib.util
import sys
import os


# ============================================================
#                    网络监控实现
# ============================================================

class NetworkMonitor:
    """网络流量监控器"""

    def __init__(self):
        self.stats = {
            'total_requests': 0,
            'total_bytes_sent': 0,
            'total_bytes_recv': 0,
            'requests': []
        }
        self._original_request = None
        self._patched = False
        self._lock = threading.Lock()
        self._proxy = None  # 代理地址

    def set_proxy(self, proxy: str):
        """
        设置代理地址

        参数：
            proxy: 代理地址，支持格式：
                - http://127.0.0.1:7890
                - socks5://127.0.0.1:1080
                - socks5://user:password@proxy:1080（带认证）
        """
        self._proxy = proxy
        print(f"[代理] 已配置: {proxy}")

    def start(self):
        """启动监控，应用 monkey-patch"""
        if self._patched:
            print("[监控] 已在运行中")
            return

        # 保存 requests.Session.request 原始方法
        # 这是所有 requests 请求的核心方法（get/post/put/delete 都调用它）
        self._original_request = requests.Session.request

        @wraps(self._original_request)
        def monitored_request(session_self, method, url, **kwargs):
            start_time = time.time()

            # ===== 如果设置了代理，注入到请求参数中 =====
            if self._proxy:
                # requests 的 proxies 格式：{'http': proxy, 'https': proxy}
                kwargs['proxies'] = {
                    'http': self._proxy,
                    'https': self._proxy
                }

            # ===== 请求前：记录发送数据 =====
            send_size = 0
            if 'data' in kwargs:
                send_size = len(str(kwargs.get('data', '')))
            elif 'json' in kwargs:
                import json
                send_size = len(json.dumps(kwargs.get('json', {})))

            # 打印请求信息
            print(f"→ [{method}] {url}")
            if self._proxy:
                print(f"   代理: {self._proxy}")
            if send_size > 0:
                print(f"   发送: {send_size / 1024:.2f} KB")

            # ===== 调用原始请求方法 =====
            try:
                response = self._original_request(session_self, method, url, **kwargs)
            except Exception as e:
                elapsed = time.time() - start_time
                print(f"✗ 错误: {e} (耗时 {elapsed:.2f}s)")
                raise

            # ===== 响应后：记录接收数据 =====
            elapsed = time.time() - start_time
            recv_size = len(response.content) if response.content else 0

            # 更新统计（线程安全）
            with self._lock:
                self.stats['total_requests'] += 1
                self.stats['total_bytes_sent'] += send_size
                self.stats['total_bytes_recv'] += recv_size
                self.stats['requests'].append({
                    'url': url,
                    'method': method,
                    'send_bytes': send_size,
                    'recv_bytes': recv_size,
                    'elapsed': elapsed,
                    'status': response.status_code,
                    'time': time.strftime('%H:%M:%S')
                })

            # 打印响应信息
            size_str = f"{recv_size / 1024:.2f} KB" if recv_size < 1024 * 1024 else f"{recv_size / 1024 / 1024:.2f} MB"
            print(f"← {response.status_code} | {size_str} | {elapsed:.2f}s")

            return response

        # 应用 monkey-patch：替换 requests.Session.request
        requests.Session.request = monitored_request
        self._patched = True

        print("\n" + "=" * 50)
        print("[网络监控] 已启动 - 所有 requests 请求将被监控")
        if self._proxy:
            print(f"[代理] {self._proxy}")
        print("=" * 50 + "\n")

    def stop(self):
        """停止监控，恢复原始方法"""
        if not self._patched:
            return

        requests.Session.request = self._original_request
        self._patched = False
        print("[监控] 已停止")

    def report(self):
        """打印统计报告"""
        print("\n" + "=" * 60)
        print("                    网络请求统计报告")
        print("=" * 60)

        if self.stats['total_requests'] == 0:
            print("无请求记录")
            return

        print(f"总请求次数: {self.stats['total_requests']}")
        print(f"总接收流量: {self._format_size(self.stats['total_bytes_recv'])}")
        print(f"总发送流量: {self._format_size(self.stats['total_bytes_sent'])}")

        # 计算平均耗时
        total_time = sum(r['elapsed'] for r in self.stats['requests'])
        avg_time = total_time / len(self.stats['requests'])
        print(f"总耗时: {total_time:.2f}s | 平均: {avg_time:.2f}s")

        print("\n请求详情:")
        print("-" * 60)
        for r in self.stats['requests']:
            recv_str = self._format_size(r['recv_bytes'])
            print(f"  [{r['time']}] {r['method']} {r['url'][:45]}...")
            print(f"      状态: {r['status']} | 接收: {recv_str} | 耗时: {r['elapsed']:.2f}s")
        print("=" * 60)

    def _format_size(self, bytes):
        """格式化字节大小"""
        if bytes < 1024:
            return f"{bytes} B"
        elif bytes < 1024 * 1024:
            return f"{bytes / 1024:.2f} KB"
        else:
            return f"{bytes / 1024 / 1024:.2f} MB"

    def get_stats(self):
        """获取统计数据（供程序使用）"""
        return self.stats.copy()


# 全局单例实例
_monitor = NetworkMonitor()


def start():
    """启动监控"""
    _monitor.start()


def stop():
    """停止监控"""
    _monitor.stop()


def set_proxy(proxy: str):
    """设置代理"""
    _monitor.set_proxy(proxy)


def report():
    """打印报告"""
    _monitor.report()


def get_stats():
    """获取统计数据"""
    return _monitor.get_stats()


# ============================================================
#                    动态导入模块
# ============================================================

def load_script(script_path: str) -> object:
    """
    动态加载 Python 脚本为模块

    参数：
        script_path: 脚本文件路径

    返回：
        加载的模块对象
    """
    # 获取脚本绝对路径
    abs_path = os.path.abspath(script_path)

    if not os.path.exists(abs_path):
        raise FileNotFoundError(f"脚本不存在: {abs_path}")

    # 从文件名生成模块名
    module_name = os.path.splitext(os.path.basename(abs_path))[0]

    # 使用 importlib 动态加载
    spec = importlib.util.spec_from_file_location(module_name, abs_path)
    module = importlib.util.module_from_spec(spec)

    # 将模块添加到 sys.modules（支持模块内的相对导入）
    sys.modules[module_name] = module

    # 执行模块代码
    spec.loader.exec_module(module)

    return module


def run_function(module, func_name: str, func_args: list = None):
    """
    运行模块中的指定函数

    参数：
        module: 模块对象
        func_name: 函数名
        func_args: 函数参数列表
    """
    # 检查函数是否存在
    if not hasattr(module, func_name):
        raise AttributeError(f"模块中不存在函数: {func_name}")

    func = getattr(module, func_name)

    # 检查是否是可调用对象
    if not callable(func):
        raise TypeError(f"{func_name} 不是可调用对象")

    # 调用函数
    if func_args:
        # 如果参数以 -- 开头，作为命令行参数传递
        func(*func_args)
    else:
        func()


def run_module_main(module):
    """
    运行模块的 main 函数或执行模块顶层代码

    参数：
        module: 模块对象
    """
    # 优先查找 main 函数
    if hasattr(module, 'main') and callable(module.main):
        module.main()
    elif hasattr(module, 'execute') and callable(module.execute):
        module.execute()
    else:
        print("提示: 未找到 main/execute 函数，模块已加载但未执行特定函数")
        print("使用 -f 参数指定函数名，如: -f main")


# ============================================================
#                    主程序入口
# ============================================================

def main():
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(
        description="网络请求监控工具 - 监控目标脚本的网络流量，支持代理配置",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python monitor_wrapper.py script.py                          # 加载脚本，执行 main/execute
  python monitor_wrapper.py script.py -f download              # 执行指定函数
  python monitor_wrapper.py script.py -p http://127.0.0.1:7890 # 使用 HTTP 代理
  python monitor_wrapper.py script.py -p socks5://proxy:1080   # 使用 SOCKS5 代理
  python monitor_wrapper.py script.py -f run -a arg1           # 传递参数给函数
  python monitor_wrapper.py script.py -f main -p http://127.0.0.1:7890 -a --verbose

代理格式：
  http://host:port              HTTP 代理
  socks5://host:port            SOCKS5 代理（需要安装 PySocks: pip install pysocks）
  socks5://user:pass@host:port  SOCKS5 代理（带认证）
"""
    )

    parser.add_argument("script", help="目标脚本路径")
    parser.add_argument("-f", "--function", default=None,
                        help="要执行的函数名（默认执行 main 或 execute）")
    parser.add_argument("-a", "--args", nargs='*', default=None,
                        help="传递给函数的参数")
    parser.add_argument("-p", "--proxy", default=None,
                        help="代理地址，如 http://127.0.0.1:7890 或 socks5://proxy:1080")

    args = parser.parse_args()

    # ===== 第1步：配置代理（如果指定）=====
    if args.proxy:
        set_proxy(args.proxy)

    # ===== 第2步：启动监控 =====
    start()

    # ===== 第3步：动态加载目标脚本 =====
    try:
        print(f"加载脚本: {args.script}")
        module = load_script(args.script)
        print(f"模块已加载: {module.__name__}\n")
    except FileNotFoundError as e:
        print(f"✗ {e}")
        stop()
        return
    except Exception as e:
        print(f"✗ 加载脚本失败: {e}")
        stop()
        return

    # ===== 第4步：执行目标函数 =====
    try:
        if args.function:
            # 执行指定函数
            print(f"执行函数: {args.function}")
            if args.args:
                print(f"参数: {args.args}")
            run_function(module, args.function, args.args)
        else:
            # 执行默认函数（main 或 execute）
            run_module_main(module)
    except AttributeError as e:
        print(f"✗ {e}")
    except TypeError as e:
        print(f"✗ {e}")
    except Exception as e:
        print(f"✗ 执行函数时出错: {e}")

    # ===== 第5步：打印监控报告 =====
    report()

    # 停止监控
    stop()


if __name__ == "__main__":
    main()
