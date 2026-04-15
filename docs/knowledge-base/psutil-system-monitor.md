# Python psutil 包总结

## 一、概述

**psutil** (Python system and process utilities) 是一个跨平台库，用于检索运行进程和系统利用率信息，包括 CPU、内存、磁盘、网络和传感器。

**主要用途：**
- 系统监控
- 性能分析
- 进程资源限制
- 运行进程管理

**特点：**
- 实现了类似 UNIX 命令行工具的功能（如 ps、top、free、iotop、netstat、lsof）
- 提供跨平台的统一接口
- 支持所有主流操作系统（Windows、Linux、macOS 等）

---

## 二、安装

```bash
pip install psutil
```

---

## 三、主要功能模块

### 1. CPU 监控

```python
import psutil

# CPU 使用率
psutil.cpu_percent(interval=1)          # 总体 CPU 使用率
psutil.cpu_percent(interval=1, percpu=True)  # 每个 CPU 核使用率

# CPU 核心数
psutil.cpu_count()           # 逻辑核心数
psutil.cpu_count(logical=False)  # 物理核心数

# CPU 时间统计
psutil.cpu_times()           # CPU 时间分布

# CPU 频率
psutil.cpu_freq()            # CPU 频率信息

# 系统负载（仅 Linux/Unix）
psutil.getloadavg()          # 返回 (1分钟, 5分钟, 15分钟) 负载
```

### 2. 内存监控

```python
import psutil

mem = psutil.virtual_memory()

print(f"总内存: {mem.total / (1024**3):.1f} GB")
print(f"可用内存: {mem.available / (1024**3):.1f} GB")
print(f"已用内存: {mem.used / (1024**3):.1f} GB")
print(f"使用率: {mem.percent}%")

# 内存告警示例
THRESHOLD = 500 * 1024 * 1024  # 500 MB
if mem.available <= THRESHOLD:
    print("警告: 内存不足!")

# 交换内存
swap = psutil.swap_memory()
print(f"交换内存总量: {swap.total / (1024**3):.1f} GB")
print(f"交换内存使用率: {swap.percent}%")
```

### 3. 磁盘监控

```python
import psutil

# 磁盘使用情况
disk = psutil.disk_usage('/')
print(f"总容量: {disk.total / (1024**3):.1f} GB")
print(f"已用: {disk.used / (1024**3):.1f} GB")
print(f"可用: {disk.free / (1024**3):.1f} GB")
print(f"使用率: {disk.percent}%")

# 磁盘分区列表
psutil.disk_partitions()

# 磁盘 I/O 统计
io = psutil.disk_io_counters()
print(f"读取: {io.read_bytes / (1024**2):.1f} MB")
print(f"写入: {io.write_bytes / (1024**2):.1f} MB")
```

### 4. 网络监控

```python
import psutil
import socket

# 网络接口地址
for iface, addrs in psutil.net_if_addrs().items():
    for addr in addrs:
        if addr.family == socket.AF_INET:
            print(f"{iface}: {addr.address}")

# 网络 I/O 统计
net = psutil.net_io_counters()
print(f"发送: {net.bytes_sent / (1024**2):.1f} MB")
print(f"接收: {net.bytes_recv / (1024**2):.1f} MB")

# 每个接口的网络 I/O
net_per_nic = psutil.net_io_counters(pernic=True)

# 活动网络连接
for conn in psutil.net_connections(kind='tcp'):
    print(f"{conn.laddr} -> {conn.raddr} status={conn.status} pid={conn.pid}")

# 网络连接实时监控
def monitor_network():
    before = psutil.net_io_counters(pernic=True)
    time.sleep(1)
    after = psutil.net_io_counters(pernic=True)
    for iface in after:
        sent = after[iface].bytes_sent - before[iface].bytes_sent
        recv = after[iface].bytes_recv - before[iface].bytes_recv
        print(f"{iface}: 发送 {sent}/s, 接收 {recv}/s")
```

### 5. 进程管理

```python
import psutil

# 当前所有进程
for proc in psutil.process_iter(['pid', 'name', 'cpu_percent']):
    print(f"PID: {proc.info['pid']}, 名称: {proc.info['name']}")

# 获取特定进程
p = psutil.Process(pid)

# 进程信息
p.name()              # 进程名
p.exe()               # 可执行文件路径
p.cwd()               # 工作目录
p.cmdline()           # 命令行参数
p.pid                 # 进程 ID
p.ppid()              # 父进程 ID
p.status()            # 进程状态
p.create_time()       # 创建时间

# 进程资源使用
p.cpu_percent()       # CPU 使用率
p.memory_info()       # 内存信息 (rss, vms 等)
p.memory_percent()    # 内存使用百分比
p.io_counters()       # I/O 统计

# 进程控制
p.terminate()         # 终止进程
p.kill()              # 强制杀死进程
p.wait()              # 等待进程结束

# 进程监控（优化版）
def monitor_process(pid, duration=10):
    p = psutil.Process(pid)
    for _ in range(duration):
        with p.oneshot():  # 优化多次属性访问
            cpu = p.cpu_percent()
            mem = p.memory_info().rss / (1024**2)
            print(f"CPU: {cpu}%, 内存: {mem:.1f} MB")
        time.sleep(1)
```

### 6. 系统信息

```python
import psutil

# 系统启动时间
psutil.boot_time()

# 用户信息
for user in psutil.users():
    print(f"用户: {user.name}, 终端: {user.terminal}")

# 传感器温度（仅 Linux）
psutil.sensors_temperatures()

# 电池状态（仅笔记本）
psutil.sensors_battery()
```

---

## 四、完整示例：系统监控脚本

```python
import psutil
import time
from datetime import datetime

def system_monitor():
    """综合系统监控"""
    print("=" * 50)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)

    # CPU
    cpu_percent = psutil.cpu_percent(interval=1)
    cpu_count = psutil.cpu_count()
    print(f"\n[CPU]")
    print(f"  核心数: {cpu_count} (逻辑)")
    print(f"  使用率: {cpu_percent}%")

    # 内存
    mem = psutil.virtual_memory()
    print(f"\n[内存]")
    print(f"  总量: {mem.total / (1024**3):.1f} GB")
    print(f"  已用: {mem.used / (1024**3):.1f} GB ({mem.percent}%)")
    print(f"  可用: {mem.available / (1024**3):.1f} GB")

    # 磁盘
    disk = psutil.disk_usage('/')
    print(f"\n[磁盘]")
    print(f"  总量: {disk.total / (1024**3):.1f} GB")
    print(f"  已用: {disk.used / (1024**3):.1f} GB ({disk.percent}%)")

    # 网络
    net = psutil.net_io_counters()
    print(f"\n[网络]")
    print(f"  发送: {net.bytes_sent / (1024**2):.1f} MB")
    print(f"  接收: {net.bytes_recv / (1024**2):.1f} MB")

    # 进程数
    process_count = len(psutil.pids())
    print(f"\n[进程]")
    print(f"  运行进程数: {process_count}")

    print("=" * 50)

# 运行监控
if __name__ == "__main__":
    system_monitor()

    # 持续监控
    # while True:
    #     system_monitor()
    #     time.sleep(5)
```

---

## 五、常用场景

| 场景 | 推荐功能 |
|------|---------|
| 系统健康监控 | `cpu_percent()`, `virtual_memory()`, `disk_usage()` |
| 进程追踪 | `process_iter()`, `Process.cpu_percent()` |
| 网络流量分析 | `net_io_counters()`, `net_connections()` |
| 资源告警 | 监控内存、磁盘使用率是否超过阈值 |
| 性能瓶颈分析 | 综合监控 CPU、内存、I/O |

---

## 六、注意事项

1. **跨平台兼容性**: 某些功能仅在特定平台可用（如 `getloadavg()` 仅 Linux/Unix）
2. **权限要求**: 某些进程信息需要管理员/root 权限
3. **性能优化**: 使用 `Process.oneshot()` 优化多次属性访问
4. **单位转换**: 返回值通常是字节，需自行转换为 GB/MB
5. **interval 参数**: `cpu_percent()` 需要间隔参数才能准确测量

---

## 七、参考资源

- 官方文档: https://psutil.readthedocs.io/
- GitHub: https://github.com/giampaolo/psutil
- PyPI: https://pypi.org/project/psutil/