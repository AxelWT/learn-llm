"""
图片压缩工具

功能：
    - 支持单文件压缩
    - 支持批量压缩目录中的图片
    - 支持命令行参数指定输入/输出路径和目标大小
    - 自动生成输出文件名（添加 _compressed 后缀）
    - 友好的错误提示

支持格式：jpg, jpeg, png, webp, bmp, tiff

使用示例：
    python test.py photo.jpg                    # 压缩单个文件
    python test.py photo.jpg -o small.jpg       # 指定输出文件名
    python test.py photo.jpg -s 500             # 目标大小 500KB
    python test.py ./images/                    # 批量压缩目录
    python test.py ./images/ -o ./compressed/   # 批量压缩到指定目录
"""

from PIL import Image, UnidentifiedImageError
import argparse
from pathlib import Path

# 支持的图片格式集合
SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}


def compress_single(input_path: Path, output_path: Path, target_size_kb: int) -> bool:
    """
    压缩单个图片文件

    通过逐步降低 JPEG 质量参数，将图片压缩到目标大小以内。

    参数：
        input_path: 输入图片文件路径
        output_path: 输出图片文件路径
        target_size_kb: 目标文件大小（KB）

    返回：
        True 表示压缩成功，False 表示压缩失败

    压缩策略：
        1. 从 quality=95 开始保存图片
        2. 如果文件大小超过目标，每次降低 quality 5
        3. 直到文件大小满足目标或 quality 降至 10
    """
    try:
        # 打开原始图片
        img = Image.open(input_path)

        # 转换为 RGB 模式
        # JPEG 格式不支持透明度，RGBA（带透明通道）和 P（调色板模式）需要转换
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")

        # 初始质量设为 95（高质量）
        quality = 95
        img.save(output_path, "JPEG", quality=quality, optimize=True)

        # 循环降低质量直到满足目标大小
        # optimize=True 启用 Huffman 表优化，可进一步减小文件大小
        while output_path.stat().st_size > target_size_kb * 1024 and quality > 10:
            quality -= 5
            img.save(output_path, "JPEG", quality=quality, optimize=True)

        # 输出压缩结果
        final_size_kb = output_path.stat().st_size / 1024
        print(f"✓ {input_path.name} -> {output_path.name}: {final_size_kb:.2f} KB (quality={quality})")
        return True

    except FileNotFoundError:
        # 文件不存在错误
        print(f"✗ 文件不存在: {input_path}")
        return False
    except UnidentifiedImageError:
        # PIL 无法识别图片格式（可能是损坏的文件或非图片文件）
        print(f"✗ 无法识别图片格式: {input_path}")
        return False
    except Exception as e:
        # 其他未知错误
        print(f"✗ 处理失败: {input_path} - {e}")
        return False


def compress_batch(input_dir: Path, output_dir: Path, target_size_kb: int) -> tuple[int, int]:
    """
    批量压缩目录中的所有图片

    遍历输入目录，对每个支持的图片文件进行压缩，
    输出到指定目录，文件名添加 _compressed 后缀。

    参数：
        input_dir: 输入目录路径
        output_dir: 输出目录路径
        target_size_kb: 目标文件大小（KB）

    返回：
        (成功数, 失败数) 的元组
    """
    # 创建输出目录（如果不存在）
    # parents=True 表示同时创建父目录
    # exist_ok=True 表示目录已存在时不报错
    output_dir.mkdir(parents=True, exist_ok=True)

    # 统计成功和失败数量
    success = 0
    failed = 0

    # 遍历输入目录中的所有文件
    for file in input_dir.iterdir():
        # 只处理文件（跳过子目录）且扩展名在支持格式列表中
        if file.is_file() and file.suffix.lower() in SUPPORTED_FORMATS:
            # 生成输出文件名：原名_compressed.jpg
            output_path = output_dir / f"{file.stem}_compressed.jpg"

            # 调用单文件压缩函数
            if compress_single(file, output_path, target_size_kb):
                success += 1
            else:
                failed += 1

    return success, failed


def main():
    """
    主函数：解析命令行参数并执行压缩任务

    命令行参数：
        input: 必需参数，输入文件或目录路径
        -o/--output: 可选参数，输出文件或目录路径
        -s/--size: 可选参数，目标大小 KB（默认 1024）
    """
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(
        description="图片压缩工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python test.py photo.jpg                    # 压缩单个文件
  python test.py photo.jpg -o small.jpg       # 指定输出文件名
  python test.py photo.jpg -s 500             # 目标大小 500KB
  python test.py ./images/                    # 批量压缩目录
  python test.py ./images/ -o ./compressed/   # 批量压缩到指定目录
"""
    )

    # 定义命令行参数
    parser.add_argument("input", help="输入文件或目录路径")
    parser.add_argument("-o", "--output", help="输出文件或目录路径（可选）")
    parser.add_argument("-s", "--size", type=int, default=1024,
                        help="目标大小 KB（默认 1024）")

    # 解析命令行参数
    args = parser.parse_args()

    # 将输入路径转换为 Path 对象
    input_path = Path(args.input)

    # 根据输入路径类型选择处理方式
    if input_path.is_file():
        # ===== 单文件处理 =====
        # 如果指定了输出路径则使用，否则自动生成
        if args.output:
            output_path = Path(args.output)
        else:
            # 自动生成：在原目录下，添加 _compressed 后缀
            output_path = input_path.with_name(f"{input_path.stem}_compressed.jpg")

        # 执行单文件压缩
        compress_single(input_path, output_path, args.size)

    elif input_path.is_dir():
        # ===== 批量处理 =====
        # 如果指定了输出目录则使用，否则在输入目录下创建 compressed 子目录
        if args.output:
            output_dir = Path(args.output)
        else:
            output_dir = input_path / "compressed"

        # 显示批量处理信息
        print(f"批量压缩: {input_path} -> {output_dir} (目标 {args.size}KB)")

        # 执行批量压缩
        success, failed = compress_batch(input_path, output_dir, args.size)

        # 显示统计结果
        print(f"完成: {success} 成功, {failed} 失败")

    else:
        # 输入路径既不是文件也不是目录（可能不存在）
        print(f"✗ 路径不存在: {input_path}")


# 程序入口点
# 只有直接运行此脚本时才执行 main()，作为模块导入时不执行
if __name__ == "__main__":
    main()
