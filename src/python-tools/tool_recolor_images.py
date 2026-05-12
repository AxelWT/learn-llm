"""
图片整体着色脚本 - 保留原图明暗关系，将图片着色为指定颜色

用法:
    python recolor.py <图片路径> <颜色名称> [--output 输出路径]

示例:
    python recolor.py photo.png gray
    python recolor.py photo.png red --output result.png
"""

import argparse
import sys
from pathlib import Path

from PIL import Image

# 颜色名称 -> RGB 映射
# ────────────────────────────────────────────────────────────
#  中文名        │ 键名          │ RGB
# ────────────────────────────────────────────────────────────
COLOR_MAP = {
    # 灰色系
    "black": (0, 0, 0),              # 黑
    "darkgray": (64, 64, 64),        # 深灰
    "gray": (128, 128, 128),         # 中灰
    "midgray": (128, 128, 128),      # 中灰（别名）
    "lightgray": (192, 192, 192),    # 浅灰
    "white": (255, 255, 255),        # 白
    # 红色系
    "darkred": (139, 0, 0),          # 深红
    "red": (255, 0, 0),              # 红
    "coral": (255, 127, 80),         # 珊瑚
    "pink": (255, 192, 203),         # 粉
    # 绿色系
    "darkgreen": (0, 100, 0),        # 深绿
    "green": (0, 128, 0),            # 绿
    "olive": (128, 128, 0),          # 橄榄
    "teal": (0, 128, 128),           # 青绿
    # 蓝色系
    "navy": (0, 0, 128),             # 海军蓝
    "darkblue": (0, 0, 139),         # 深蓝
    "blue": (0, 0, 255),             # 蓝
    "cyan": (0, 255, 255),           # 青
    # 黄/橙色系
    "yellow": (255, 255, 0),         # 黄
    "gold": (255, 215, 0),           # 金
    "orange": (255, 165, 0),         # 橙
    # 紫/品红色系
    "purple": (128, 0, 128),         # 紫
    "magenta": (255, 0, 255),        # 品红
    # 棕色系
    "brown": (139, 69, 19),          # 棕
}


def colorize(image: Image.Image, target_rgb: tuple[int, int, int]) -> Image.Image:
    """将图片着色为目标颜色，保留原图明暗关系。

    原理：将原图转为灰度获取亮度，再用亮度对目标颜色进行缩放。
    """
    gray = image.convert("L")
    r, g, b = target_rgb

    result = Image.new("RGB", image.size)
    pixels = result.load()
    gray_pixels = gray.load()

    for y in range(image.height):
        for x in range(image.width):
            luminance = gray_pixels[x, y] / 255.0
            pixels[x, y] = (
                int(r * luminance),
                int(g * luminance),
                int(b * luminance),
            )

    return result


def parse_color(color_str: str) -> tuple[int, int, int]:
    """解析颜色名称或十六进制色值为 RGB 元组。"""
    # 支持十六进制色值（如 #808080）
    if color_str.startswith("#"):
        hex_str = color_str.lstrip("#")
        if len(hex_str) != 6:
            print(f"错误: 十六进制色值格式应为 #RRGGBB，收到 #{hex_str}")
            sys.exit(1)
        return tuple(int(hex_str[i : i + 2], 16) for i in (0, 2, 4))

    # 颜色名称查找（不区分大小写）
    key = color_str.lower().replace(" ", "")
    if key in COLOR_MAP:
        return COLOR_MAP[key]

    print(f"错误: 未知颜色名称 '{color_str}'")
    print(f"可用颜色: {', '.join(sorted(COLOR_MAP.keys()))}")
    print("或使用十六进制色值，如 #808080")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="图片整体着色 - 保留明暗关系")
    parser.add_argument("image", help="输入图片路径")
    parser.add_argument("color", help="目标颜色名称或十六进制色值（如 gray, #808080）")
    parser.add_argument("--output", "-o", help="输出图片路径（默认: <原文件名>_<颜色>.png）")

    args = parser.parse_args()

    src = Path(args.image)
    if not src.exists():
        print(f"错误: 文件不存在 '{src}'")
        sys.exit(1)

    target_rgb = parse_color(args.color)

    img = Image.open(src).convert("RGB")
    result = colorize(img, target_rgb)

    # 确定输出路径
    if args.output:
        out_path = Path(args.output)
    else:
        out_path = src.with_stem(f"{src.stem}_{args.color}").with_suffix(".png")

    result.save(out_path)
    print(f"已完成: {out_path}")


if __name__ == "__main__":
    main()
