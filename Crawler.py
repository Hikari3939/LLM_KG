import os
import re
from my_packages.StrokeScrape import stroke_scrape

def sanitize_filename(filename):
    """清理文件名，移除非法字符"""
    illegal_chars = r'[<>:"/\\|?*]'
    return re.sub(illegal_chars, '_', filename)

if __name__ == "__main__":
    # 爬取深度0和1的内容
    results, report, logs, related_pages = stroke_scrape(max_depth=1)

    output_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(output_dir, exist_ok=True)

    # 创建深度文件夹
    depth0_dir = os.path.join(output_dir, "脑卒中_深度0")
    os.makedirs(depth0_dir, exist_ok=True)
    depth1_dir = os.path.join(output_dir, "脑卒中_深度1")
    os.makedirs(depth1_dir, exist_ok=True)

    # 保存每个页面的独立文件
    for page_name, (content, depth) in results.items():
        # 清理文件名
        safe_filename = sanitize_filename(page_name) + ".txt"
        
        if depth == 0:
            # 保存到深度0文件夹
            file_path = os.path.join(depth0_dir, safe_filename)
        elif depth == 1:
            # 保存到深度1文件夹
            file_path = os.path.join(depth1_dir, safe_filename)
        else:
            continue  # 跳过其他深度
        
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"页面标题: {page_name}\n")
            f.write(f"深度: {depth}\n")
            f.write("=" * 50 + "\n\n")
            f.write("\n".join(content))

    print(f"\n爬取完成！")