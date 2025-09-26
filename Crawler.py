import os
import re
from my_packages.StrokeScrape import stroke_scrape
from my_packages.StrokeScrape import MAX_CRAWL_DEPTH

def sanitize_filename(filename):
    """清理文件名，移除非法字符"""
    illegal_chars = r'[<>:"/\\|?*]'
    return re.sub(illegal_chars, '_', filename)

if __name__ == "__main__":
    # 设置爬取深度
    results, report, logs, related_pages = stroke_scrape()

    output_dir = os.path.join(os.path.dirname(__file__), "data")

    # 动态创建深度文件夹
    depth_dirs = {}
    for depth in range(MAX_CRAWL_DEPTH + 1):  # +1 因为深度从0开始
        folder_name = f"脑卒中_深度{depth}"
        folder_path = os.path.join(output_dir, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        depth_dirs[depth] = folder_path

    # 保存每个页面到对应深度文件夹
    for page_name, (content, depth) in results.items():
        # 清理文件名
        safe_filename = sanitize_filename(page_name) + ".txt"

        if depth not in depth_dirs:
            continue

        file_path = os.path.join(depth_dirs[depth], safe_filename)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"页面标题: {page_name}\n")
            f.write(f"深度: {depth}\n")
            f.write("=" * 50 + "\n\n")
            f.write("\n".join(content))

    print(f"\n爬取完成！")
