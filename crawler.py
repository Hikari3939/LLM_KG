import os
import re
from my_packages.StrokeScrape import stroke_scrape

# 指定数据保存路径
DIRECTORY_PATH = './data'

def sanitize_filename(filename):
    """清理文件名，移除非法字符"""
    illegal_chars = r'[<>:"/\\|?*]'
    return re.sub(illegal_chars, '_', filename)

if __name__ == "__main__":
    # 爬取页面
    results, report, logs = stroke_scrape()

    # 保存每个页面到文件夹
    for page_name, (content, depth) in results.items():
        # 清理文件名
        safe_filename = sanitize_filename(page_name) + ".txt"

        # 保存文件
        file_path = os.path.join(DIRECTORY_PATH, safe_filename)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(content))

    print(f"\n爬取完成！")
