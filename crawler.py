import os
import re
from my_packages.StrokeScrape import stroke_scrape
from my_packages.StrokeScrape import MAX_CRAWL_DEPTH

# 指定数据保存路径
DIRECTORY_PATH = './data'

def sanitize_filename(filename):
    """清理文件名，移除非法字符"""
    illegal_chars = r'[<>:"/\\|?*]'
    return re.sub(illegal_chars, '_', filename)

if __name__ == "__main__":
    # 爬取页面
    results, report, logs = stroke_scrape()

# 动态创建深度文件夹 
depth_dirs = {} 
for depth in range(MAX_CRAWL_DEPTH + 1):  # +1 因为深度从0开始 
    folder_name = f"depth_{depth}" 
    folder_path = os.path.join(DIRECTORY_PATH, folder_name) 
    os.makedirs(folder_path, exist_ok=True) 
    depth_dirs[depth] = folder_path

# 保存每个页面到对应深度的文件夹
for page_name, (content, depth) in results.items():
    # 清理文件名
    safe_filename = sanitize_filename(page_name) + ".txt"

    # 使用对应深度文件夹
    if depth in depth_dirs:
        file_path = os.path.join(depth_dirs[depth], safe_filename)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(content))

print(f"\n爬取完成！")
