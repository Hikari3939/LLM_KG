import time
import requests
from bs4 import BeautifulSoup
from my_packages.GetPageText import get_page_text

BASE_URL = "https://zh.wikipedia.org"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/117.0.0.0 Safari/537.36"
}

# 全局配置变量
MAX_CRAWL_DEPTH = 1  # 默认爬取深度（0=主页面，1=第一层，2=第二层）
REQUEST_DELAY = 1    # 请求延迟（秒）
STOP_SECTIONS = ["参考文献", "延伸阅读", "参见", "外部链接"]

def stroke_scrape(max_depth=None):
    """
    爬取 '脑卒中' 主页面及相关页面
    :param max_depth: 最大爬取深度，如果为None则使用全局配置
    """
    # 使用参数或全局配置
    actual_max_depth = max_depth if max_depth is not None else MAX_CRAWL_DEPTH
    
    logs = []
    logs.append(f"开始爬取‘脑卒中’主页面（最大深度：{actual_max_depth}）…")
    
    stroke_url = BASE_URL + "/wiki/脑卒中"
    stroke_data, stroke_links = get_page_text(stroke_url)
    logs.append(f"主页面爬取完成，共 {len(stroke_data)} 条内容")

    # 提取主页面内链
    logs.append("提取主页面内链…")
    related_pages = dict(stroke_links)
    logs.append(f"找到 {len(related_pages)} 个相关页面。\n")

    results = {"脑卒中": (stroke_data, 0)}  # 存储内容和深度
    report = []

    # 队列存储待爬取页面：(title, url, depth)
    queue = []
    for title, url in related_pages.items():
        queue.append((title, url, 1))
    
    visited = set(["脑卒中"])

    idx = 0
    while queue:
        title, url, depth = queue.pop(0)

        if title in visited:
            continue
        visited.add(title)

        # 检查深度限制
        if depth > actual_max_depth:
            logs.append(f"跳过页面 {title}（深度{depth}超过限制{actual_max_depth}）")
            continue

        idx += 1
        total_discovered = len(related_pages)
        remaining = len([p for p in related_pages.items() if p[0] not in visited])
        log_entry = f"[{idx}/{remaining} of {total_discovered}] 深度{depth}：{title}"
        print(log_entry)
        logs.append(log_entry)

        try:
            page_data, page_links = get_page_text(url)
            results[title] = (page_data, depth)

            # 添加新发现的链接
            new_links_count = 0
            for t, u in page_links.items():
                if t not in related_pages:
                    related_pages[t] = u
                    new_links_count += 1
                    if t not in visited and depth + 1 <= actual_max_depth:
                        queue.append((t, u, depth + 1))

            success_msg = f"  -> 完成：{len(page_data)}条，深度{depth}，发现{new_links_count}新链接\n"
            print(success_msg)
            logs.append(success_msg.strip())
            report.append({
                "title": title,
                "items": len(page_data),
                "depth": depth,
                "status": "success",
                "new_links": new_links_count
            })
            time.sleep(REQUEST_DELAY)
        except Exception as e:
            error_msg = f"  -> 爬取失败（深度{depth}）: {e}\n"
            print(error_msg)
            logs.append(error_msg.strip())
            report.append({
                "title": title,
                "depth": depth,
                "status": "failed",
                "error": str(e)
            })

    return results, report, logs, related_pages
