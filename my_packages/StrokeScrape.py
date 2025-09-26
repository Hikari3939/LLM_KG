import time
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

def stroke_scrape():

    actual_max_depth = MAX_CRAWL_DEPTH
    
    logs = []
    logs.append(f"开始爬取：脑卒中（最大深度：{actual_max_depth}）")

    # ====== 深度0：主页面 ======
    stroke_url = BASE_URL + "/wiki/脑卒中"
    stroke_data, stroke_links = get_page_text(stroke_url)

    results = {"脑卒中": (stroke_data, 0)}  
    related_pages = dict(stroke_links)

    print(f"[深度0] 正在抓取：脑卒中")
    print(f"抓取成功：正文段落 {len(stroke_data)} 条，新发现链接 {len(stroke_links)} 个\n")
    logs.append(f"[深度0] 抓取成功：正文段落 {len(stroke_data)} 条，新发现链接 {len(stroke_links)} 个")

    # 队列存储待爬取页面：(title, url, depth)
    queue = [(title, url, 1) for title, url in related_pages.items()]
    visited = set(["脑卒中"])

    # ====== 其他页面 ======
    while queue:
        title, url, depth = queue.pop(0)

        if title in visited:
            continue
        visited.add(title)

        if depth > actual_max_depth:
            skip_msg = f"跳过：{title}（深度 {depth} > 限制 {actual_max_depth}）"
            logs.append(skip_msg)
            continue

        print(f"[深度{depth}] 正在抓取：{title}")
        logs.append(f"[深度{depth}] 正在抓取：{title}")

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

            success_msg = (
                f"抓取成功：正文段落 {len(page_data)} 条，"
                f"新发现链接 {new_links_count} 个\n"
            )
            print(success_msg)
            logs.append(success_msg.strip())
        except Exception as e:
            error_msg = f"抓取失败（深度{depth}）：{e}\n"
            print(error_msg)
            logs.append(error_msg.strip())

        time.sleep(REQUEST_DELAY)

    # ====== 爬取结束统计 ======
    final_summary = (
        f"爬取结束！\n"
        f"页面总数：{len(results)}\n"
        f"正文条目：{sum(len(v[0]) for v in results.values())}\n"
        f"内链总数：{len(related_pages)}"
    )
    print(final_summary)
    logs.append(final_summary)

    return results, logs, related_pages
