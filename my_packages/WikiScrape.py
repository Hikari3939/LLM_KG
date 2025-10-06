from bs4 import BeautifulSoup
import requests
import zhconv
import time
import os
import re

# 全局变量
BASE_URL = "https://zh.wikipedia.org"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/117.0.0.0 Safari/537.36"
}
MAX_CRAWL_DEPTH = 1  # 默认爬取深度（0=主页面，1=第一层，2=第二层）
REQUEST_DELAY = 1    # 请求延迟（秒）
STOP_SECTIONS = ["参考文献", "延伸阅读", "参见", "外部链接"]

# 清理文件名，移除非法字符
def sanitize_filename(filename):
    illegal_chars = r'[<>:"/\\|?*]'
    return re.sub(illegal_chars, '_', filename)

# 爬取单个wiki页面
def get_wiki_page(url):
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status() # 失败处理

    soup = BeautifulSoup(resp.text, "html.parser")
    content_div = soup.find("div", {"id": "mw-content-text"}) # 选择正文
    data = []
    links = {}

    for elem in content_div.find_all(["p", "h2", "h3"]):
        # 删除参考文献标注（<sup class="reference">）
        for sup in elem.find_all("sup", {"class": "reference"}):
            sup.decompose()
        text = zhconv.convert(elem.get_text(strip=True).replace("[编辑]", ""), "zh-cn")
        if not text:
            continue
        if any(stop in text for stop in STOP_SECTIONS):
            break
        data.append(text) # 将文本加入正文列表

        for a in elem.find_all("a"): # 遍历段落中的 <a> 标签（超链接)
            href = a.get("href")
            if href and href.startswith("/wiki/") and ":" not in href:
                title = a.get("title") or a.get_text(strip=True)
                if title:
                    title = zhconv.convert(title, "zh-cn")
                    links[title] = BASE_URL + href.split("#")[0]

    return data, links

# 爬取脑卒中相关wiki页面
def wiki_scrape():
    actual_max_depth = MAX_CRAWL_DEPTH
    
    logs = []
    logs.append(f"开始爬取：脑卒中（最大深度：{actual_max_depth}）")

    # ====== 深度0：主页面 ======
    stroke_url = BASE_URL + "/wiki/脑卒中"
    stroke_data, stroke_links = get_wiki_page(stroke_url)

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
            page_data, page_links = get_wiki_page(url)
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

# 保存爬取的wiki数据
def save_wiki_data(results, save_dir):
    # 创建保存目录
    save_dir = os.path.join(save_dir, "wiki_data")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 动态创建深度文件夹 
    depth_dirs = {} 
    for depth in range(MAX_CRAWL_DEPTH + 1):  # +1 因为深度从0开始 
        folder_name = f"depth_{depth}" 
        folder_path = os.path.join(save_dir, folder_name) 
        os.makedirs(folder_path, exist_ok=True) 
        depth_dirs[depth] = folder_path

    # 保存每个页面到对应深度的文件夹
    for page_name, (content, depth) in results.items():
        # 清理文件名
        safe_filename = "wiki_" + sanitize_filename(page_name) + ".txt"

        # 使用对应深度文件夹
        if depth in depth_dirs:
            file_path = os.path.join(depth_dirs[depth], safe_filename)
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("\n".join(content))

    print(f"\n爬取完成！")
