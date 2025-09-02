#调用 get_page_text 获取页面内容，返回结构化数据和爬取报告
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

def scrape_stroke_and_symptoms():
    print("开始爬取‘脑卒中’主页面…")
    stroke_url = BASE_URL + "/wiki/脑卒中"
    stroke_data = get_page_text(stroke_url)
    print(f"主页面爬取完成，共 {len(stroke_data)} 条内容\n")

    print("提取主页面内链（可能的症状相关页面）…")
    resp = requests.get(stroke_url, headers=HEADERS)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    links = soup.select("div#mw-content-text a")

    related_pages = {}
    for a in links:
        href = a.get("href")
        if href and href.startswith("/wiki/") and not ":" in href:
            title = a.get_text(strip=True)
            if any(kw in title for kw in ["症", "瘫", "失语", "头痛", "晕", "麻木", "感觉"]):
                related_pages[title] = BASE_URL + href

    print(f"找到 {len(related_pages)} 个症状相关页面。\n")

    results = {"脑卒中": stroke_data}
    report = []

    for idx, (title, url) in enumerate(related_pages.items(), 1):
        print(f"[{idx}/{len(related_pages)}] 正在爬取页面：{title} -> {url}")
        try:
            page_data = get_page_text(url)
            results[title] = page_data
            num_titles = sum(1 for item in page_data if item['type']=='title')
            num_paragraphs = sum(1 for item in page_data if item['type']=='paragraph')
            print(f"  -> 爬取完成：{num_titles} 个标题，{num_paragraphs} 个段落\n")
            report.append({
                "title": title,
                "url": url,
                "titles": num_titles,
                "paragraphs": num_paragraphs,
                "status": "success"
            })
            time.sleep(1)  # 避免请求过快
        except Exception as e:
            print(f"  -> 爬取失败: {e}\n")
            report.append({
                "title": title,
                "url": url,
                "status": "failed",
                "error": str(e)
            })

    return results, report