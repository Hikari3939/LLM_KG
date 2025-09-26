# 用于爬取单个 Wikipedia 页面（正文文本 + 正文内链）
import requests
from bs4 import BeautifulSoup
import zhconv

BASE_URL = "https://zh.wikipedia.org"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/117.0.0.0 Safari/537.36"
}

# 不需要的末尾章节标题
STOP_SECTIONS = ["参考文献", "延伸阅读", "参见", "外部链接"]

def get_page_text(url):
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
