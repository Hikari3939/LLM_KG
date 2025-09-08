# 用于爬取单个 Wikipedia 页面（正文文本 + 正文内链）
import requests
from bs4 import BeautifulSoup
from opencc import OpenCC
cc = OpenCC('t2s')  # 繁体转简体

BASE_URL = "https://zh.wikipedia.org"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/117.0.0.0 Safari/537.36"
}

# 不需要的末尾章节标题
STOP_SECTIONS = ["参考文献", "延伸阅读", "参见", "外部链接"]

from opencc import OpenCC
cc = OpenCC('t2s')  # 繁体转简体

def get_page_text(url):
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    content_div = soup.find("div", {"id": "mw-content-text"})
    data = []
    links = {}

    for elem in content_div.find_all(["p", "h2", "h3"]):
        text = cc.convert(elem.get_text(strip=True).replace("[编辑]", ""))
        if not text:
            continue
        if any(stop in text for stop in STOP_SECTIONS):
            break
        data.append(text)

        for a in elem.find_all("a"):
            href = a.get("href")
            if href and href.startswith("/wiki/") and ":" not in href:
                title = a.get("title") or a.get_text(strip=True)
                if title:
                    title = cc.convert(title)
                    links[title] = BASE_URL + href.split("#")[0]

    return data, links
