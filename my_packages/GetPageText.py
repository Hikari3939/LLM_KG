#用于爬取单个 Wikipedia 页面
import requests
from bs4 import BeautifulSoup

BASE_URL = "https://zh.wikipedia.org"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/117.0.0.0 Safari/537.36"
}

def get_page_text(url):
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    content_div = soup.find("div", {"id": "mw-content-text"})
    data = []
    for elem in content_div.find_all(["p", "h2", "h3"]):
        if elem.name in ["h2", "h3"]:
            title = elem.get_text(strip=True).replace("[编辑]", "")
            data.append({"type": "title", "text": title})
        elif elem.name == "p":
            text = elem.get_text(strip=True)
            if text:
                data.append({"type": "paragraph", "text": text})
    return data
