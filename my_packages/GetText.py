import requests
requests.packages.urllib3.disable_warnings()
import re
import time
import socket
import http.client
import ssl
from pyquery import PyQuery

def get(url, params=None, headers=None, cookies=None, proxies=None, verify=True, 
        timeout=10, allow_redirects=True, sleep=1, try_num=0):
    if headers is None:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/101.0.4951.7 Safari/537.36'
        }
    try_index = 0
    while True:
        try_index += 1
        if try_num != 0 and try_index >= try_num:
            return None
        try:
            response = requests.get(url=url, params=params, headers=headers, 
                                  cookies=cookies, proxies=proxies, verify=verify, 
                                  timeout=timeout, allow_redirects=allow_redirects)
            response.encoding = response.apparent_encoding
            break
        except (requests.exceptions.ProxyError, requests.exceptions.ConnectTimeout,
                requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError,
                requests.exceptions.ChunkedEncodingError, requests.exceptions.TooManyRedirects,
                ConnectionResetError, socket.timeout, http.client.IncompleteRead, 
                ssl.SSLWantReadError) as e:
            print(f'get——{type(e).__name__}')
            time.sleep(sleep)
            continue
    return response

def pq(text):
    if not text or not text.strip():
        return PyQuery('<div></div>')
    try:
        html = PyQuery(text)
        return html
    except Exception as e:
        print(f'PyQuery解析错误: {e}')
        return PyQuery('<div></div>')

def re_title(title):
    illegal_chars = r'[<>:"/\\|?*\x00-\x1f\x7f]'
    title = re.sub(illegal_chars, '', title)
    title = title.replace('\xa0', ' ')
    title = re.sub(r'[\u200b-\u200f\u202a-\u202e]', '', title)
    title = re.sub(r'\\\\+', r'\\', title)
    title = title.strip('. ')
    title = re.sub(r'\s+', ' ', title)
    return title.strip()

