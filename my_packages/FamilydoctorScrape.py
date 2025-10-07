import os
import re
import ssl
import time
import socket
import requests
import http.client
from pyquery import PyQuery
from typing import Optional, Dict, Any

# 禁用SSL警告
requests.packages.urllib3.disable_warnings()

# 全局变量
MAX_PAGES = 30  # 默认爬取页数
HEADERS = {
    'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,'
                'image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
    'accept-encoding': 'gzip, deflate, br, zstd',
    'accept-language': 'zh-CN,zh;q=0.9',
    'cookie': 'acw_tc=6f0caf1817596269988003281eea3f56f105fc4fdefacdda521f0f1d05; '
                'cdn_sec_tc=6f0caf1817596269988003281eea3f56f105fc4fdefacdda521f0f1d05; '
                'Hm_lvt_f46dd4cc550b93aefde9b00265bb533d=1759627024; '
                'HMACCOUNT=280E67F5FB89C93F; '
                'ASP.NET_SessionId=bsnsnjmk2fpszlgxwfakiax3; '
                'Hm_lpvt_f46dd4cc550b93aefde9b00265bb533d=1759627167',
    'priority': 'u=0, i',
    'referer': 'https://so.familydoctor.com.cn/search?q=%E8%84%91%E5%8D%92%E4%B8%AD&t=info',
    'sec-ch-ua': '"Not A(Brand";v="8", "Chromium";v="132", "Google Chrome";v="132"',
    'sec-ch-ua-mobile': '?0',
    'sec-ch-ua-platform': '"Windows"',
    'sec-fetch-dest': 'document',
    'sec-fetch-mode': 'navigate',
    'sec-fetch-site': 'same-origin',
    'sec-fetch-user': '?1',
    'upgrade-insecure-requests': '1',
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                    '(KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36'
}

def get_http_response(
    url: str,
    params: Optional[Dict] = None,
    headers: Optional[Dict] = None,
    cookies: Optional[Dict] = None,
    proxies: Optional[Dict] = None,
    verify: bool = True,
    timeout: int = 10,
    allow_redirects: bool = True,
    sleep: int = 1,
    max_retries: int = 0
) -> Optional[requests.Response]:
    """
    发送HTTP GET请求，包含重试机制
    
    Args:
        url: 请求URL
        params: 查询参数
        headers: 请求头
        cookies: cookies
        proxies: 代理设置
        verify: SSL验证
        timeout: 超时时间
        allow_redirects: 是否允许重定向
        sleep: 重试间隔
        max_retries: 最大重试次数
    
    Returns:
        Response对象或None
    """
    if headers is None:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 '
                         '(KHTML, like Gecko) Chrome/101.0.4951.7 Safari/537.36'
        }
    
    retry_count = 0
    while True:
        retry_count += 1
        if max_retries != 0 and retry_count > max_retries:
            return None
            
        try:
            response = requests.get(
                url=url,
                params=params,
                headers=headers,
                cookies=cookies,
                proxies=proxies,
                verify=verify,
                timeout=timeout,
                allow_redirects=allow_redirects
            )
            response.encoding = response.apparent_encoding
            return response
            
        except (requests.exceptions.ProxyError,
                requests.exceptions.ConnectTimeout,
                requests.exceptions.ReadTimeout,
                requests.exceptions.ConnectionError,
                requests.exceptions.ChunkedEncodingError,
                requests.exceptions.TooManyRedirects,
                ConnectionResetError,
                socket.timeout,
                http.client.IncompleteRead,
                ssl.SSLWantReadError) as e:
            print(f'请求错误 - {type(e).__name__}: {e}')
            time.sleep(sleep)
            continue

def parse_html(html_text: str) -> PyQuery:
    """
    解析HTML文本为PyQuery对象
    
    Args:
        html_text: HTML文本内容
        
    Returns:
        PyQuery对象
    """
    if not html_text or not html_text.strip():
        return PyQuery('<div></div>')
        
    try:
        return PyQuery(html_text)
    except Exception as e:
        print(f'PyQuery解析错误: {e}')
        return PyQuery('<div></div>')

def should_skip_content(title: str) -> bool:
    """
    判断是否应该跳过这个内容（通知类、直播预告等）
    
    Args:
        title: 文章标题
        
    Returns:
        bool: 是否跳过
    """
    if not title:
        return True
    
    # 过滤以方框日期开头的标题，如"[7.18]"
    if re.match(r'^\[.*?\]', title):
        return True
    
    # 过滤包含特定关键词的通知类标题
    if re.match(r'^【.*?】', title):
        return True
    
    return False

def clean_filename(title: str) -> str:
    """
    清理标题中的非法字符，使其适合作为文件名
    
    Args:
        title: 原始标题
        
    Returns:
        清理后的标题
    """
    if not title:
        return None
    
    # 移除非法文件名字符
    illegal_chars = r'[<>:"/\\|?*\x00-\x1f\x7f]'
    title = re.sub(illegal_chars, '', title)
    
    # 处理特殊字符
    title = title.replace('\xa0', ' ')
    title = re.sub(r'[\u200b-\u200f\u202a-\u202e]', '', title)
    
    # 清理多余的反斜杠和空格
    title = re.sub(r'\\\\+', r'\\', title)
    title = title.strip('. ')
    title = re.sub(r'\s+', ' ', title)
    
    return title.strip()

def clean_article_content(content: str) -> str:
    """
    清理文章内容
    
    Args:
        content: 原始内容
        
    Returns:
        清理后的内容
    """
    if not content:
        return ""
    
    # 移除网站版权信息
    patterns_to_remove = [
        r'家庭医生在线（www\.familydoctor\.com\.cn）原创内容，未经授权不得转载，违者必究，内容合作请联系：020-37617238'
    ]
    
    for pattern in patterns_to_remove:
        content = re.sub(pattern, '', content)
    
    return content.strip()

def save_article_content(content: str, file_path: str) -> bool:
    """
    保存文章内容到文件
    
    Args:
        content: 文章内容
        file_path: 文件路径
        
    Returns:
        bool: 是否保存成功
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(content)
        return True
    except Exception as e:
        print(f"保存文件失败 {file_path}: {e}")
        return False

def crawl_article_detail(
    page_url: str, 
    headers: Dict[str, str] = HEADERS
) -> Optional[Dict[str, Any]]:
    """
    爬取文章详情页
    
    Args:
        page_url: 文章详情页URL
        headers: 请求头
        
    Returns:
        包含标题和内容的字典，或None
    """
    response = get_http_response(url=page_url, headers=headers, max_retries=3)
    if not response:
        return None
    
    response.encoding = 'utf-8'
    html = parse_html(response.text)
    title = html('.article-titile>h1').text()
    cleaned_title = clean_filename(title)
    
    if not cleaned_title:
        return None
    
    # 再次检查标题是否需要跳过
    if should_skip_content(cleaned_title):
        print(f'跳过通知类内容: {cleaned_title}')
        return None
    
    content = html('.viewContent').text()
    cleaned_content = clean_article_content(content)
    
    return {
        'title': cleaned_title,
        'content': cleaned_content,
        'url': page_url
    }

def crawl_familydoctor_data(save_dir):
    """
    主爬虫函数 - 爬取医疗数据
    """        
    # 创建保存目录
    save_dir = os.path.join(save_dir, "familydoctor_data")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 初始化变量
    current_content = ""  # 当前累积的内容
    file_id = 1  # 文件ID计数器
    total_articles = 0  # 总文章计数
    
    for page_num in range(1, MAX_PAGES + 1):
        print('=' * 60)
        print(f'正在采集第 {page_num} 页')
        
        # 构建搜索API参数
        api_url = 'https://so.familydoctor.com.cn/search'
        params = {
            'q': '脑卒中',
            't': 'info',
            'page': page_num,
        }
        
        # 获取列表页
        response = get_http_response(url=api_url, params=params, headers=HEADERS)
        if not response:
            print(f"第 {page_num} 页请求失败")
            continue
        
        # 解析列表
        html = parse_html(response.text)
        articles = html('.result-row>.cont')
        
        if len(articles) == 0:
            print("没有找到更多文章，爬取结束")
            break
        
        articles_count = 0
        for article in articles.items():
            title = article('h3>a').text()
            page_url = article('h3>a').attr('href')
            
            if not title or not page_url:
                continue
                
            # 检查是否需要跳过
            if should_skip_content(title):
                print(f'跳过通知类内容: {title}')
                continue
            
            # 爬取文章详情
            article_data = crawl_article_detail(page_url)
            if not article_data:
                continue
            
            # 将文章内容添加到当前累积内容中
            article_text = f"{article_data['content']}\n\n"
            current_content += article_text
            articles_count += 1
            total_articles += 1
            
            print(f'已采集: {article_data["title"]}')
            
            # 检查当前内容是否大于5000字符
            if len(current_content) >= 5000:
                # 保存当前内容到文件
                file_path = os.path.join(save_dir, f'familydoctor_{file_id}.txt')
                if save_article_content(current_content, file_path):
                    print(f'已保存第 {file_id} 个文件，包含 {articles_count} 篇文章')
                    current_content = ""  # 重置当前内容
                    file_id += 1  # 文件ID递增
                    articles_count = 0  # 重置当前页文章计数
        
        print(f'第 {page_num} 页完成，采集了 {articles_count} 篇文章')
        
        # 添加延迟，避免请求过于频繁
        time.sleep(1)
    
    # 处理剩余内容（如果有）
    if current_content:
        file_path = os.path.join(save_dir, f'familydoctor_{file_id}.txt')
        if save_article_content(current_content, file_path):
            print(f'已保存最后第 {file_id} 个文件，包含剩余文章')
    
    print(f'爬取完成！总共采集了 {total_articles} 篇文章，保存为 {file_id} 个文件')
