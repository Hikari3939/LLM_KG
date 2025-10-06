import requests
requests.packages.urllib3.disable_warnings()
import re
import os
import time
import socket
import http.client
import ssl
from pyquery import PyQuery



def get(url,params=None,headers=None,cookies=None,proxies=None,verify=True,timeout=10,allow_redirects=True,sleep=1,try_num=0):
    if headers==None:
        headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/101.0.4951.7 Safari/537.36'
        }
    try_index=0
    while True:
        try_index+=1
        if try_num!=0 and try_index>=try_num:
            return None
        try:
            response=requests.get(url=url,params=params,headers=headers,cookies=cookies,proxies=proxies,verify=verify,timeout=timeout,allow_redirects=allow_redirects)
            response.encoding=response.apparent_encoding
            break
        except requests.exceptions.ProxyError:
            print('get——ProxyError')
            time.sleep(sleep)
            continue
        except requests.exceptions.ConnectTimeout:
            print('get——ConnectTimeout')
            time.sleep(sleep)
            continue
        except requests.exceptions.ReadTimeout:
            print('get——ReadTimeout')
            time.sleep(sleep)
            continue
        except requests.exceptions.ConnectionError:
            print('get——ConnectionError')
            time.sleep(sleep)
            continue
        except requests.exceptions.ChunkedEncodingError:
            print('get——ChunkedEncodingError')
            time.sleep(sleep)
            continue
        except requests.exceptions.TooManyRedirects:
            print('get——TooManyRedirects')
            time.sleep(sleep)
            continue
        except ConnectionResetError:
            print('get——ConnectionResetError')
            time.sleep(sleep)
            continue
        except socket.timeout:
            print('get——socket.timeout')
            time.sleep(sleep)
            continue
        except http.client.IncompleteRead:
            print('get——http.client.IncompleteRead')
            time.sleep(sleep)
            continue
        except ssl.SSLWantReadError:
            print('get——ssl.SSLWantReadError')
            time.sleep(sleep)
            continue

        
    return response

def pq(text):
    # 添加空值检查
    if not text or not text.strip():
        # 返回一个空的PyQuery对象，而不是抛出异常
        return PyQuery('<div></div>')
    try:
        html = PyQuery(text)
        return html
    except Exception as e:
        print(f'PyQuery解析错误: {e}')
        # 返回空的PyQuery对象作为容错
        return PyQuery('<div></div>')

def re_title(title):
    illegal_chars = r'[<>:"/\\|?*\x00-\x1f\x7f]'
    title = re.sub(illegal_chars, '', title)
    title = title.replace('\xa0', ' ')
    title = re.sub(r'[\u200b-\u200f\u202a-\u202e]', '', title)
    title = re.sub(r'\\\\+', r'\\', title)
    title = title.strip('. ')
    
    # 合并多个空格
    title = re.sub(r'\s+', ' ', title)
    return title.strip()

# 这里修改关键词
keyword='脑卒中'
# 这里是保存txt的文件夹
save_dir = os.path.join(os.getcwd(), 'data')

headers={'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7', 'accept-encoding': 'gzip, deflate, br, zstd', 'accept-language': 'zh-CN,zh;q=0.9', 'cookie': 'acw_tc=6f0caf1817596269988003281eea3f56f105fc4fdefacdda521f0f1d05; cdn_sec_tc=6f0caf1817596269988003281eea3f56f105fc4fdefacdda521f0f1d05; Hm_lvt_f46dd4cc550b93aefde9b00265bb533d=1759627024; HMACCOUNT=280E67F5FB89C93F; ASP.NET_SessionId=bsnsnjmk2fpszlgxwfakiax3; Hm_lpvt_f46dd4cc550b93aefde9b00265bb533d=1759627167', 'priority': 'u=0, i', 'referer': 'https://so.familydoctor.com.cn/search?q=%E8%84%91%E5%8D%92%E4%B8%AD&t=info', 'sec-ch-ua': '"Not A(Brand";v="8", "Chromium";v="132", "Google Chrome";v="132"', 'sec-ch-ua-mobile': '?0', 'sec-ch-ua-platform': '"Windows"', 'sec-fetch-dest': 'document', 'sec-fetch-mode': 'navigate', 'sec-fetch-site': 'same-origin', 'sec-fetch-user': '?1', 'upgrade-insecure-requests': '1', 'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36'}

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for page_num in range(1,31):
    print('*'*60)
    print('采集第 {} 项'.format(page_num))
    api='https://so.familydoctor.com.cn/search'
    params={
        'q': keyword,
        't': 'info',
        'page': page_num,
    }

    response=get(url=api,params=params,headers=headers)

    html=pq(response.text)
    lis=html('.result-row>.cont')
    if len(lis)==0:
        break
    for li in lis.items():
        title=li('h3>a').text()
        page_url=li('h3>a').attr('href')
        response=get(url=page_url,headers=headers,try_num=3)
        if response==None:
            continue
        response.encoding='utf-8'
        html=pq(response.text)
        art_title=html('.article-titile>h1').text()
        art_title=re_title(title=art_title)
        if len(art_title)==0:
            continue

        save_path=save_dir+r'\{}.txt'.format(art_title)
        save_path=re.sub(r'\\\\+', r'\\', save_path)

        date=html('.info .left').text()
        content=html('.viewContent').text()


        if os.path.exists(save_path)==True:
            print(save_path,'is exists!')
        else:
            save_dir=save_path.replace(save_path.split('\\')[-1],'')
            if os.path.exists(save_dir)==False:
                os.makedirs(save_dir)    
            with open(save_path,'w',encoding='utf-8') as fw:
                fw.write(content)
