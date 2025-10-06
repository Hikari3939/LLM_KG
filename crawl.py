import re
import os
from my_packages.GetText import get, pq, re_title
from my_packages.GetText import should_skip_content

def crawl_medical_data():
    # 配置参数
    keyword = '脑卒中'
    save_dir = os.path.join(os.getcwd(), 'data', 'familydoctor')
    
    headers = {
        'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
        'accept-encoding': 'gzip, deflate, br, zstd',
        'accept-language': 'zh-CN,zh;q=0.9',
        'cookie': 'acw_tc=6f0caf1817596269988003281eea3f56f105fc4fdefacdda521f0f1d05; cdn_sec_tc=6f0caf1817596269988003281eea3f56f105fc4fdefacdda521f0f1d05; Hm_lvt_f46dd4cc550b93aefde9b00265bb533d=1759627024; HMACCOUNT=280E67F5FB89C93F; ASP.NET_SessionId=bsnsnjmk2fpszlgxwfakiax3; Hm_lpvt_f46dd4cc550b93aefde9b00265bb533d=1759627167',
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
        'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/132.0.0.0 Safari/537.36'
    }

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for page_num in range(1, 31):
        print('*' * 60)
        print('采集第 {} 页'.format(page_num))
        
        api = 'https://so.familydoctor.com.cn/search'
        params = {
            'q': keyword,
            't': 'info',
            'page': page_num,
        }

        response = get(url=api, params=params, headers=headers)
        if not response:
            continue

        html = pq(response.text)
        lis = html('.result-row>.cont')
        
        if len(lis) == 0:
            break
            
        for li in lis.items():
            title = li('h3>a').text()
            page_url = li('h3>a').attr('href')
            
            # 先检查标题是否需要跳过
            if should_skip_content(title, ""):
                print(f'跳过通知类内容: {title}')
                continue
                
            response = get(url=page_url, headers=headers, try_num=3)
            if not response:
                continue
                
            response.encoding = 'utf-8'
            html = pq(response.text)
            art_title = html('.article-titile>h1').text()
            art_title = re_title(title=art_title)
            
            if len(art_title) == 0:
                continue

            # 再次检查处理后的标题
            if should_skip_content(art_title, ""):
                print(f'跳过通知类内容: {art_title}')
                continue

            save_path = os.path.join(save_dir, f'{art_title}.txt')
            save_path = re.sub(r'\\\\+', r'\\', save_path)

            date = html('.info .left').text()
            content = html('.viewContent').text()
            
            # 清理内容中的联系信息
            content = re.sub(r'家庭医生在线（www\.familydoctor\.com\.cn）原创内容，未经授权不得转载，违者必究，内容合作请联系：020-37617238', '', content)
            content = content.strip()

            if os.path.exists(save_path):
                print(save_path, '已存在!')
            else:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                with open(save_path, 'w', encoding='utf-8') as fw:
                    fw.write(content)
                print(f'已保存: {art_title}')

if __name__ == '__main__':
    crawl_medical_data()