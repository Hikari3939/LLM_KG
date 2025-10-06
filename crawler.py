from my_packages import WikiScrape
from my_packages import FamilydoctorScrape

# 指定数据保存路径
DIRECTORY_PATH = './data'

if __name__ == "__main__":
    # 爬取wiki数据
    results, _, _ = WikiScrape.wiki_scrape()
    WikiScrape.save_wiki_data(results, DIRECTORY_PATH)
    
    # 爬取医疗网站数据
    FamilydoctorScrape.crawl_familydoctor_data(DIRECTORY_PATH)
