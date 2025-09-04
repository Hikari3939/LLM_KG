import os
import time
from dotenv import load_dotenv

from my_packages.GetWikiPicture import Neo4jImageUpdater

# 加载环境变量
load_dotenv(".env")
NEO4J_URI = os.environ["NEO4J_URI"]
NEO4J_USERNAME = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]

if __name__ == "__main__":
    # 初始化更新器
    updater = Neo4jImageUpdater(NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD)

    # 为所有实体添加图片
    label = "__Entity__"
    t0 = time.time()
    results = updater.process_nodes_by_label(label)
    t2 = time.time()
    print("添加图片总耗时：",(t2-t0)/60,"分钟")
    print("\n")


    # 关闭连接
    updater.close()
