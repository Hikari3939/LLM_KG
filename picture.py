import os
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
    print(f"\n正在处理标签: {label}")
    results = updater.process_nodes_by_label(label, delay=1.0)
    print(f"处理完成: 总共 {results['total']} 个节点, "
            f"成功 {results['success']}, "
            f"维基百科中不存在 {results['not_found']}, "
            f"失败 {results['failed']}")
    
    # 关闭连接
    updater.close()
