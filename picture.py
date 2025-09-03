import logging

from my_packages.GetPicture import Neo4jImageUpdater

# 设置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 使用示例
if __name__ == "__main__":
    # 数据库配置 - 请根据您的实际设置修改
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "Wyhzybdyxl0304"

    # 初始化更新器
    updater = Neo4jImageUpdater(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

    try:
        # 列出所有标签
        print("正在获取所有标签...")
        labels = updater.list_all_labels()

        if labels:
            print("\n可用的标签:")
            for i, label in enumerate(labels, 1):
                print(f"{i}. {label}")

            # 让用户选择标签
            choice = input("\n请选择要处理的标签编号 (输入 'all' 处理所有标签): ").strip()

            if choice.lower() == 'all':
                # 处理所有标签
                for label in labels:
                    print(f"\n正在处理标签: {label}")
                    results = updater.process_nodes_by_label(label, delay=1.0)
                    print(f"处理完成: 总共 {results['total']} 个节点, "
                          f"成功 {results['success']}, "
                          f"维基百科中不存在 {results['not_found']}, "
                          f"失败 {results['failed']}")
            else:
                try:
                    index = int(choice) - 1
                    if 0 <= index < len(labels):
                        label = labels[index]
                        print(f"\n正在处理标签: {label}")
                        results = updater.process_nodes_by_label(label, delay=1.0)
                        print(f"处理完成: 总共 {results['total']} 个节点, "
                              f"成功 {results['success']}, "
                              f"维基百科中不存在 {results['not_found']}, "
                              f"失败 {results['failed']}")
                    else:
                        print("无效的选择")
                except ValueError:
                    print("请输入有效的数字")
        else:
            print("数据库中未找到任何标签")

    finally:
        # 关闭连接
        updater.close()
