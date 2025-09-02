import neo4j
import wikipediaapi
import requests
import time
import logging
from typing import Optional
from urllib.parse import quote

# 设置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Neo4jWikiImageUpdater:
    def __init__(self, uri: str, user: str, password: str):
        """
        初始化Neo4j连接

        Args:
            uri: Neo4j数据库URI
            user: 用户名
            password: 密码
        """
        self.driver = neo4j.GraphDatabase.driver(uri, auth=(user, password))
        self.wiki_wiki = wikipediaapi.Wikipedia(
            language='zh',  # 使用中文维基百科
            extract_format=wikipediaapi.ExtractFormat.WIKI,
            user_agent="Neo4jWikiImageUpdater/1.0 (your_email@example.com)"
        )

    def close(self):
        """关闭数据库连接"""
        self.driver.close()

    def find_node_by_id(self, node_id: str) -> Optional[dict]:
        """
        根据节点ID查找节点

        Args:
            node_id: 节点ID

        Returns:
            节点属性字典或None（如果未找到）
        """
        # 修改查询语句，不使用elementId
        query = (
            "MATCH (n {id: $node_id}) "
            "RETURN n.id as id, n.description as description"
        )

        with self.driver.session() as session:
            result = session.run(query, node_id=node_id)
            record = result.single()

            if record:
                return {
                    "id": record["id"],
                    "description": record["description"]
                }
            else:
                return None

    def search_wiki_image(self, search_term: str) -> Optional[str]:
        """
        在维基百科中搜索术语并获取最相关的图片URL

        Args:
            search_term: 搜索术语

        Returns:
            图片URL或None（如果未找到）
        """
        try:
            # 使用维基百科API直接获取页面信息
            url = f"https://zh.wikipedia.org/api/rest_v1/page/summary/{quote(search_term)}"
            headers = {
                'User-Agent': 'Neo4jWikiImageUpdater/1.0 (your_email@example.com)'
            }

            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                data = response.json()
                # 检查是否有缩略图
                if 'thumbnail' in data and 'source' in data['thumbnail']:
                    return data['thumbnail']['source']
                else:
                    logger.warning(f"页面 '{search_term}' 没有缩略图")
                    return None
            else:
                logger.warning(f"无法获取页面 '{search_term}' 的信息，状态码: {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"搜索维基百科时出错: {e}")
            return None

    def update_node_with_image(self, node_id: str, image_url: str) -> bool:
        """
        使用图片URL更新节点

        Args:
            node_id: 节点ID
            image_url: 图片URL

        Returns:
            更新是否成功
        """
        # 修改查询语句，使用节点ID
        query = (
            "MATCH (n {id: $node_id}) "
            "SET n.image_url = $image_url, n.image_updated = timestamp() "
            "RETURN n"  # 这个RETURN n不是必须的，如果你不需要处理结果，可以去掉
        )

        try:
            with self.driver.session() as session:
                result = session.run(query, node_id=node_id, image_url=image_url)
                # 获取结果摘要
                summary = result.consume()
                # 检查是否有属性被设置 (使用 properties_set)
                if summary.counters.properties_set > 0:
                    logger.info(f"成功更新节点 '{node_id}' 的属性，设置了 {summary.counters.properties_set} 个属性。")
                    return True
                else:
                    logger.warning(f"未更新节点 '{node_id}' 的属性。可能未找到匹配的节点。")
                    return False
        except Exception as e:
            logger.error(f"更新节点时出错: {e}")
            return False

    def process_node(self, node_id: str) -> bool:
        """
        处理单个节点：查找节点，搜索图片，更新数据库

        Args:
            node_id: 要处理的节点ID

        Returns:
            处理是否成功
        """
        logger.info(f"处理节点: {node_id}")

        # 查找节点
        node = self.find_node_by_id(node_id)
        if not node:
            logger.error(f"未找到ID为 '{node_id}' 的节点")
            return False

        logger.info(f"找到节点: {node}")

        # 搜索维基百科图片
        image_url = self.search_wiki_image(node_id)
        if not image_url:
            logger.warning(f"未找到 '{node_id}' 的图片")
            return False

        logger.info(f"找到图片URL: {image_url}")

        # 更新节点
        success = self.update_node_with_image(node_id, image_url)
        if success:
            logger.info(f"成功更新节点 '{node_id}' 的图片URL")
        else:
            logger.error(f"更新节点 '{node_id}' 失败")

        return success


# 使用示例
if __name__ == "__main__":
    # 数据库配置 - 请根据您的实际设置修改
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = "Wyhzybdyxl0304"

    # 初始化更新器
    updater = Neo4jWikiImageUpdater(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)

    try:
        # 处理节点 - 以"高血压"为例
        node_id = "高血压"
        success = updater.process_node(node_id)

        if success:
            print(f"成功处理节点 '{node_id}'")
        else:
            print(f"处理节点 '{node_id}' 失败")

    finally:
        # 关闭连接
        updater.close()