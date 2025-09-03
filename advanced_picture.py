import neo4j
import wikipediaapi
import requests
import time
import logging
import zhconv  # 用于简繁转换
from typing import Optional, List, Dict
from urllib.parse import quote

# 设置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Neo4jImageUpdater:
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
            user_agent="Neo4jImageUpdater/1.0 (your_email@example.com)"
        )

    def close(self):
        """关闭数据库连接"""
        self.driver.close()

    def get_node_ids_by_label(self, label: str) -> List[str]:
        """
        获取指定标签的所有节点的ID

        Args:
            label: 节点标签

        Returns:
            节点ID列表
        """
        query = f"MATCH (n:{label}) RETURN n.id as node_id"

        try:
            with self.driver.session() as session:
                result = session.run(query)
                node_ids = [record["node_id"] for record in result if record["node_id"]]
                return node_ids
        except Exception as e:
            logger.error(f"查询节点时出错: {e}")
            return []

    def get_all_labels(self) -> List[str]:
        """
        获取数据库中所有的节点标签

        Returns:
            标签列表
        """
        query = "CALL db.labels() YIELD label RETURN label"

        try:
            with self.driver.session() as session:
                result = session.run(query)
                labels = [record["label"] for record in result]
                return labels
        except Exception as e:
            logger.error(f"查询标签时出错: {e}")
            return []

    def find_node_by_id(self, node_id: str) -> Optional[dict]:
        """
        根据节点ID查找节点

        Args:
            node_id: 节点ID

        Returns:
            节点属性字典或None（如果未找到）
        """
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
        只有确切存在的页面才会返回图片URL

        Args:
            search_term: 搜索术语

        Returns:
            图片URL或None（如果未找到）
        """
        try:
            # 使用维基百科API直接获取页面信息
            url = f"https://zh.wikipedia.org/api/rest_v1/page/summary/{quote(search_term)}"
            headers = {
                'User-Agent': 'Neo4jImageUpdater/1.0 (your_email@example.com)'
            }

            response = requests.get(url, headers=headers)

            # 只有200状态码表示页面确切存在
            if response.status_code == 200:
                data = response.json()
                # 检查是否有缩略图
                if 'thumbnail' in data and 'source' in data['thumbnail']:
                    return data['thumbnail']['source']
                else:
                    logger.warning(f"页面 '{search_term}' 存在但没有缩略图")
                    return None
            elif response.status_code == 404:
                # 页面不存在，不记录警告，这是正常情况
                return None
            else:
                # 其他HTTP错误（如500）可能是临时问题，记录警告
                logger.warning(f"搜索 '{search_term}' 时收到状态码 {response.status_code}")
                return None

        except Exception as e:
            logger.error(f"搜索维基百科时出错: {e}")
            return None

    def search_wiki_image_with_retry(self, search_term: str, max_retries: int = 3) -> Optional[str]:
        """
        带重试机制的精确图片搜索
        只返回确切存在的页面的图片

        Args:
            search_term: 搜索术语
            max_retries: 最大重试次数

        Returns:
            图片URL或None（如果未找到）
        """
        # 先尝试简体搜索
        image_url = self.search_wiki_image(search_term)
        if image_url:
            return image_url

        # 如果简体搜索返回404，尝试繁体搜索
        traditional_term = zhconv.convert(search_term, 'zh-tw')
        if traditional_term != search_term:
            logger.info(f"尝试繁体搜索: {traditional_term}")
            image_url = self.search_wiki_image(traditional_term)
            if image_url:
                return image_url

        # 如果直接搜索都返回404，说明页面确实不存在
        # 对于服务器错误（5xx）进行重试
        for attempt in range(max_retries):
            try:
                # 检查是否是服务器错误，如果是则重试
                url = f"https://zh.wikipedia.org/api/rest_v1/page/summary/{quote(search_term)}"
                headers = {'User-Agent': 'Neo4jImageUpdater/1.0'}

                response = requests.get(url, headers=headers)

                if response.status_code == 200:
                    data = response.json()
                    if 'thumbnail' in data and 'source' in data['thumbnail']:
                        return data['thumbnail']['source']
                    else:
                        return None
                elif response.status_code >= 500:
                    # 服务器错误，等待后重试
                    wait_time = 2 ** attempt
                    logger.warning(f"服务器错误，等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                    continue
                else:
                    # 404或其他客户端错误，不重试
                    return None

            except Exception as e:
                logger.error(f"第 {attempt + 1} 次尝试失败: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)

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
        query = (
            "MATCH (n {id: $node_id}) "
            "SET n.image_url = $image_url, n.image_updated = timestamp() "
            "RETURN n"
        )

        try:
            with self.driver.session() as session:
                result = session.run(query, node_id=node_id, image_url=image_url)
                # 获取结果摘要
                summary = result.consume()
                # 检查是否有属性被设置
                if summary.counters.properties_set > 0:
                    logger.info(f"成功更新节点 '{node_id}' 的属性")
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
        只有在维基百科中确切存在的页面才会导入图片

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

        # 搜索维基百科图片（精确搜索，只返回确切存在的页面图片）
        image_url = self.search_wiki_image_with_retry(node_id)
        if not image_url:
            logger.info(f"节点 '{node_id}' 在维基百科中不存在对应页面，跳过")
            return False

        logger.info(f"找到图片URL: {image_url}")

        # 更新节点
        success = self.update_node_with_image(node_id, image_url)
        if success:
            logger.info(f"成功更新节点 '{node_id}' 的图片URL")
        else:
            logger.error(f"更新节点 '{node_id}' 失败")

        return success

    def process_nodes_by_label(self, label: str, delay: float = 1.0) -> Dict[str, int]:
        """
        处理指定标签的所有节点
        只有在维基百科中确切存在的页面才会导入图片

        Args:
            label: 节点标签
            delay: 每个请求之间的延迟时间（秒），避免请求过于频繁

        Returns:
            处理结果统计
        """
        results = {
            'total': 0,
            'success': 0,
            'failed': 0,
            'not_found': 0,  # 新增：记录在维基百科中不存在的节点数
            'failed_nodes': []
        }

        # 获取指定标签的所有节点ID
        node_ids = self.get_node_ids_by_label(label)
        if not node_ids:
            logger.warning(f"未找到标签为 '{label}' 的节点")
            return results

        results['total'] = len(node_ids)
        logger.info(f"找到 {results['total']} 个标签为 '{label}' 的节点")

        for i, node_id in enumerate(node_ids, 1):
            logger.info(f"处理进度: {i}/{results['total']} - {node_id}")

            try:
                success = self.process_node(node_id)
                if success:
                    results['success'] += 1
                else:
                    # 区分是节点不存在还是其他错误
                    if not self.find_node_by_id(node_id):
                        results['failed'] += 1
                        results['failed_nodes'].append(node_id)
                    else:
                        results['not_found'] += 1
            except Exception as e:
                logger.error(f"处理节点 '{node_id}' 时发生异常: {e}")
                results['failed'] += 1
                results['failed_nodes'].append(node_id)

            # 添加延迟，避免请求过于频繁
            if i < results['total'] and delay > 0:
                time.sleep(delay)

        logger.info(f"批量处理完成: "
                    f"总共 {results['total']} 个节点, "
                    f"成功 {results['success']}, "
                    f"维基百科中不存在 {results['not_found']}, "
                    f"失败 {results['failed']}")

        if results['failed_nodes']:
            logger.info(f"失败的节点: {', '.join(results['failed_nodes'])}")

        return results

    def list_all_labels(self) -> List[str]:
        """
        列出数据库中所有标签

        Returns:
            标签列表
        """
        labels = self.get_all_labels()
        if labels:
            logger.info("数据库中的标签:")
            for label in labels:
                logger.info(f" - {label}")
        else:
            logger.warning("未找到任何标签")

        return labels


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