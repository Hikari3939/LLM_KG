from typing import Optional, List, Dict
from urllib.parse import quote
import wikipediaapi
import requests
import logging
import zhconv  # 用于简繁转换
import neo4j
import time

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
            user_agent="Neo4jImageUpdater/1.0 (fatails_hikari@yeah.net)"
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

            # 200状态码表示页面确切存在
            if response.status_code == 200:
                data = response.json()
                # 检查是否有缩略图
                if 'thumbnail' in data and 'source' in data['thumbnail']:
                    return data['thumbnail']['source']
                else:
                    logger.warning(f"页面 '{search_term}' 存在但没有缩略图")
                    return None
            elif response.status_code == 404:
                # 页面不存在，不记录警告
                return None
            else:
                # 其他HTTP错误（如500），记录警告
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
            "SET n.image_url = $image_url "
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

        # 搜索维基百科图片（精确搜索）
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
