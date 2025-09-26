from typing import List, Dict, Optional
from urllib.parse import quote
import requests
import logging
import zhconv
import neo4j
import time

REQUEST_DELAY = 1    # 请求延迟（秒）

# 设置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger()

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
        self.session = self.driver.session()
        logger.info("成功连接数据库")

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
            result = self.session.run(query)
            return [record["node_id"] for record in result if record["node_id"]]
        except Exception as e:
            logger.error(f"查询节点时出错: {e}")
            return []

    def search_wiki_image(self, search_term: str) -> Optional[str]:
        """
        在维基百科中搜索术语并获取最相关的图片URL

        Args:
            search_term: 搜索术语

        Returns:
            图片URL或None（如果未找到）
        """
        # 尝试简体搜索
        image_url = self._fetch_image_url(search_term)
        if image_url:
            return image_url

        # 尝试繁体搜索
        traditional_term = zhconv.convert(search_term, 'zh-tw')
        if traditional_term != search_term:
            logger.info(f"尝试繁体搜索: {traditional_term}")
            return self._fetch_image_url(traditional_term)
            
        return None

    def _fetch_image_url(self, term: str, max_retries: int = 3) -> Optional[str]:
        """
        内部方法：获取维基百科图片URL
        
        Args:
            term: 搜索术语
            max_retries: 最大重试次数
            
        Returns:
            图片URL或None
        """
        url = f"https://zh.wikipedia.org/api/rest_v1/page/summary/{quote(term)}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/117.0.0.0 Safari/537.36"
        }
        
        for attempt in range(max_retries):
            try:
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
        query = "MATCH (n {id: $node_id}) SET n.image_url = $image_url RETURN n"
        
        try:
            result = self.session.run(query, node_id=node_id, image_url=image_url)
            return result.consume().counters.properties_set > 0
        except Exception as e:
            logger.error(f"更新节点时出错: {e}")
            return False

    def process_single_node(self, node_id: str) -> tuple:
        """
        处理单个节点：搜索图片并更新数据库

        Args:
            node_id: 要处理的节点ID

        Returns:
            (节点ID, 成功与否, 图片URL或错误信息)
        """
        try:
            image_url = self.search_wiki_image(node_id)
            if not image_url:
                return (node_id, False, "未找到图片")
                
            if self.update_node_with_image(node_id, image_url):
                return (node_id, True, image_url)
            else:
                return (node_id, False, "更新数据库失败")
                
        except Exception as e:
            logger.error(f"处理节点 '{node_id}' 时发生异常: {e}")
            return (node_id, False, str(e))

    def process_nodes_by_label(self, label: str) -> Dict[str, int]:
        """
        处理指定标签的所有节点

        Args:
            label: 节点标签

        Returns:
            处理结果统计
        """
        # 获取所有节点ID
        node_ids = self.get_node_ids_by_label(label)
        if not node_ids:
            logger.warning(f"未找到标签为 '{label}' 的节点")
            return {'total': 0, 'success': 0, 'failed': 0}
        
        logger.info(f"找到 {len(node_ids)} 个标签为 '{label}' 的节点")
        
        # 处理节点
        results = {'total': len(node_ids), 'success': 0, 'failed': 0}
        for i, node_id in enumerate(node_ids, start=1):
            node_id, success, message = self.process_single_node(node_id)
            
            if success:
                results['success'] += 1
                logger.info(f"成功处理节点 {i}/{len(node_ids)}: {node_id} -> {message}")
            else:
                results['failed'] += 1
                logger.warning(f"处理节点失败 {i}/{len(node_ids)}: {node_id} - {message}")
                
            time.sleep(REQUEST_DELAY)
        
        logger.info(f"处理完成: 总共 {results['total']} 个节点, 成功 {results['success']}, 失败 {results['failed']}")
        return results
