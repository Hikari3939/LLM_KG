import time
import requests
from neo4j import GraphDatabase

# 配置信息
NEO4J_URI = "bolt://localhost:7687"  # Neo4j数据库地址
NEO4J_USER = "neo4j"                 # 数据库用户名
NEO4J_PASSWORD = "your_password"     # 数据库密码

# 初始化Neo4j驱动
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def get_wikipedia_image_url(page_title):
    """
    根据Wikipedia页面标题获取图片URL
    """
    # Wikipedia API端点
    api_url = "https://en.wikipedia.org/w/api.php"
    
    # 请求参数
    params = {
        "action": "query",
        "titles": page_title,
        "prop": "images",
        "format": "json"
    }
    
    try:
        # 发送请求获取页面使用的图片列表
        response = requests.get(api_url, params=params)
        data = response.json()
        
        # 提取页面ID
        pages = data.get("query", {}).get("pages", {})
        page_id = next(iter(pages)) if pages else None
        
        if not page_id or page_id == "-1":
            print(f"未找到页面: {page_title}")
            return None
        
        # 获取图片列表
        images = pages[page_id].get("images", [])
        if not images:
            print(f"页面 {page_title} 没有图片")
            return None
        
        # 选择第一张图片（可以根据需要修改选择逻辑）
        image_title = images[0]["title"]
        
        # 获取图片URL
        params2 = {
            "action": "query",
            "titles": image_title,
            "prop": "imageinfo",
            "iiprop": "url",
            "format": "json"
        }
        
        response2 = requests.get(api_url, params=params2)
        data2 = response2.json()
        
        # 提取图片URL
        pages2 = data2.get("query", {}).get("pages", {})
        image_page_id = next(iter(pages2)) if pages2 else None
        
        if image_page_id and image_page_id != "-1":
            image_info = pages2[image_page_id].get("imageinfo", [])
            if image_info:
                return image_info[0]["url"]
        
        return None
        
    except Exception as e:
        print(f"获取 {page_title} 图片时出错: {str(e)}")
        return None

def get_nodes_without_images(tx, label, batch_size=100):
    """
    从Neo4j获取没有图片URL的节点
    """
    query = f"""
    MATCH (n:{label}) 
    WHERE n.image_url IS NULL AND n.name IS NOT NULL
    RETURN n.name AS name, id(n) AS id
    LIMIT $batch_size
    """
    result = tx.run(query, batch_size=batch_size)
    return [{"id": record["id"], "name": record["name"]} for record in result]

def update_node_image(tx, node_id, image_url):
    """
    更新节点的图片URL
    """
    query = """
    MATCH (n) 
    WHERE id(n) = $node_id
    SET n.image_url = $image_url
    """
    tx.run(query, node_id=node_id, image_url=image_url)

def process_batch(label, batch_size=10):
    """
    处理一批节点
    """
    with driver.session() as session:
        # 获取一批没有图片的节点
        nodes = session.execute_read(get_nodes_without_images, label, batch_size)
        
        if not nodes:
            print("所有节点已处理完毕或没有符合条件的节点")
            return False
        
        print(f"开始处理 {len(nodes)} 个节点")
        
        # 处理每个节点
        for node in nodes:
            print(f"处理节点: {node['name']}")
            
            # 获取Wikipedia图片URL
            image_url = get_wikipedia_image_url(node['name'])
            
            if image_url:
                # 更新节点
                session.execute_write(update_node_image, node['id'], image_url)
                print(f"已为 {node['name']} 添加图片: {image_url}")
            else:
                print(f"未找到 {node['name']} 的图片")
            
            # 添加延迟，避免请求过于频繁
            time.sleep(1)
        
        return True

def main():
    """
    主函数
    """
    label = "Disease"  # 你的节点标签，根据实际情况修改
    
    print("开始处理节点...")
    
    # 持续处理直到所有节点处理完毕
    while process_batch(label, batch_size=10):
        print("处理完一批节点，继续下一批...")
    
    print("所有节点处理完毕!")
    
    # 关闭驱动
    driver.close()

if __name__ == "__main__":
    main()