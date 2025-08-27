# script_wikipedia.py
# 这个脚本教你如何从Wikipedia获取图片（如果Bing API用不了，可以用这个）
#test&test

import wikipediaapi
from neo4j import GraphDatabase

# 设置（同样需要修改！）
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "Wyhzybdyxl0304"

node_list = ["Paris", "London", "New York", "Tokyo", "Beijing"]


def connect_neo4j(uri, user, password):
    print("正在尝试连接Neo4j数据库...")
    try:
        driver = GraphDatabase.driver(uri, auth=(user, password))
        driver.verify_connectivity()
        print("✅ 数据库连接成功！")
        return driver
    except Exception as e:
        print("❌ 连接失败：", e)
        return None


def get_wikipedia_image(page_title):
    print(f"   正在从Wikipedia获取『{page_title}』的图片...")
    # 设置User-Agent，这是礼貌的做法
    wiki_wiki = wikipediaapi.Wikipedia(
        user_agent='MyKGBot/1.0 (myemail@example.com)',
        language='en'
    )
    page = wiki_wiki.page(page_title)

    if page.exists() and page.thumbnail:
        # 如果页面存在且有缩略图，返回图片URL
        image_url = page.thumbnail['source']
        print(f"   ✅ 找到图片：{image_url}")
        return image_url
    else:
        print(f"   ❌ Wikipedia上未找到该页面或图片。")
        return None


def update_node_with_image(tx, node_name, image_url):
    query = """
    MERGE (n:Entity {name: $name})
    SET n.wikipedia_image_url = $image_url
    RETURN n
    """
    result = tx.run(query, name=node_name, image_url=image_url)
    return result.single()


def main():
    print("📚 使用Wikipedia方案启动...")
    driver = connect_neo4j(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    if not driver:
        return

    for node_name in node_list:
        print(f"\n处理节点: {node_name}")
        image_url = get_wikipedia_image(node_name)

        if image_url:
            with driver.session() as session:
                try:
                    result = session.execute_write(update_node_with_image, node_name, image_url)
                    print(f"   ✅ 成功更新数据库节点！")
                except Exception as e:
                    print(f"   ❌ 更新数据库时出错：{e}")
        else:
            print(f"   ⚠️  未找到图片，跳过。")

    driver.close()
    print("\n🎉 所有任务完成！")


if __name__ == "__main__":
    main()