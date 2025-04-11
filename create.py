import os
import time
import traceback
from dotenv import load_dotenv

from langchain_neo4j import Neo4jGraph
from langchain_deepseek import ChatDeepSeek
from langchain_experimental.graph_transformers import LLMGraphTransformer

import loader

# 加载环境变量
load_dotenv(".env")
NEO4J_URI = os.environ["NEO4J_URI"]
NEO4J_USERNAME = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]

DEEPSEEK_API_KEY = os.environ["DEEPSEEK_API_KEY"]

# 指定模型名称
INSTRUCT_MODEL = 'deepseek-chat'

# 指定测试数据的目录路径
directory_path = './data'


if __name__ == '__main__':
    # Langchain连接数据库
    graph = Neo4jGraph(
        url=NEO4J_URI, 
        username=NEO4J_USERNAME, 
        password=NEO4J_PASSWORD
    )
    print("数据库成功连接")

    # 删除数据库文件
    graph.query("MATCH (n) DETACH DELETE n")

    # 处理文件
    all_documents = loader.split_folder(directory_path, 2048, 256)
    for filename, document in all_documents:
        print(filename)
        for chunk in document:
            print(len(chunk.page_content))

    # 连接模型
    llm = ChatDeepSeek(
        model=INSTRUCT_MODEL,
        temperature=1.0
    )
    # 创建LLM到图文档的转换器
    llm_transformer = LLMGraphTransformer(llm=llm)

    # 设置合理批次大小（根据模型承受能力调整）
    batch_size = 10
    # 设置最大重试次数
    max_retries = 3
    # 设置重试之间的延迟时间（秒）
    retry_delay = 1
    # 提取各文档的分块内容并存入neo4j
    for filename, document in all_documents:
        for i in range(0, len(document), batch_size):
            batch = document[i:i+batch_size]
            retry_count = 0
            while retry_count < max_retries:
                try:
                    # 将当前批次转换为图文档
                    graph_documents = llm_transformer.convert_to_graph_documents(batch)
                    # 构建知识图谱
                    graph.add_graph_documents(graph_documents, include_source=True, baseEntityLabel=True)
                    
                    print(f'Batch {i} processed successfully.')
                    break  # 如果没有错误，跳出循环，处理下一个批次
                except Exception as e:
                    # 如果发生错误，打印错误信息，并增加重试计数
                    print(f"Error processing batch {i}, attempt {retry_count + 1}/{max_retries}: {e}")
                    traceback.print_exc()
                    retry_count += 1
                    if retry_count < max_retries:
                        # 在下一次尝试之前等待一段时间
                        time.sleep(retry_delay)
                    else:
                        print(f"Failed to process batch {i} after {max_retries} attempts. Moving on to the next batch.")

    # 关闭数据库
    graph.close()