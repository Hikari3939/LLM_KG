import os
from dotenv import load_dotenv
from graphdatascience import GraphDataScience
from langchain_community.vectorstores import Neo4jVector
from langchain_huggingface import HuggingFaceEmbeddings

from my_packages import DatabaseAbout
from my_packages import LLMAbout
from my_packages.MyNeo4j import MyNeo4jGraph

# 加载环境变量
load_dotenv(".env")
NEO4J_URI = os.environ["NEO4J_URI"]
NEO4J_USERNAME = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]

DEEPSEEK_API_KEY = os.environ["DEEPSEEK_API_KEY"]

# 指定模型名称
INSTRUCT_MODEL = 'deepseek-chat'

if __name__ == '__main__':
    graph = MyNeo4jGraph(refresh_schema=False)
    print("数据库成功连接")
    print('')

    # 加载BAAI/BGE-M3(3.7G显存)
    embeddings = HuggingFaceEmbeddings(
                model_name="BAAI/bge-m3", 
                cache_folder="./model"
            )
    print("Embedding模型成功加载")
    print('')

    # 用['id', 'description']来计算实体结点的Embedding。
    vector = Neo4jVector.from_existing_graph(
        embeddings,
        node_label='__Entity__',
        text_node_properties=['id', 'description'],
        embedding_node_property='embedding'
    )
    print("Embedding嵌入完成")
    print('')
    
    # GDS连接Neo4j
    gds = GraphDataScience(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"])
    )
    
    potential_duplicate_candidates = DatabaseAbout.knn_similarity(graph, gds)
        
    merged_entities = LLMAbout.decide_entity_merge(INSTRUCT_MODEL, potential_duplicate_candidates)
