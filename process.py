import os
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph
from langchain_deepseek import ChatDeepSeek
from graphdatascience import GraphDataScience
from langchain_community.vectorstores import Neo4jVector
from langchain_huggingface import HuggingFaceEmbeddings

import DatabaseAbout

# 加载环境变量
load_dotenv(".env")
NEO4J_URI = os.environ["NEO4J_URI"]
NEO4J_USERNAME = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]

DEEPSEEK_API_KEY = os.environ["DEEPSEEK_API_KEY"]

# 指定模型名称
INSTRUCT_MODEL = 'deepseek-chat'

if __name__ == '__main__':
    graph = Neo4jGraph(refresh_schema=False)

    # # 加载BAAI/BGE-M3(3.7G显存)
    # embeddings = HuggingFaceEmbeddings(
    #             model_name="BAAI/bge-m3", 
    #             cache_folder="E:/Study/Projects/SRTP_About/code/LLM_KG/model"
    #         )

    # # 用['id', 'description']来计算实体结点的Embedding。
    # vector = Neo4jVector.from_existing_graph(
    #     embeddings,
    #     node_label='__Entity__',
    #     text_node_properties=['id', 'description'],
    #     embedding_node_property='embedding'
    # )
    
    # GDS连接Neo4j
    gds = GraphDataScience(
        os.environ["NEO4J_URI"],
        auth=(os.environ["NEO4J_USERNAME"], os.environ["NEO4J_PASSWORD"])
    )
    
    potential_duplicate_candidates = DatabaseAbout.knn_similarity(graph, gds)    
    
    # # 连接模型
    # llm = ChatDeepSeek(
    #     model=INSTRUCT_MODEL,
    #     temperature=1.0
    # )
