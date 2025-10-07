import os
from dotenv import load_dotenv
from langchain_neo4j import Neo4jVector
from graphdatascience import GraphDataScience
from langchain_huggingface import HuggingFaceEmbeddings

from my_packages import LLMAbout
from my_packages import GraphAbout
from my_packages.MyNeo4j import MyNeo4jGraph

# 加载环境变量
load_dotenv(".env")
NEO4J_URI = os.environ["NEO4J_URI"]
NEO4J_USERNAME = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]

if __name__ == '__main__':
    graph = MyNeo4jGraph()
    print("数据库成功连接")
    print('')
    
    # 初步重写描述
    LLMAbout.rewrite_entity_descriptions(graph, 1000)
    LLMAbout.rewrite_relationship_descriptions(graph)
    print("描述初步重写完成")

    # 加载BAAI/BGE-M3(3.7G显存)
    embeddings = HuggingFaceEmbeddings(
                model_name="BAAI/bge-m3",
                model_kwargs = {"device": "cuda"},
                cache_folder="./model"
            )
    print("Embedding模型成功加载")
    print('')

    # 使用['id', 'description']计算实体结点的Embedding。
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
    
    # K近邻算法初步筛选相似实体
    potential_duplicate_candidates = GraphAbout.knn_similarity(graph, gds)
        
    # LLM进一步筛选
    merged_entities = LLMAbout.decide_entity_merge(potential_duplicate_candidates)
    GraphAbout.merge_similar_entities(graph, embeddings, merged_entities)
    print("相似实体成功合并")
    print('')

    # 清理孤立实体
    largest_component_id, wcc_result = GraphAbout.find_largest_connected_component(gds)
    GraphAbout.clean_isolated_entities(graph, largest_component_id, wcc_result)
    print("孤立实体清理完成")

    # 重写描述
    LLMAbout.rewrite_entity_descriptions(graph)
    LLMAbout.rewrite_relationship_descriptions(graph)
    print("描述重写完成")

    # 构建社区
    GraphAbout.clean_communities(graph)
    GraphAbout.build_communities(graph, gds)
    print("社区构建完成")
    print('')

    # 生成摘要
    LLMAbout.community_abstract(graph)
    print("社区摘要完成")
    print('')
