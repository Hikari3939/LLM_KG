import os
import json
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Neo4jVector
from langchain_core.output_parsers import StrOutputParser
from concurrent.futures import ThreadPoolExecutor, as_completed

# 环境变量配置
load_dotenv(".env")
os.environ["LANGCHAIN_TRACING_V2"] = "true"

NEO4J_URI = os.environ.get("NEO4J_URI")
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")

# LLM配置
llm = ChatDeepSeek(
    model='deepseek-chat'
)

# 嵌入模型配置
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs = {"device": "cpu"},
    cache_folder="./model"
)

# 创建Neo4j数据库连接
graph = Neo4jGraph()

def local_retriever(query: str) -> str:
    """局部检索器：检索知识图谱中的具体信息"""
    # 检索参数配置
    # 向量匹配部分
    topEntities = 10
    # Cypher查询部分
    topChunks = 3
    topOutsideRels = 10
    topInsideRels = 10
    
    # Cypher查询语句
    lc_retrieval_query = """
        WITH collect(node) as nodes
        // Entity - Text Unit Mapping
        WITH
        collect {
            UNWIND nodes as n
            MATCH (n)<-[:MENTIONS]-(c:__Chunk__)
            WITH distinct c, count(distinct n) as freq
            RETURN {id:c.id, text: c.text} AS chunkText
            ORDER BY freq DESC
            LIMIT $topChunks
        } AS text_mapping,
        // Outside Relationships 
        collect {
            UNWIND nodes as n
            MATCH (n)-[r]-(m:__Entity__) 
            WHERE NOT m IN nodes
            RETURN r.description AS descriptionText
            ORDER BY r.weight DESC 
            LIMIT $topOutsideRels
        } as outsideRels,
        // Inside Relationships 
        collect {
            UNWIND nodes as n
            MATCH (n)-[r]-(m:__Entity__) 
            WHERE m IN nodes
            RETURN r.description AS descriptionText
            ORDER BY r.weight DESC 
            LIMIT $topInsideRels
        } as insideRels,
        // Entities description
        collect {
            UNWIND nodes as n
            RETURN n.description AS descriptionText
        } as entities
        RETURN {Chunks: text_mapping, 
            Relationships: outsideRels + insideRels, 
            Entities: entities} AS text, 
            1.0 AS score, {} AS metadata
        """
    
    # 局部检索的Neo4j向量存储与索引
    lc_vector = Neo4jVector.from_existing_index(
        embeddings,
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        index_name="vector",
        retrieval_query=lc_retrieval_query,
    )
    
    # 进行向量相似性搜索
    docs = lc_vector.similarity_search(
        query,
        k=topEntities,
        params={
            "topChunks": topChunks,
            "topOutsideRels": topOutsideRels,
            "topInsideRels": topInsideRels,
        },
    )
    report_data = docs[0].page_content

    # 返回检索结果
    return report_data

def global_retriever(query: str, level: int = 0) -> str:
    """全局检索器：通过社区摘要检索全局信息"""
    # 相关性评估提示词
    evaluate_system_prompt = """
    ---角色--- 
    你是一位有用的相关性评估助手，负责评估社区摘要与用户问题的相关程度。

    ---任务描述--- 
    - 仔细分析用户问题和提供的社区摘要
    - 评估社区摘要与用户问题的相关程度，给出0-100的整数分数
    - 评分标准：
        90-100分：摘要直接且完整地回答了用户问题
        80-89分：摘要高度相关，提供了大部分关键信息
        70-79分：摘要相关，提供了部分有用信息
        60-69分：摘要有一定相关性，但信息有限
        0-59分：摘要与问题不相关或相关性很低

    ---回复格式--- 
    仅返回一个JSON对象，包含分数和简要理由：
    {{"score": 整数分数, "reason": "一句话的评分理由"}}
    """

    # 构建提示词
    evaluate_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", evaluate_system_prompt),
            (
                "human",
                """
                用户问题：{question}
                
                社区摘要：{context_data}
                
                请评估该摘要与问题的相关度：
                """,
            ),
        ]
    )
    evaluate_chain = evaluate_prompt | llm | StrOutputParser()
        
    # 数据库连接
    graph = Neo4jGraph(refresh_schema=False)
    
    # 查询社区数据
    community_data = graph.query(
        """
        MATCH (c:__Community__)
        WHERE c.level = $level 
        AND c.summary IS NOT NULL 
        AND c.summary <> ""
        RETURN {communityId:c.id, summary:c.summary} AS output
        """,
        params={"level": level},
    )
    
    # 社区处理函数
    def process_community(community_info):
        community_id = community_info["communityId"]
        summary = community_info["summary"]
        
        try:
            # 获取相关度评分
            score_response = evaluate_chain.invoke(
                {"question": query, "context_data": summary}
            )
            
            # 解析评分结果
            score_data = json.loads(score_response)
            score = score_data.get("score", 0)
            
            # 仅保留分数>=60的摘要
            if score >= 60:
                return {
                    "communityId": community_id,
                    "summary": summary,
                    "score": score
                }
            else:
                return None
                
        except Exception:
            return None

    # 存储符合条件的结果
    qualified_results = []
    # 执行并行处理
    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = {
            executor.submit(process_community, community["output"]): 
                community for community in community_data
        }
        
        # 收集结果
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                qualified_results.append(result)
    
    # 按分数降序排列
    qualified_results.sort(key=lambda x: x["score"], reverse=True)
    return qualified_results

def get_source(source_id):
    """根据给定的ID查看文本块或社区摘要"""
    ChunkCypher = """
    MATCH (n:`__Chunk__`) WHERE n.id = $id RETURN n.fileName, n.text
    """
    CommunityCypher = """
    MATCH (n:`__Community__`) WHERE n.id = $id RETURN n.id, n.summary
    """
    
    temp = len(source_id.split("-"))
    if temp == 2:
        result = graph.query(CommunityCypher, params={"id": source_id})
    else:
        result = graph.query(ChunkCypher, params={"id": source_id})
    
    if result:
        if temp == 2:  # Community
            resp = "\n\n##### 社区ID: " + result[0]["n.id"] + "\n\n社区摘要:\n\n" + result[0]["n.summary"] + "\n\n"
        else:  # Chunk
            resp = "\n\n##### 文件名称:" + result[0]["n.fileName"] + "\n\n文本内容:\n\n" + result[0]["n.text"] + "\n\n"
    else:
        resp = "##### 未检索到该语料。"
    return resp
