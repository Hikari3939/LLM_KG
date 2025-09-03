import os
import re
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langchain_neo4j import GraphCypherQAChain
from langchain_community.vectorstores import Neo4jVector
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever

from my_packages.MyNeo4j import MyNeo4jGraph

# 加载环境变量
load_dotenv(".env")
NEO4J_URI = os.environ["NEO4J_URI"]
NEO4J_USERNAME = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]

DEEPSEEK_API_KEY = os.environ["DEEPSEEK_API_KEY"]

# 指定模型名称
INSTRUCT_MODEL = 'deepseek-chat'

# ----------------------------
# 简单路由：局部/全局
# ----------------------------
def classify_query(text: str) -> str:
    if not text:
        return "global"
    t = text.strip().lower()
    if t.startswith("局部:"):
        return "local"
    if t.startswith("全局:"):
        return "global"
    # 明显聊天/编程/闲聊 → 全局
    if re.search(r"你是谁|笑话|新闻|天气|怎么写代码|python|java|报错|debug", t):
        return "global"
    # 医学/关系/实体等关键词 → 局部
    local_keys = [
        "是什么", "有哪些", "关系", "定义", "属性", "症状", "并发症", "病因",
        "原因", "治疗", "药物", "属于", "类型", "关联", "实体", "路径", "相似",
        "who", "what", "which", "relation", "symptom", "cause", "treat",
    ]
    if any(k in text for k in local_keys):
        return "local"
    return "global"

if __name__ == '__main__':
    # 连接neo4j数据库
    graph = MyNeo4jGraph(
        url=NEO4J_URI, 
        username=NEO4J_USERNAME, 
        password=NEO4J_PASSWORD,
        enhanced_schema=True
    )
    print("数据库成功连接")

    # 刷新数据库结构信息
    graph.refresh_schema()

    # 连接大模型
    llm = ChatDeepSeek(model=INSTRUCT_MODEL, temperature=0)

    # 全局（直接 LLM）已就绪；下面构建“局部/GraphRAG”链
    # 向量索引名称（需与已创建的一致）
    index_name = "vector"

    # 局部检索参数
    topChunks = 3
    topCommunities = 3
    topOutsideRels = 10
    topInsideRels = 10
    topEntities = 10

    # 和参考实现一致的检索 Cypher（按重要性聚合多模态文本）
    lc_retrieval_query = f"""
    WITH collect(node) as nodes
    // Entity - Text Unit Mapping
    WITH
    collect {{
        UNWIND nodes as n
        MATCH (n)<-[:MENTIONS]-(c:__Chunk__)
        WITH distinct c, count(distinct n) as freq
        RETURN {{id:c.id, text: c.text}} AS chunkText
        ORDER BY freq DESC
        LIMIT {topChunks}
    }} AS text_mapping,
    // Entity - Report Mapping
    collect {{
        UNWIND nodes as n
        MATCH (n)-[:IN_COMMUNITY]->(c:__Community__)
        WITH distinct c, c.community_rank as rank, c.weight AS weight
        RETURN c.summary 
        ORDER BY rank, weight DESC
        LIMIT {topCommunities}
    }} AS report_mapping,
    // Outside Relationships 
    collect {{
        UNWIND nodes as n
        MATCH (n)-[r]-(m:__Entity__) 
        WHERE NOT m IN nodes
        RETURN r.description AS descriptionText
        ORDER BY r.weight DESC 
        LIMIT {topOutsideRels}
    }} as outsideRels,
    // Inside Relationships 
    collect {{
        UNWIND nodes as n
        MATCH (n)-[r]-(m:__Entity__) 
        WHERE m IN nodes
        RETURN r.description AS descriptionText
        ORDER BY r.weight DESC 
        LIMIT {topInsideRels}
    }} as insideRels,
    // Entities description
    collect {{
        UNWIND nodes as n
        RETURN n.description AS descriptionText
    }} as entities
    // We don't have covariates or claims here
    RETURN {{Chunks: text_mapping, Reports: report_mapping, 
           Relationships: outsideRels + insideRels, 
           Entities: entities}} AS text, 1.0 AS score, {{}} AS metadata
    """

    # 用与构建时相同的 Embedding 模型（BGE-M3）
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        cache_folder="./model",
    )

    lc_vector = Neo4jVector.from_existing_index(
        embeddings,
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        index_name=index_name,
        retrieval_query=lc_retrieval_query,
    )
    retriever = lc_vector.as_retriever(search_kwargs={"k": topEntities})

    response_type = "多个段落"

    LC_SYSTEM_PROMPT = """
    ---角色--- 
    您是一个有用的助手，请根据用户输入的上下文，综合上下文中多个分析报告的数据，来回答问题，并遵守回答要求。

    ---任务描述--- 
    总结来自多个不同分析报告的数据，生成要求长度和格式的回复，以回答用户的问题。 

    ---回答要求---
    - 你要严格根据分析报告的内容回答，禁止根据常识和已知信息回答问题。
    - 对于不知道的问题，直接回答“不知道”。
    - 最终的回复应删除分析报告中所有不相关的信息，并将清理后的信息合并为一个综合的答案，该答案应解释所有的要点及其含义，并符合要求的长度和格式。 
    - 根据要求的长度和格式，把回复划分为适当的章节和段落，并用markdown语法标记回复的样式。 
    - 回复应保留之前包含在分析报告中的所有数据引用，但不要提及各个分析报告在分析过程中的作用。 
    - 如果回复引用了Entities、Reports及Relationships类型分析报告中的数据，则用它们的顺序号作为ID。
    - 如果回复引用了Chunks类型分析报告中的数据，则用原始数据的id作为ID。 
    - 不要在一个引用中列出超过5个引用记录的ID，相反，列出前5个最相关的引用记录ID。 
    - 不要包括没有提供支持证据的信息。
    """

    contextualize_q_system_prompt = """
    给定一组聊天记录和最新的用户问题，
    该问题可能会引用聊天记录中的上下文，
    据此构造一个不需要聊天记录也可以理解的独立问题，
    不要回答它。
    如果需要，就重新构造出上述的独立问题，否则按原样返回原来的问题。
    """

    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )

    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    lc_prompt_with_history = ChatPromptTemplate.from_messages(
        [
            ("system", LC_SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            (
                "human",
                """
                ---分析报告--- 
                请注意，下面提供的分析报告按重要性降序排列。
                
                {context}
                
                用户的问题是：
                {input}
                """,
            ),
        ]
    )

    question_answer_chain_with_history = create_stuff_documents_chain(
        llm, lc_prompt_with_history
    )
    rag_chain_with_history = create_retrieval_chain(
        history_aware_retriever, question_answer_chain_with_history
    )

    chat_history = []

    while True:
        questions = input("\n请输入问题：")
        if questions!="exit":
            route = classify_query(questions)
            if route == "local":
                ai_msg = rag_chain_with_history.invoke({
                    "input": questions,
                    "response_type": response_type,
                    "chat_history": chat_history,
                })
                print(ai_msg.get("answer", ""))
                chat_history.append(HumanMessage(content=questions))
                chat_history.append(AIMessage(content=ai_msg.get("answer", "")))
            else:
                resp = llm.invoke(questions)
                print(resp.content)
        else:
            break

    # 关闭数据库
    graph.close()
