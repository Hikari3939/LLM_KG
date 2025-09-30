import os
from tqdm import tqdm
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Neo4jVector
from langchain_core.output_parsers import StrOutputParser
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langsmith import traceable

# 加载环境变量
load_dotenv(".env")
NEO4J_URI = os.environ.get("NEO4J_URI")
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
LANGCHAIN_API_KEY = os.environ.get("LANGCHAIN_API_KEY")

# 加载LLM
llm = ChatDeepSeek(
    model='deepseek-chat',
    temperature=0.7
)

# 加载Embedding模型
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs = {"device": "cpu"},
    cache_folder="./model"
)

# LLM以多段的形式回答问题。
response_type: str = "多个段落"

# 全局检索器REDUCE阶段系统提示词
REDUCE_SYSTEM_PROMPT = """
---角色--- 
你是一个有用的助手，请根据用户输入的上下文，综合上下文中多个要点列表的数据，来回答问题，并遵守回答要求。

---任务描述--- 
总结来自多个不同要点列表的数据，生成要求长度和格式的回复，以回答用户的问题。 

---回答要求---
- 你要严格根据要点列表的内容回答，禁止根据常识和已知信息回答问题。
- 对于不知道的信息，直接回答"不知道"。
- 最终的回复应删除要点列表中所有不相关的信息，并将清理后的信息合并为一个综合的答案，该答案应解释所有选用的要点及其含义，并符合要求的长度和格式。 
- 根据要求的长度和格式，把回复划分为适当的章节和段落，并用markdown语法标记回复的样式。 
- 回复应保留之前包含在要点列表中的要点引用，并且包含引用要点来源社区原始的communityId，但不要提及各个要点在分析过程中的作用。 
- **不要在一个引用中列出超过5个要点引用的ID**，相反，列出前5个最相关要点引用的顺序号作为ID。 
- 不要包括没有提供支持证据的信息。
例如： 
#############################
"X是Y公司的所有者，他也是X公司的首席执行官{{'points':[(1,'0-0'),(3,'0-0')]}}，
受到许多不法行为指控{{'points':[(2,'0-0'), (3,'0-0'), (6,'0-1'), (9,'0-1'), (10,'0-3')]}}。" 
其中1、2、3、6、9、10表示相关要点引用的顺序号，'0-0'、'0-1'、'0-3'是要点来源的communityId。 
#############################

---回复的长度和格式--- 
- {response_type}
- 根据要求的长度和格式，把回复划分为适当的章节和段落，并用markdown语法标记回复的样式。  
- 输出要点引用的格式：
{{'points': [逗号分隔的要点元组]}}
每个要点元组的格式如下：
(要点顺序号, 来源社区的communityId)
例如：
{{'points':[(1,'0-0'),(3,'0-0')]}}
{{'points':[(2,'0-0'), (3,'0-0'), (6,'0-1'), (9,'0-1'), (10,'0-3')]}}
- 要点引用的说明放在引用之后，不要单独作为一段。
例如： 
#############################
"X是Y公司的所有者，他也是X公司的首席执行官{{'points':[(1,'0-0'),(3,'0-0')]}}，
受到许多不法行为指控{{'points':[(2,'0-0'), (3,'0-0'), (6,'0-1'), (9,'0-1'), (10,'0-3')]}}。" 
其中1、2、3、6、9、10表示相关要点引用的顺序号，'0-0'、'0-1'、'0-3'是要点来源的communityId。
#############################
"""

# 局部检索器
@traceable
def local_retriever(query: str, response_type: str = response_type, chat_history=None) -> str:
    # 检索语料数量限制
    topChunks = 3
    topCommunities = 3
    topOutsideRels = 10
    topInsideRels = 10
    topEntities = 10

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
    // Entity - Report Mapping
    collect {
        UNWIND nodes as n
        MATCH (n)-[:IN_COMMUNITY]->(c:__Community__)
        WITH distinct c, c.community_rank as rank, c.weight AS weight
        RETURN c.summary 
        ORDER BY rank, weight DESC
        LIMIT $topCommunities
    } AS report_mapping,
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
    // We don't have covariates or claims here
    RETURN {Chunks: text_mapping, Reports: report_mapping, 
        Relationships: outsideRels + insideRels, 
        Entities: entities} AS text, 1.0 AS score, {} AS metadata
    """

    # 系统提示词
    lc_system_prompt="""
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
    - **不要在一个引用中列出超过5个引用记录的ID**，相反，列出前5个最相关的引用记录ID。 
    - 不要包括没有提供支持证据的信息。
    例如： 
    #############################
    “X是Y公司的所有者，他也是X公司的首席执行官，他受到许多违规行为指控，其中的一些已经涉嫌违法。” 

    {{'data': {{'Entities':[3], 'Reports':[2, 6], 'Relationships':[12, 13, 15, 16, 64], 'Chunks':['d0509111239ae77ef1c630458a9eca372fb204d6','74509e55ff43bc35d42129e9198cd3c897f56ecb'] }} }}
    #############################
    ---回复的长度和格式--- 
    - {response_type}
    - 根据要求的长度和格式，把回复划分为适当的章节和段落，并用markdown语法标记回复的样式。  
    - 在回复的最后才输出数据引用的情况，单独作为一段。
    输出引用数据的格式：
    {{'data': {{'Entities':[逗号分隔的顺序号列表], 'Reports':[逗号分隔的顺序号列表], 'Relationships':[逗号分隔的顺序号列表], 'Chunks':[逗号分隔的id列表] }} }}
    例如：
    {{'data': {{'Entities':[3], 'Reports':[2, 6], 'Relationships':[12, 13, 15, 16, 64], 'Chunks':['d0509111239ae77ef1c630458a9eca372fb204d6','74509e55ff43bc35d42129e9198cd3c897f56ecb'] }} }}

    """
    
    # 构建提示词
    lc_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                lc_system_prompt,
            ),
            (
                "human",
                """
                ---分析报告--- 
                请注意，下面提供的分析报告按**重要性降序排列**。
                
                {report_data}
                

                用户的问题是：
                {question}
                """,
            ),
        ]
    )
    
    # 局部检索的chain
    lc_chain = lc_prompt | llm | StrOutputParser()
    
    # 局部检索的Neo4j向量存储与索引
    lc_vector = Neo4jVector.from_existing_index(
        embeddings,
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        index_name="vector",
        retrieval_query=lc_retrieval_query,
    )
    
    # 定义向量检索器
    retriever = lc_vector.as_retriever(search_kwargs={"k": topEntities})
    
    # 如果提供了聊天历史，使用历史感知检索器
    if chat_history is not None and len(chat_history) > 0:
        # 上下文检索器的系统提示词
        contextualize_q_system_prompt = """
        给定一组聊天记录和最新的用户问题 ，
        该问题可能会引用聊天记录中的上下文， 
        据此构造一个不需要聊天记录也可以理解的独立问题， 
        不要回答它。
        如果需要，就重新构造出上述的独立问题，否则按原样返回原来的问题。
        """
        
        # 上下文检索器的提示词
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        
        # 定义上下文检索器
        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )
        
        # 局部查询的提示词（支持历史）
        lc_prompt_with_history = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    lc_system_prompt,
                ),
                MessagesPlaceholder("chat_history"),        
                (
                    "human",
                    """
                    ---分析报告--- 
                    请注意，下面提供的分析报告按**重要性降序排列**。
                        
                    {context}
                        

                    用户的问题是：
                    {input}
                    """,
                ),
            ]
        )
        
        # 定义局部查询的问答链
        question_answer_chain_with_history = create_stuff_documents_chain(llm, lc_prompt_with_history)
        # 定义支持上下文的局部查询问答链
        rag_chain_with_history = create_retrieval_chain(history_aware_retriever, question_answer_chain_with_history)
        
        # 使用历史感知检索器
        ai_msg = rag_chain_with_history.invoke({
            "input": query, 
            "response_type": response_type, 
            "chat_history": chat_history
        })
        return ai_msg["answer"]
    else:
        # 原有的处理方式（无历史上下文）
        # 先进行向量相似性搜索
        docs = lc_vector.similarity_search(
            query,
            k=topEntities,
            params={
                "topChunks": topChunks,
                "topCommunities": topCommunities,
                "topOutsideRels": topOutsideRels,
                "topInsideRels": topInsideRels,
            },
        )
        
        print(docs[0].page_content)
        
        # 向量相似性搜索的结果注入提示词并提交给LLM
        lc_response = lc_chain.invoke(
            {
                "report_data": docs[0].page_content,
                "question": query,
                "response_type": response_type,
            }
        )
        
        # 返回LLM的答复
        return lc_response


# 全局检索器 - MAP阶段
@traceable
def global_retriever(query: str, level: int) -> str:
    """
    全局检索器的MAP阶段，只返回中间结果（要点列表）
    """
    # MAP阶段生成中间结果的prompt与chain
    map_system_prompt = """
    ---角色--- 
    你是一位有用的助手，可以回答有关所提供社区摘要的问题。 

    ---任务描述--- 
    - 生成一个回答用户问题所需的要点列表，总结输入社区摘要中的所有相关信息。
    - 你应该使用下面提供的社区摘要作为生成回复的主要上下文。
    - 仔细分析社区摘要，寻找与用户问题相关的任何信息，包括同义词和相关概念。
    - 如果社区摘要中包含相关信息（即使是间接相关的），请提取并总结这些信息。
    - 只有在社区摘要完全没有任何相关信息时才回答"不知道"。
    - 数据支持的要点应列出相关的社区引用作为参考，并列出产生该要点社区的communityId。
    - **不要在一个引用中列出超过5个引用记录的ID**。相反，列出前5个最相关引用记录的顺序号作为ID。

    ---回答要求---
    回复中的每个要点都应包含以下元素： 
    - 描述：对该要点的综合描述。 
    - 重要性评分：0-100之间的整数分数，表示该要点在回答用户问题时的重要性。"不知道"类型的回答应该得0分。 

    ---回复的格式--- 
    回复应采用JSON格式，如下所示： 
    {{ 
    "points": [ 
    {{"description": "Description of point 1 {{'communities': [community_ids list seperated by comma], 'communityId': communityId form context data}}", "score": score_value}}, 
    {{"description": "Description of point 2 {{'communities': [community_ids list seperated by comma], 'communityId': communityId form context data}}", "score": score_value}}, 
    ] 
    }}
    例如： 
    ####################
    {{"points": [
    {{"description": "X是Y公司的所有者，他也是X公司的首席执行官。 {{'communities': [1,3], 'communityId':'0-0'}}", "score": 80}}, 
    {{"description": "X受到许多不法行为指控。 {{'communities': [1,3], 'communityId':'0-0'}}", "score": 90}}
    ] 
    }}
    ####################
    """

    map_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                map_system_prompt,
            ),
            (
                "human",
                """
                ---社区摘要--- 
                {context_data}
                
                
                用户的问题是：
                {question}
                
                注意：请仔细分析社区摘要，寻找与问题相关的任何信息，包括同义词和相关概念。
                如果社区摘要中包含相关信息（即使是间接相关的），请提取并总结这些信息。
                """,
            ),
        ]
    )
    map_chain = map_prompt | llm | StrOutputParser()
    
    # 连接Neo4j
    graph = Neo4jGraph(refresh_schema=False)
    
    # 仅获取summary存在且不为空的社区
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
    print(f"找到 {len(community_data)} 个社区")
    
    # 并行处理所有社区
    intermediate_results = []
    max_workers = 12  # 并行线程数  
    
    # 处理单个社区的函数
    def process_community(community_info):
        i, community = community_info
        context_data = community["output"]
        
        try:
            intermediate_response = map_chain.invoke(
                {"question": query, "context_data": context_data}
            )
            return i, intermediate_response, None
        except Exception as e:
            return i, None, str(e)

    # 并行处理所有社区
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_community = {
            executor.submit(process_community, (i, community)): i 
            for i, community in enumerate(community_data)
        }
        
        # 使用进度条显示进度
        with tqdm(total=len(community_data), desc="处理社区", unit="个") as pbar:
            for future in as_completed(future_to_community):
                i, response, error = future.result()
                
                if error:
                    print(f"\n处理社区 {i+1} 时出错: {error}")
                    pbar.update(1)
                    continue
                    
                if response:
                    intermediate_results.append(response)
                
                pbar.update(1)
    
    # 返回MAP阶段的中间结果
    return intermediate_results

# 辅助函数：管理聊天历史
def add_to_chat_history(question, answer, chat_history_list):
    """
    将问答对添加到对话历史中
    
    Args:
        question: 用户问题
        answer: AI回答
        chat_history_list: 对话历史列表
    
    Returns:
        list: 更新后的对话历史
    """
    chat_history_list.append(HumanMessage(content=question))
    chat_history_list.append(AIMessage(content=answer))
    return chat_history_list

def clear_chat_history(chat_history_list):
    """
    清空对话历史
    
    Args:
        chat_history_list: 对话历史列表
    
    Returns:
        list: 空的对话历史列表
    """
    chat_history_list.clear()
    return chat_history_list
