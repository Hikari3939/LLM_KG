import os
import time
from dotenv import load_dotenv
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_deepseek import ChatDeepSeek
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.config import RunnableConfig
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate
)

# 加载环境变量
load_dotenv(".env")
DEEPSEEK_API_KEY = os.environ["DEEPSEEK_API_KEY"]

# 指定模型名称
INSTRUCT_MODEL = 'deepseek-chat'

# 由LLM来最终决定哪些实体该合并 
def decide_entity_merge(candidates):
    system_template = """
    你是一名数据处理助理。您的任务是识别列表中的重复实体，并决定应合并哪些实体。 
    这些实体在格式或内容上可能略有不同，但本质上指的是同一个实体。运用你的分析技能来确定重复的实体。 
    以下是识别重复实体的规则： 
    1.语义上差异较小的实体应被视为重复。 
    2.格式不同但内容相同的实体应被视为重复。 
    3.引用同一现实世界对象或概念的实体，即使描述不同，也应被视为重复。 
    4.如果它指的是不同的数字、日期或产品型号，请不要合并实体。
    5.概念类实体和对象类实体请不要合并。
    输出格式：
    1.将要合并的实体输出为Python列表的格式，输出时保持它们输入时的原文。
    2.如果有多组可以合并的实体，每组输出为一个单独的列表，每组分开输出为一行。
    3.如果没有要合并的实体，就输出一个空的列表。
    4.只输出列表即可，不需要其它的说明。
    5.不要输出嵌套的列表，只输出列表。
    ###################### 
    -示例- 
    ###################### 
    Example 1:
    ['Star Ocean The Second Story R', 'Star Ocean: The Second Story R', 'Star Ocean: A Research Journey']
    #############
    Output:
    ['Star Ocean The Second Story R', 'Star Ocean: The Second Story R']
    #############################
    Example 2:
    ['Sony', 'Sony Inc', 'Google', 'Google Inc', 'OpenAI']
    #############
    Output:
    ['Sony', 'Sony Inc']
    ['Google', 'Google Inc']
    #############################
    Example 3:
    ['December 16, 2023', 'December 2, 2023', 'December 23, 2023', 'December 26, 2023']
    Output:
    []
    #############################
    Example 4:
    ['性别', '女性']
    Output:
    []
    #############################
    """
    user_template = """
    以下是要处理的实体列表： 
    {entities} 
    请识别重复的实体，提供可以合并的实体列表。
    输出：
    """


    # 定义输出结构：列表的列表，每个内部列表包含应该合并的实体
    class DuplicateEntities(BaseModel):
        entities: List[str] = Field(
            description="Entities that represent the same object or real-world entity and should be merged"
        )

    class Disambiguate(BaseModel):
        merge_entities: Optional[List[DuplicateEntities]] = Field(
            description="Lists of entities that represent the same object or real-world entity and should be merged"
        )


    # 连接模型
    structured_llm  = ChatDeepSeek(
        model=INSTRUCT_MODEL,
        temperature=1.0
    ).with_structured_output(Disambiguate)
    
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    human_message_prompt = HumanMessagePromptTemplate.from_template(user_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = chat_prompt | structured_llm

    # 调用LLM得到可以合并实体的列表
    inputs = [{"entities": ', '.join(candidate['combinedResult'])} for candidate in candidates]
    results = chain.batch(inputs, config=RunnableConfig(max_concurrency=12))

    # 解析结果
    merged_entities = []
    for result in results:
        if result.merge_entities is not None:
            for el in result.merge_entities:
                if len(el.entities) > 1:
                    merged_entities.append(el.entities)
    
    return merged_entities
    
# 重写实体节点描述
def rewrite_entity_descriptions(graph, min_length = 500):        
    # 获取需要重写的实体节点
    query = """
    MATCH (n:__Entity__)
    WHERE n.description IS NOT NULL AND size(n.description) > $min_length
    RETURN elementId(n) as node_id, n.description as description
    """
    
    entities = graph.query(query, params={"min_length": min_length})
    
    # 构建提示模板链
    prompt = ChatPromptTemplate.from_template(
        """
        请重写以下实体描述，合并其中的重复信息，使其更加简洁、连贯且信息完整。保持专业性和准确性。

        原始描述：
        {description}

        重写要求：
        1. 合并重复部分
        2. 保留所有重要信息
        3. 使描述更加流畅自然
        4. 不要添加新的信息
        5. 请直接输出重写后的描述

        重写后的描述：
        """
    )
    
    llm  = ChatDeepSeek(
        model=INSTRUCT_MODEL,
    )

    chain = prompt | llm
    
    # 准备批量处理数据
    inputs = [{"description": entity["description"]} for entity in entities]
    
    # 批量并行处理
    if inputs:
        responses = chain.batch(inputs, config=RunnableConfig(max_concurrency=12))
        
        # 批量更新数据库
        for entity, response in zip(entities, responses):
            new_description = response.content.strip()
            
            update_query = """
            MATCH (n:__Entity__)
            WHERE elementId(n) = $node_id
            SET n.description = $new_description
            """
            
            _ = graph.query(update_query, params={
                "node_id": entity["node_id"], 
                "new_description": new_description
            })

# 重写关系描述
def rewrite_relationship_descriptions(graph, min_length = 60):        
    # 获取需要重写的关系
    query = """
    MATCH (:__Entity__)-[r]->(:__Entity__)
    WHERE r.description IS NOT NULL AND size(r.description) > $min_length
    RETURN elementId(r) as rel_id, r.description as description
    """
    
    relationships = graph.query(query, params={"min_length": min_length})
    
    # 构建提示模板链
    prompt = ChatPromptTemplate.from_template(
        """请重写以下关系描述，合并其中的重复信息，使其更加简洁、连贯且信息完整。保持专业性和准确性。

        原始描述：
        {description}

        重写要求：
        1. 合并重复部分
        2. 保留所有重要信息
        3. 使描述更加流畅自然
        4. 不要添加新的信息
        5. 请直接输出重写后的描述

        重写后的描述："""
    )
    
    llm  = ChatDeepSeek(
        model=INSTRUCT_MODEL,
    )
    
    chain = prompt | llm
    
    # 准备批量处理数据
    inputs = [{"description": rel["description"]} for rel in relationships]
    
    # 批量并行处理
    if inputs:
        responses = chain.batch(inputs, config=RunnableConfig(max_concurrency=12))
        
        # 批量更新数据库
        for rel, response in zip(relationships, responses):
            new_description = response.content.strip()
            
            update_query = """
            MATCH (:__Entity__)-[r]->(:__Entity__)
            WHERE elementId(r) = $rel_id
            SET r.description = $new_description
            """

            _ = graph.query(update_query, params={
                "rel_id": rel["rel_id"],
                "new_description": new_description
            })

# 使用LLM进行社区摘要
# 按优先级准备社区信息，确保不超过token限制
def prepare_prioritized_string(info, max_tokens=120000):
    nodes = info['nodes']
    rels = info['rels']
    max_length = max_tokens / 0.6  # 1 个中文字符 ≈ 0.6 个 token
    
    # 构建节点ID到描述的映射
    node_desc_map = {node['id']: f"id: {node['id']}, type: {node['type']}, description: {node['description']}" 
                    for node in nodes}
    
    # 按优先级处理关系
    prioritized_nodes = []
    prioritized_rels = []
    current_length = 0
    
    for rel in rels:
        # 添加源节点描述（如果尚未添加）
        if rel['start'] in node_desc_map:
            node_desc = node_desc_map[rel['start']]
            if current_length + len(node_desc) < max_length:
                prioritized_nodes.append(node_desc)
                current_length += len(node_desc)
                del node_desc_map[rel['start']]  # 避免重复添加
        
        # 添加目标节点描述（如果尚未添加）
        if rel['end'] in node_desc_map:
            node_desc = node_desc_map[rel['end']]
            if current_length + len(node_desc) < max_length:
                prioritized_nodes.append(node_desc)
                current_length += len(node_desc)
                del node_desc_map[rel['end']]  # 避免重复添加
        
        # 添加关系描述
        rel_desc = f"{rel['start']} --[{rel['type']}]--> {rel['end']}: {rel['description']}"
        if current_length + len(rel_desc) < max_length:
            prioritized_rels.append(rel_desc)
            current_length += len(rel_desc)
        else:
            break  # 达到token限制，停止添加
    
    # 添加剩余节点
    for node_desc in node_desc_map.values():
        if current_length + len(node_desc) < max_length:
            prioritized_nodes.append(node_desc)
            current_length += len(node_desc)
        else:
            break  # 达到token限制，停止添加

    nodes_str = "Nodes are:\n"
    nodes_str += "\n".join(prioritized_nodes)

    rels_str = "Relationships are:\n"
    rels_str += "\n".join(prioritized_rels)

    return nodes_str + "\n" + rels_str

# 进行摘要并存入数据库
def community_abstract(graph):
    # 检索社区所包含的结点与边的信息
    community_info = graph.query(
        """
        MATCH (c:`__Community__`)<-[:IN_COMMUNITY]-(e:__Entity__)
        WHERE c.level IN [0]
        WITH c, collect(e) AS nodes
        WHERE size(nodes) > 3
        // 获取社区内所有实体节点之间的关系
        WITH c, nodes
        UNWIND nodes AS n
        OPTIONAL MATCH (n)-[r]-(m)
        WHERE type(r) <> 'IN_COMMUNITY' AND m IN nodes
        // 计算节点度中心性（显著度）
        WITH c, nodes, collect(DISTINCT r) AS rels
        WITH c, rels,
                [n IN nodes | {
                    id: n.id, 
                    description: n.description, 
                    type: [el in labels(n) WHERE el <> '__Entity__'],
                    degree: size([(n)--() | 1])  // 计算节点的度
                }] AS nodes_with_degree
        // 计算关系的优先级（源节点度 + 目标节点度）
        WITH c, nodes_with_degree,
                [r IN rels | {
                    start: startNode(r).id, 
                    end: endNode(r).id,
                    type: type(r), 
                    description: r.description,
                    priority: (
                        [n IN nodes_with_degree WHERE n.id = startNode(r).id | n.degree][0] +
                        [n IN nodes_with_degree WHERE n.id = endNode(r).id | n.degree][0]
                    )
                }] AS rels_with_priority
        WITH c,
            // 按节点度降序排序节点
            apoc.coll.sortMaps(nodes_with_degree, "degree") AS sorted_nodes,
            // 按优先级降序排序关系
            apoc.coll.sortMaps(rels_with_priority, "priority") AS sorted_rels
        RETURN c.id AS communityId,
               sorted_nodes AS nodes,
               sorted_rels AS rels
        """
    )
    
    community_template = """
    ---
    基于所提供的属于同一图社区的节点和关系，
    生成所提供图社区信息的自然语言摘要。
    要求输出为一整个段落，直接输出最终摘要。
    ---
    图社区信息：
    {community_info}
    ---
    摘要：
    """  
    community_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "给定一个输入三元组，生成信息摘要。没有序言。",
            ),
            ("human", community_template),
        ]
    )
    
    llm  = ChatDeepSeek(
        model=INSTRUCT_MODEL,
    )

    community_chain = community_prompt | llm | StrOutputParser()
    
    t0 = time.time()
    # 准备批量处理的输入
    batch_inputs = []
    community_id_map = {}  # 用于存储索引和对应的社区ID
    for idx, info in enumerate(community_info):
        stringify_info = prepare_prioritized_string(info)
        batch_inputs.append({'community_info': stringify_info})
        community_id_map[idx] = info['communityId']  # 使用索引作为键

    # 使用batch并行处理
    summaries = community_chain.batch(batch_inputs, config=RunnableConfig(max_concurrency=12))

    # 组合结果
    results = []
    for idx, summary in enumerate(summaries):
        results.append({
            "community": community_id_map[idx],
            "summary": summary
        })
    t2 = time.time()
    print("摘要耗时：",t2-t0,"秒")
    print("")

    # 存储社区摘要
    graph.query("""
    UNWIND $data AS row
    MERGE (c:__Community__ {id:row.community})
    SET c.summary = row.summary
    """, params={"data": results})
