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

# 使用LLM进行社区摘要
# 转换社区信息为字符串
def prepare_string(data):
    nodes_str = "Nodes are:\n"
    for node in data['nodes']:
        node_id = node['id']
        node_type = node['type']
        if 'description' in node and node['description']:
            node_description = f", description: {node['description']}"
        else:
            node_description = ""
        nodes_str += f"id: {node_id}, type: {node_type}{node_description}\n"

    rels_str = "Relationships are:\n"
    for rel in data['rels']:
        start = rel['start']
        end = rel['end']
        rel_type = rel['type']
        if 'description' in rel and rel['description']:
            description = f", description: {rel['description']}"
        else:
            description = ""
        rels_str += f"({start})-[:{rel_type}]->({end}){description}\n"

    return nodes_str + "\n" + rels_str

# 进行摘要并存入数据库
def community_abstract(graph):
    # 检索0和1级的社区所包含的结点与边的信息
    community_info = graph.query(
        """
        MATCH (c:`__Community__`)<-[:IN_COMMUNITY*]-(e:__Entity__)
        WHERE c.level IN [0,1]
        WITH c, collect(e ) AS nodes
        WHERE size(nodes) > 3
        CALL apoc.path.subgraphAll(nodes[0], {
            whitelistNodes:nodes
        })
        YIELD relationships
        RETURN c.id AS communityId,
            [n in nodes | {id: n.id, description: n.description, type: [el in labels(n) WHERE el <> '__Entity__'][0]}] AS nodes,
            [r in relationships | {start: startNode(r).id, type: type(r), end: endNode(r).id, description: r.description}] AS rels
        """
    )
    
    community_template = """
    基于所提供的属于同一图社区的节点和关系， 
    生成所提供图社区信息的自然语言摘要： 
    {community_info} 
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
        temperature=1.0
    )

    community_chain = community_prompt | llm | StrOutputParser()
        
    # 准备批量处理的输入
    batch_inputs = []
    community_id_map = {}  # 用于存储索引和对应的社区ID
    for idx, info in enumerate(community_info):
        stringify_info = prepare_string(info)
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
    
    # 存储社区摘要
    graph.query("""
    UNWIND $data AS row
    MERGE (c:__Community__ {id:row.community})
    SET c.summary = row.summary
    """, params={"data": results})
