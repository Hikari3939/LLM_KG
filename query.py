import os
import re
import pprint
from typing import Dict, List, Any, TypedDict, Annotated, Literal, Sequence
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langchain_community.vectorstores import Neo4jVector
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from langchain.tools.retriever import create_retriever_tool
from langchain_core.output_parsers import StrOutputParser
from langsmith import traceable

from my_packages.MyNeo4j import MyNeo4jGraph

# 全局查询模式：匹配闲聊、通用问答等非知识图谱相关查询
GLOBAL_PATTERN = re.compile(r"(你是谁|笑话|新闻|天气|怎么写代码|python|java|报错|debug)", re.I)

# 规则打分权重：本地/全局关键词（可按需调整或外置配置）
LOCAL_CN_WEIGHTS = {
    "是什么": 0.8, "有哪些": 0.7, "关系": 0.8, "定义": 0.8, "属性": 0.6,
    "症状": 1.0, "并发症": 0.9, "病因": 0.9, "原因": 0.9, "治疗": 1.0,
    "药物": 0.9, "属于": 0.5, "类型": 0.6, "关联": 0.6, "实体": 0.5,
    "路径": 0.6, "相似": 0.5, "预后": 0.9, "指南": 0.8, "适应症": 0.9,
    "禁忌症": 0.9, "用药": 1.0, "剂量": 0.9, "疗效": 0.9, "不良反应": 0.9,
}
LOCAL_EN_WEIGHTS = {
    "who": 0.6, "what": 0.6, "which": 0.6, "relation": 0.8, "symptom": 1.0,
    "cause": 0.9, "treat": 1.0, "contraindication": 0.9, "dosage": 0.9,
    "guideline": 0.8, "prognosis": 0.9,
}
GLOBAL_WEIGHTS = {
    "你是谁": 1.0, "笑话": 1.0, "新闻": 0.9, "天气": 0.9, "怎么写代码": 1.0,
    "python": 0.9, "java": 0.9, "报错": 0.8, "debug": 0.8,
}

# 问句模板与模式（中文/英文），命中时按目标类别加权
_QUESTION_PATTERNS = [
    # 中文本地类
    (re.compile(r"^(什么是|如何|为什么|是否|有无|怎样|怎么|请解释)"), 0.6, "local"),
    (re.compile(r"(有哪些|列出|给出|总结|比较)"), 0.5, "local"),
    (re.compile(r"(症状|病因|治疗|药物|预后|并发症|适应症|禁忌症|用药|剂量)"), 0.8, "local"),
    # 英文本地类
    (re.compile(r"\b(who|what|which|how|why|whether)\b"), 0.5, "local"),
    (re.compile(r"\b(symptom|cause|treat|prognosis|guideline|dosage|contraindication)s?\b"), 0.8, "local"),
    # 全局类
    (re.compile(r"(你是谁|讲个?笑话|新闻|天气)"), 0.9, "global"),
    (re.compile(r"\b(debug|python|java|error|exception)\b"), 0.7, "global"),
]

# 阈值配置
LOCAL_THRESHOLD = 0.55
GLOBAL_THRESHOLD = 0.55
MIN_CONFIDENCE = 0.35  # 两类得分都过低时回退全局
NEGATION_LOCAL_PENALTY = 0.2  # 否定词轻微降低本地分


load_dotenv(".env")

# 设置LangSmith日志记录
if os.environ.get("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    print("LangSmith日志记录已启用")
else:
    print("警告: 未找到LANGCHAIN_API_KEY环境变量，LangSmith日志记录功能将被禁用")
    os.environ["LANGCHAIN_TRACING_V2"] = "false"

# 从环境变量获取数据库和API配置
NEO4J_URI = os.environ["NEO4J_URI"]
NEO4J_USERNAME = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]
DEEPSEEK_API_KEY = os.environ["DEEPSEEK_API_KEY"]
INSTRUCT_MODEL = 'deepseek-chat'

# 调试和辅助函数
def my_add_messages(left, right):
    """
    调试函数：用于查看消息合并过程
    
    参数:
        left: 左侧消息
        right: 右侧消息
    
    返回:
        合并后的消息
    """
    print("\nLeft:\n")
    print(left)
    print("\nRight\n")
    print(right)
    return add_messages(left, right)

def _normalize_text(text: str) -> str:
    """
    文本标准化：去除首尾空白，统一大小写（使用casefold），转换全角到半角。
    """
    if not text:
        return ""
    t = text.strip()
    # 全角转半角
    def _to_halfwidth(s: str) -> str:
        res = []
        for ch in s:
            code = ord(ch)
            if code == 0x3000:
                code = 0x20
            elif 0xFF01 <= code <= 0xFF5E:
                code -= 0xFEE0
            res.append(chr(code))
        return ''.join(res)
    t = _to_halfwidth(t)
    return t.casefold()

def _contains_negation(text: str) -> bool:
    """
    简单否定检测：用于减少否定问句造成的误判，如“不是…是什么”。
    """
    if not text:
        return False
    neg_words = [
        "不", "不是", "无", "没有", "非", "别", "别是", "别的不是",
        "not", "no", "without", "isn't", "aren't", "don't", "doesn't",
    ]
    return any(w in text for w in neg_words)

def _score_by_keywords(text: str, weights: dict) -> tuple:
    """
    根据加权关键词进行打分。
    返回: (score: float, hits: List[str])
    """
    score = 0.0
    hits = []
    for key, w in weights.items():
        if key in text:
            score += w
            hits.append(key)
    return score, hits


def _score_by_patterns(text: str, patterns: list) -> tuple:
    """
    根据问句模板/正则模式打分，分别累加到local/global。
    返回: ((local_score, local_hits), (global_score, global_hits))
    """
    local_score = 0.0
    global_score = 0.0
    local_hits = []
    global_hits = []
    for pat, weight, target in patterns:
        if pat.search(text):
            if target == "local":
                local_score += weight
                local_hits.append(pat.pattern)
            else:
                global_score += weight
                global_hits.append(pat.pattern)
    return (local_score, local_hits), (global_score, global_hits)


def _score_rules(text: str) -> dict:
    """
    规则层综合打分：关键词 + 模板。
    返回: {
        'local_score': float,
        'global_score': float,
        'features': { 'local': [...], 'global': [...] }
    }
    """
    norm = _normalize_text(text)

    # 关键词打分（本地/全局，含中英文）
    loc_cn, loc_cn_hits = _score_by_keywords(norm, LOCAL_CN_WEIGHTS)
    loc_en, loc_en_hits = _score_by_keywords(norm, LOCAL_EN_WEIGHTS)
    glo, glo_hits = _score_by_keywords(norm, GLOBAL_WEIGHTS)

    # 模式打分
    (loc_pat, loc_pat_hits), (glo_pat, glo_pat_hits) = _score_by_patterns(norm, _QUESTION_PATTERNS)

    local_score = loc_cn + loc_en + loc_pat
    global_score = glo + glo_pat

    # 否定轻微惩罚，避免“不是…是什么”类误导性本地触发
    if _contains_negation(norm):
        local_score = max(0.0, local_score - NEGATION_LOCAL_PENALTY)

    return {
        'local_score': local_score,
        'global_score': global_score,
        'features': {
            'local': loc_cn_hits + loc_en_hits + loc_pat_hits,
            'global': glo_hits + glo_pat_hits,
        }
    }

# 状态定义
class AgentState(TypedDict):
    """
    Agent状态类型定义
    
    定义工作流中传递的状态结构，包含消息序列
    messages字段使用add_messages函数进行消息合并，而不是替换
    """
    # The add_messages function defines how an update should be processed
    # Default is to replace. add_messages says "append"
    messages: Annotated[Sequence[BaseMessage], add_messages]

@traceable
def classify_query(text: str) -> str:
    """
    查询分类函数（增强版 - 规则打分）
    
    规则来源：
      - 加权关键词（中/英）
      - 问句模板/正则模式
      - 否定惩罚
    决策：
      - 若 local_score 与 global_score 都低于 MIN_CONFIDENCE，则回退 'global'
      - 若 local_score >= LOCAL_THRESHOLD 且大于 global_score -> 'local'
      - 若 global_score >= GLOBAL_THRESHOLD 且 >= local_score -> 'global'
      - 否则按较大者返回（平局时回退 'global'）
    """
    if not text or not text.strip():
        return "global"

    # 先用旧的全局模式快速短路（如闲聊/天气/编程）
    if GLOBAL_PATTERN.search(_normalize_text(text)):
        return "global"

    result = _score_rules(text)
    local_score = result['local_score']
    global_score = result['global_score']

    # 低置信度回退
    if max(local_score, global_score) < MIN_CONFIDENCE:
        return "global"

    # 阈值判定
    if local_score >= LOCAL_THRESHOLD and local_score > global_score:
        return "local"
    if global_score >= GLOBAL_THRESHOLD and global_score >= local_score:
        return "global"

    # 兜底（较大者），相等时偏全局
    return "local" if local_score > global_score else "global"

@traceable
def grade_documents(state, llm) -> Literal["generate", "rewrite"]:
    """
    文档相关性评估函数
    
    使用LLM评估检索到的文档是否与用户问题相关，决定是生成答案还是重写查询
    
    参数:
        state: 当前工作流状态，包含消息历史
        llm: 语言模型实例
    
    返回:
        Literal["generate", "rewrite"]: 
            - "generate": 文档相关，可以生成答案
            - "rewrite": 文档不相关，需要重写查询
    
    工作流程:
        1. 从状态中提取用户问题和检索到的文档
        2. 使用LLM评估文档相关性
        3. 根据评估结果决定下一步操作
    """
    print("---CHECK RELEVANCE---")

    # LLM
    model = llm

    # Prompt
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
        input_variables=["context", "question"],
    )
    
    # Chain
    chain = prompt | llm

    messages = state["messages"]
    last_message = messages[-1]

    # 在对话历史中，问题消息是倒数第3条
    question = messages[-3].content if len(messages) >= 3 else messages[-1].content
    
    docs = last_message.content

    scored_result = chain.invoke({"question": question, "context": docs})

    score = scored_result.content

    if score.lower().strip() == "yes":
        print("---DECISION: DOCS RELEVANT---")
        return "generate"
    else:
        print("---DECISION: DOCS NOT RELEVANT---")
        print(score)
        return "rewrite"

@traceable
def agent(state, llm, tools):
    """
    Agent决策函数
    
    调用语言模型，根据当前状态和可用工具决定下一步操作
    
    参数:
        state: 当前工作流状态
        llm: 语言模型实例
        tools: 可用工具列表
    
    返回:
        dict: 包含模型响应的消息字典
    
    功能:
        1. 将工具绑定到语言模型
        2. 根据当前状态生成响应
        3. 返回响应消息
    """
    print("---CALL AGENT---")
    messages = state["messages"]
    model = llm

    model = model.bind_tools(tools)
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}

@traceable
def rewrite(state, llm):
    """
    查询重写函数
    
    当检索到的文档不相关时，重写用户查询以获得更好的检索结果
    
    参数:
        state: 当前工作流状态
        llm: 语言模型实例
    
    返回:
        dict: 包含重写后查询的消息字典
    
    工作流程:
        1. 从状态中提取原始问题
        2. 使用LLM分析问题意图并重写
        3. 返回重写后的查询
    """
    print("---TRANSFORM QUERY---")
    messages = state["messages"]
    # 在对话历史中，问题消息是倒数第3条
    question = messages[-3].content if len(messages) >= 3 else messages[-1].content

    msg = [
        HumanMessage(
            content=f""" \n 
    Look at the input and try to reason about the underlying semantic intent / meaning. \n 
    Here is the initial question:
    \n ------- \n
    {question} 
    \n ------- \n
    Formulate an improved question: """,
        )
    ]

    # Rewriter
    model = llm

    response = model.invoke(msg)
    return {"messages": [response]}

@traceable
def generate(state, llm, LC_SYSTEM_PROMPT, response_type):
    """
    答案生成函数
    
    基于检索到的相关文档，生成结构化的答案
    
    参数:
        state: 当前工作流状态
        llm: 语言模型实例
        LC_SYSTEM_PROMPT: 系统提示词
        response_type: 响应类型
    
    返回:
        dict: 包含生成答案的AIMessage字典
    
    工作流程:
        1. 从状态中提取用户问题和检索到的文档
        2. 使用系统提示词构建生成链
        3. 基于文档内容生成结构化答案
        4. 返回AIMessage格式的答案
    """
    print("---GENERATE---")
    messages = state["messages"]
    # 在对话历史中，问题消息是倒数第3条
    question = messages[-3].content if len(messages) >= 3 else messages[-1].content
    
    last_message = messages[-1]

    docs = last_message.content

    # 局部查询的提示词
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                LC_SYSTEM_PROMPT,
            ),
            (
                "human",
                """
                ---分析报告--- 
                请注意，下面提供的分析报告按**重要性降序排列**。
                
                {context}
                

                用户的问题是：
                {question}
                """,
            ),
        ]
    )
    
    # Post-processing
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Chain
    rag_chain = prompt | llm | StrOutputParser()

    # Run
    response = rag_chain.invoke({"context": docs, "question": question, "response_type": response_type})
    
    # 明确返回一条AIMessage
    return {"messages": [AIMessage(content=response)]}

@traceable
def build_custom_agent(llm, retriever, response_type, LC_SYSTEM_PROMPT):
    """
    构建自定义Agent工作流
    
    创建一个基于LangGraph的智能问答工作流，包含查询分类、文档检索、相关性评估、
    查询重写和答案生成等节点
    
    参数:
        llm: 语言模型实例
        retriever: 文档检索器
        response_type: 响应类型
        LC_SYSTEM_PROMPT: 系统提示词
    
    返回:
        compiled_workflow: 编译后的工作流实例
    
    工作流节点:
        - agent: 决策节点，决定是否使用工具
        - retrieve: 文档检索节点
        - rewrite: 查询重写节点
        - generate: 答案生成节点
    
    工作流边:
        - START -> agent: 从agent节点开始
        - agent -> retrieve/END: 根据决策决定检索或结束
        - retrieve -> generate/rewrite: 根据相关性决定生成或重写
        - generate -> END: 生成完成后结束
        - rewrite -> agent: 重写后重新决策
    """
    # 创建检索工具
    retriever_tool = create_retriever_tool(
        retriever,
        "medical_knowledge_retriever",
        "检索医疗知识图谱中的疾病、症状、治疗等相关信息。",
    )
    
    # 工具列表
    global tools
    tools = [retriever_tool]
    
    # 管理对话历史
    global memory
    memory = MemorySaver()

    # Define a new graph
    workflow = StateGraph(AgentState)

    # Define the nodes we will cycle between
    workflow.add_node("agent", lambda state: agent(state, llm, tools))  # agent
    retrieve = ToolNode([retriever_tool])
    workflow.add_node("retrieve", retrieve)  # retrieval
    workflow.add_node("rewrite", lambda state: rewrite(state, llm))  # Re-writing the question
    workflow.add_node(
        "generate", lambda state: generate(state, llm, LC_SYSTEM_PROMPT, response_type)
    )  # Generating a response after we know the documents are relevant
    
    # Call agent node to decide to retrieve or not
    workflow.add_edge(START, "agent")

    # Decide whether to retrieve
    workflow.add_conditional_edges(
        "agent",
        # Assess agent decision
        tools_condition,
        {
            # Translate the condition outputs to nodes in our graph
            "tools": "retrieve",
            END: END,
        },
    )

    # Edges taken after the `action` node is called.
    workflow.add_conditional_edges(
        "retrieve",
        # Assess agent decision
        lambda state: grade_documents(state, llm),
    )
    workflow.add_edge("generate", END)
    workflow.add_edge("rewrite", "agent")

    # Compile
    return workflow.compile(checkpointer=memory)

def ask_agent(query, agent, config):
    """
    询问Agent并显示执行过程
    
    参数:
        query (str): 用户查询
        agent: 工作流实例
        config: 配置参数
    
    功能:
        1. 创建查询输入
        2. 流式执行工作流
        3. 打印每个节点的输出
    """
    inputs = {"messages": [HumanMessage(content=query)]}
    for output in agent.stream(inputs, config=config):
        for key, value in output.items():
            pprint.pprint(f"Output from node '{key}':")
            pprint.pprint("---")
            pprint.pprint(value, indent=2, width=80, depth=None)
        pprint.pprint("\n---\n")

def get_answer(config):
    """
    获取最终答案
    
    参数:
        config: 配置参数
    
    返回:
        str: 最终生成的答案
    
    功能:
        从内存中获取对话历史，提取最后一条消息作为答案
    """
    chat_history = memory.get(config)["channel_values"]["messages"]
    answer = chat_history[-1].content
    return answer

if __name__ == '__main__':
    """
    主程序入口
    
    初始化系统组件，构建工作流，启动交互式问答循环
    """

    graph = MyNeo4jGraph(
        url=NEO4J_URI, 
        username=NEO4J_USERNAME, 
        password=NEO4J_PASSWORD,
        enhanced_schema=True
    )
    print("数据库成功连接")
    graph.refresh_schema()

    llm = ChatDeepSeek(model=INSTRUCT_MODEL, temperature=0)
    index_name = "vector"

    topChunks = 3          # 检索的文本块数量
    topCommunities = 3     # 检索的社区数量
    topOutsideRels = 10    # 检索的外部关系数量
    topInsideRels = 10     # 检索的内部关系数量
    topEntities = 10       # 检索的实体数量

    lc_retrieval_query = f"""
    WITH collect(node) as nodes
    WITH
    collect {{
        UNWIND nodes as n
        MATCH (n)<-[:MENTIONS]-(c:__Chunk__)
        WITH distinct c, count(distinct n) as freq
        RETURN {{id:c.id, text: c.text}} AS chunkText
        ORDER BY freq DESC
        LIMIT {topChunks}
    }} AS text_mapping,
    collect {{
        UNWIND nodes as n
        MATCH (n)-[:IN_COMMUNITY]->(c:__Community__)
        WITH distinct c, c.community_rank as rank, c.weight AS weight
        RETURN c.summary 
        ORDER BY rank, weight DESC
        LIMIT {topCommunities}
    }} AS report_mapping,
    collect {{
        UNWIND nodes as n
        MATCH (n)-[r]-(m:__Entity__) 
        WHERE NOT m IN nodes
        RETURN r.description AS descriptionText
        ORDER BY r.weight DESC 
        LIMIT {topOutsideRels}
    }} as outsideRels,
    collect {{
        UNWIND nodes as n
        MATCH (n)-[r]-(m:__Entity__) 
        WHERE m IN nodes
        RETURN r.description AS descriptionText
        ORDER BY r.weight DESC 
        LIMIT {topInsideRels}
    }} as insideRels,
    collect {{
        UNWIND nodes as n
        RETURN n.description AS descriptionText
    }} as entities
    RETURN {{Chunks: text_mapping, Reports: report_mapping, 
           Relationships: outsideRels + insideRels, 
           Entities: entities}} AS text, 1.0 AS score, {{}} AS metadata
    """

    # 初始化HuggingFace嵌入模型
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",  # 使用BGE-M3多语言嵌入模型
        cache_folder="./model",    # 模型缓存目录
    )

    # 从现有索引创建Neo4j向量存储
    lc_vector = Neo4jVector.from_existing_index(
        embeddings,
        url=NEO4J_URI,
        username=NEO4J_USERNAME,
        password=NEO4J_PASSWORD,
        index_name=index_name,
        retrieval_query=lc_retrieval_query,  # 使用自定义检索查询
    )
    
    # 创建检索器
    retriever = lc_vector.as_retriever(search_kwargs={"k": topEntities})

    response_type = "多个段落"

    # 系统提示词：定义AI助手的角色、任务和回答要求
    LC_SYSTEM_PROMPT = """
    ---角色--- 
    您是一个有用的助手，请根据用户输入的上下文，综合上下文中多个分析报告的数据，来回答问题，并遵守回答要求。

    ---任务描述--- 
    总结来自多个不同分析报告的数据，生成要求长度和格式的回复，以回答用户的问题。 

    ---回答要求---
    - 你要严格根据分析报告的内容回答，禁止根据常识和已知信息回答问题。
    - 对于不知道的问题，直接回答"不知道"。
    - 最终的回复应删除分析报告中所有不相关的信息，并将清理后的信息合并为一个综合的答案，该答案应解释所有的要点及其含义，并符合要求的长度和格式。 
    - 根据要求的长度和格式，把回复划分为适当的章节和段落，并用markdown语法标记回复的样式。 
    - 回复应保留之前包含在分析报告中的所有数据引用，但不要提及各个分析报告在分析过程中的作用。 
    - 如果回复引用了Entities、Reports及Relationships类型分析报告中的数据，则用它们的顺序号作为ID。
    - 如果回复引用了Chunks类型分析报告中的数据，则用原始数据的id作为ID。 
    - 不要在一个引用中列出超过5个引用记录的ID，相反，列出前5个最相关的引用记录ID。 
    - 不要包括没有提供支持证据的信息。
    """

    # 构建自定义Agent工作流
    custom_agent = build_custom_agent(llm, retriever, response_type, LC_SYSTEM_PROMPT)
    
    # 配置工作流参数，限制重写次数避免无限循环
    config = {"configurable": {"thread_id": "medical_agent", "recursion_limit": 5}}

    # 启动交互式问答循环
    while True:
        questions = input("\n请输入问题：")
        if questions == "exit":
            break

        print(f"\n=== 处理问题: {questions} ===")
        ask_agent(questions, custom_agent, config)
        final_answer = get_answer(config)
        print(f"\n=== 最终答案 ===\n{final_answer}")

    # 关闭数据库连接
    graph.close()