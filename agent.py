import os
import pprint
import pandas as pd
from dotenv import load_dotenv
from langchain_core.tools import tool
from typing import Literal, Dict, Any
from neo4j import GraphDatabase, Result
from langgraph.graph import MessagesState
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import PromptTemplate
from langgraph.graph import END, StateGraph, START
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, RemoveMessage

from my_packages.AgentTool import local_retriever, global_retriever


# 环境变量配置
load_dotenv(".env")

NEO4J_URI = os.environ.get("NEO4J_URI")
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD")

os.environ["LANGCHAIN_TRACING_V2"] = "true"
LANGCHAIN_API_KEY = os.environ.get("LANGCHAIN_API_KEY")

# LLM配置
llm = ChatDeepSeek(
    model='deepseek-chat'
)

# LLM响应类型配置
response_type: str = "多个段落"

# 创建Neo4j数据库连接
driver = GraphDatabase.driver()


### Tools
# 创建局部检索器工具
@tool
def local_retriever_tool(query: str) -> str:
    """检索知识图谱中的具体信息，适用于查找特定实体、关系、属性等详细信息。
    当用户询问具体的人物、事件、概念的具体细节、定义、分类、特征、属性、关系等时使用此工具。
    例如：某个疾病的具体分类、某个概念的定义、某个实体的属性等。"""
    return local_retriever(query)

# 创建全局检索器工具
@tool
def global_retriever_tool(query: str) -> str:
    """回答有关知识图谱的全局性、综合性问题，通过分析多个社区摘要来提供全面答案。
    适用于需要综合分析、总结性回答、趋势分析、比较分析、整体概述等宏观问题。
    例如：整体趋势分析、多个概念的比较、知识图谱的全局概况等。"""
    return global_retriever(query)

# 工具列表
tools = [local_retriever_tool, global_retriever_tool]


### Nodes
def agent(state):
    """
    Invokes the agent model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply end.
    """
    print("---CALL AGENT---")
    messages = state["messages"]
    model = llm.bind_tools(tools)
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}

def rewrite(state):
    """
    Transform the query to produce a better question.
    """
    print("---TRANSFORM QUERY---")
    messages = state["messages"]
    # 在对话历史中，问题消息是倒数第3条。
    question = messages[-3].content
    msg = [
        HumanMessage(
            content=f"""
                你是一个专业的查询重构助手，专门优化知识图谱查询问题。

                ## 任务描述
                分析用户的问题，理解其真实意图，并重构为一个更清晰、更具体、更适合知识图谱检索的问题。

                ## 原始问题
                {question}

                ## 重构原则
                1. **明确实体和关系**：识别问题中涉及的具体实体、概念、关系
                2. **具体化描述**：将抽象概念转化为具体的、可检索的描述
                3. **保持原意**：确保重构后的问题保持用户的原始意图
                4. **严禁发散**：若因关键信息缺失等原因无法进行重构，请直接返回原始问题，不要基于常识或猜测添加任何内容

                ## 重构策略
                - 如果问题太宽泛，添加限定条件或具体领域
                - 如果问题太简单，扩展为更详细的描述
                - 如果问题包含模糊词汇，替换为更精确的术语

                ## 输出要求
                只输出重构后的问题，不能重构时输出原始问题，不要添加任何解释或额外内容。

                输出问题：
                """
        )
    ]
    # Rewriter
    response = llm.invoke(msg)
    return {"messages": [response]}

def generate(state):
    """
    Generate answer
    """
    print("---GENERATE---")
    messages = state["messages"]
    # 在对话历史中，问题消息是倒数第3条。
    question = messages[-3].content
    # 在对话历史中，检索结果是最后一条。
    docs = messages[-1].content

    # 局部查询的提示词
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
    response = lc_chain.invoke(
        {"context": docs, "question": question, "response_type":response_type}
    )
    
    # 明确返回一条AIMessage。
    return {"messages": [AIMessage(content = response)]}

def reduce(state):
    """
    Generate answer for global retrieve
    """
    print("---REDUCE---")
    messages = state["messages"]
    
    # 在对话历史中，问题消息是倒数第3条。
    question = messages[-3].content
    # 在对话历史中，检索结果是最后一条。
    last_message = messages[-1]
    docs = last_message.content

    # Reduce阶段生成最终结果的prompt与chain
    reduce_system_prompt = """
        ---角色--- 
        你是一个有用的助手，请根据用户输入的上下文，综合上下文中多个社区摘要的数据，来回答问题，并遵守回答要求。

        ---任务描述--- 
        总结来自多个不同社区摘要的数据，生成要求长度和格式的回复，以回答用户的问题。 

        ---回答要求---
        - 你要严格根据社区摘要的内容回答，禁止根据常识和已知信息回答问题。
        - 对于不知道的信息，直接回答"不知道"。
        - 最终的回复应删除所有不相关的信息，并将清理后的信息合并为一个综合的答案，该答案应解释所有选用的摘要及其含义，并符合要求的长度和格式。 
        - 根据要求的长度和格式，把回复划分为适当的章节和段落，并用markdown语法标记回复的样式。 
        - 回复应保留摘要引用，并且包含引用来源社区的原始communityId，但不要提及各个摘要在分析过程中的作用。 
        - **不要在一个引用中列出超过5个摘要引用的ID**，相反，列出前5个最相关摘要引用的顺序号作为ID。 
        - 不要包括没有提供支持证据的信息。
        
        ---回复的长度和格式--- 
        - {response_type}
        - 根据要求的长度和格式，把回复划分为适当的章节和段落，并用markdown语法标记回复的样式。  
        - 输出摘要引用的格式：
        {{'points': [逗号分隔的摘要ID列表]}}
        例如：
        {{'points':['0-0','0-1']}}
        {{'points':['0-0', '0-1', '0-3']}}
        其中'0-0'、'0-1'、'0-3'是摘要来源的communityId。
        - 摘要引用的说明放在引用之后，不要单独作为一段。
        例如： 
        #############################
        "X是Y公司的所有者，他也是X公司的首席执行官{{'points':['0-0']}}，
        受到许多不法行为指控{{'points':['0-0', '0-1', '0-3']}}。"  
        #############################
        """
    reduce_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                reduce_system_prompt,
            ),
            (
                "human",
                """
            ---相关社区摘要--- 
            {report_data}

            用户的问题是：
            {question}
                """,
            ),
        ]
    )
    reduce_chain = reduce_prompt | llm | StrOutputParser()

    # 使用LLM根据筛选出的社区摘要生成最终答复
    response = reduce_chain.invoke(
        {
            "report_data": docs,
            "question": question,
            "response_type": response_type,
        }
    )
    # 明确返回一条AIMessage。
    return {"messages": [AIMessage(content = response)]}


### Edges
def grade_documents(state) -> Literal["generate", "rewrite", "reduce"]:
    """
    根据调用的工具和结果进行分流处理。
    如果是全局检索，转到reduce结点生成回复。
    如果是局部检索并且检索结果与问题相关，转到generate结点生成回复。
    如果是局部检索并且检索结果与问题不相关，转到rewrite结点重构问题。
    局部检索的结果是否与问题相关，提交给LLM去判断。
    """
    messages = state["messages"]
    # 倒数第2条消息是LLM发出工具调用请求的AIMessage。
    retrieve_message = messages[-2]
    
    # 如果是全局查询，直接转到reduce结点。
    if retrieve_message.additional_kwargs["tool_calls"][0]["function"]["name"]== 'global_retriever':
        print("---GLOBAL RETRIEVE---")
        return "reduce"

    print("---CHECK RELEVANCE---")
    # 判断结果是否与问题相关
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
        input_variables=["context", "question"],
    )
    chain = prompt | llm
    
    # 最后一条消息是检索器返回的结果。
    last_message = messages[-1]
    # 在对话历史中，问题消息是倒数第3条。
    question = messages[-3].content
    
    docs = last_message.content

    scored_result = chain.invoke({"question": question, "context": docs})
    # LLM会给出检索结果与问题是否相关的判断, yes或no
    score = scored_result.content
    # 保险起见要转为小写！！！
    if score.lower() == "yes":
        print("---DECISION: DOCS RELEVANT---")
        print(score)
        return "generate"

    else:
        print("---DECISION: DOCS NOT RELEVANT---")
        print(score)
        return "rewrite"


### Graph
# 管理对话历史
memory = MemorySaver()

# 定义图
workflow = StateGraph(MessagesState)

# 定义节点
# agent
workflow.add_node("agent", agent)
# retrieval
retrieve = ToolNode(tools)
workflow.add_node("retrieve", retrieve)
# Re-writing the question
workflow.add_node("rewrite", rewrite)
# Generating a response after we know the documents are relevant
workflow.add_node("generate", generate)
# 全局查询的reduce结点
workflow.add_node("reduce", reduce)

# 定义边
# Start from the "agent" node
workflow.add_edge(START, "agent")
# Decide whether to retrieve
workflow.add_conditional_edges(
    "agent",
    # Assess agent decision
    tools_condition,          # tools_condition()的输出是"tools"或END
    {
        "tools": "retrieve",  # 转到retrieve结点，执行局部检索或全局检索
        END: END,             # 直接结束
    },
)
# grade_documents决定流转到哪个结点
workflow.add_conditional_edges(
    "retrieve",
    # Assess agent decision
    grade_documents,
)
# 如果是重构问题，转到agent结点重新开始。
workflow.add_edge("rewrite", "agent")
# 如果是局部查询或全局查询生成，直接结束
workflow.add_edge("generate", END)
workflow.add_edge("reduce", END)

# 编译工作流
agent = workflow.compile(checkpointer=memory)


# Run
# 限制rewrite的次数，避免陷入无限循环
config = {"configurable": {"thread_id": "226", "recursion_limit":5}}

def ask_agent(query,agent, config):
    inputs = {"messages": [("user", query)]}
    for output in agent.stream(inputs, config=config):
        for key, value in output.items():
            pprint.pprint(f"Output from node '{key}':")
            pprint.pprint("---")
            pprint.pprint(value, indent=2, width=80, depth=None)
        pprint.pprint("\n---\n")

def get_answer(config):
    chat_history = memory.get(config)["channel_values"]["messages"]
    answer = chat_history[-1].content
    return answer

# 溯源查验功能函数 ----------------------------------------------------------------
# 溯源查验相关配置
recursion_limit = 5

# 溯源查验工具
@tool
def get_source_tool(sourcedId: str) -> str:
    """根据给定的ID查看文本块或社区，输出成HTML文本。用于溯源查验功能。"""
    return get_source(sourcedId)

# 执行Cypher查询
def db_query(cypher: str, params: Dict[str, Any] = {}) -> pd.DataFrame:
    """执行Cypher语句并返回DataFrame"""
    return driver.execute_query(
        cypher, parameters_=params, result_transformer_=Result.to_df
    )

def get_source(sourcedId):
    """根据给定的ID查看文本块或社区，输出成HTML文本。"""
    ChunkCypher = """
    match (n:`__Chunk__`) where n.id=$id return n.fileName, n.text
    """
    CommunityCypher = """
    match (n:`__Community__`) where n.id=$id return n.summary, n.id    
    """
    
    temp = sourcedId.split(",")
    if temp[0] == '2':
        result = db_query(ChunkCypher, params={"id": temp[-1]})
    else:
        result = db_query(CommunityCypher, params={"id": temp[-1]})
    
    if result.shape[0] > 0:
        if temp[0] == '2':  # Chunk
            resp = result.iloc[0, 0] + "\n\n" + result.iloc[0, 1]
        else:  # Community
            resp = "社区摘要:\n" + result.iloc[0, 0] + "\n\n社区ID: " + result.iloc[0, 1]
        resp = resp.replace("\n", "<br/>")
    else:
        resp = "在知识图谱中没有检索到该语料。"
    return resp

def user_config(sessionId):
    """用户session对象，以session ID为thread_id"""
    config = {"configurable": {"thread_id": sessionId, "recursion_limit": recursion_limit}}
    return config

def format_messages(messages):
    """格式化输出成文本，只返回AI回复"""
    resp = ""
    for message in messages:
        # 只输出AIMessage的内容，不添加前缀
        if isinstance(message, AIMessage) and len(message.content) > 0:
            resp = resp + message.content
    return resp

def add_links_to_text(text):
    """保持原始引用格式，用于溯源查询"""
    # 保持原始的引用格式，不转换为HTML链接
    return text

def ask_agent_with_source(query, sessionId):
    """执行一轮对话，包含溯源查验功能"""
    config = user_config(sessionId)
    
    # 创建新的用户消息
    new_message = HumanMessage(content=query)
    
    try:
        # 使用stream方式获取详细的调试信息，传入完整的消息历史
        for output in agent.stream({"messages": [new_message]}, config=config):
            for key, value in output.items():
                pprint.pprint(f"Output from node '{key}':")
                pprint.pprint("---")
                pprint.pprint(value, indent=2, width=80, depth=None)
            pprint.pprint("\n---\n")
        
        messages = agent.get_state(config).values["messages"]
        # 只获取最后一条AIMessage的内容，保持引用格式用于溯源
        for message in reversed(messages):
            if isinstance(message, AIMessage) and len(message.content) > 0:
                return message.content
        return "抱歉，没有获取到有效回复。"
    except Exception as e:
        return f"抱歉，查询过程中出现错误：{str(e)}"

def clear_session(sessionId):
    """清除对话历史"""
    config = user_config(sessionId)
    try:
        messages = agent.get_state(config).values["messages"]
        # 要后入先删的次序到过来删，否则会抛Exception。
        i = len(messages)
        for message in reversed(messages):
            # 如果倒数第3条消息是ToolMessage，保留前面4条信息，它是一轮完整的对话。
            if i == 4 and isinstance(messages[2], ToolMessage):
                break
            agent.update_state(config, {"messages": RemoveMessage(id=message.id)})
            i = i - 1
            # 保留第1轮对话。
            if i == 2:
                break
        messages = agent.get_state(config).values["messages"]
        return format_messages(messages)
    except Exception as e:
        print(e)
        return str(e)

def get_messages(sessionId):
    """获取对话的上下文"""
    config = user_config(sessionId)
    messages = agent.get_state(config).values["messages"]
    return messages
