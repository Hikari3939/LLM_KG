from my_packages.QueryAbout import llm, response_type, NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, RemoveMessage
from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated, Literal, Sequence, TypedDict, Dict, Any
import pprint
import pandas as pd
import re
from neo4j import GraphDatabase, Result

# 使用QueryAbout中已有的检索器和提示词
from my_packages.QueryAbout import local_retriever, global_retriever, response_type, REDUCE_SYSTEM_PROMPT
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 溯源查验相关配置
recursion_limit = 5

# 创建Neo4j数据库连接
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD))

# 执行Cypher查询
def db_query(cypher: str, params: Dict[str, Any] = {}) -> pd.DataFrame:
    """执行Cypher语句并返回DataFrame"""
    return driver.execute_query(
        cypher, parameters_=params, result_transformer_=Result.to_df
    )

# 创建局部检索器工具
@tool
def local_retriever_tool(query: str) -> str:
    """检索知识图谱中的具体信息，适用于查找特定实体、关系、属性等详细信息。当用户询问具体的人物、事件、概念的具体细节、定义、分类、特征、属性、关系等时使用此工具。例如：某个疾病的具体分类、某个概念的定义、某个实体的属性等。"""
    return local_retriever(query)

# 创建全局检索器工具（MAP阶段）
@tool
def global_retriever_tool(query: str) -> str:
    """回答有关知识图谱的全局性、综合性问题，通过分析多个社区摘要来提供全面答案。适用于需要综合分析、总结性回答、趋势分析、比较分析、整体概述等宏观问题。例如：整体趋势分析、多个概念的比较、知识图谱的全局概况等。"""
    return global_retriever(query, level=0)

# 溯源查验工具
@tool
def get_source_tool(sourcedId: str) -> str:
    """根据给定的ID查看文本块或社区，输出成HTML文本。用于溯源查验功能。"""
    return get_source(sourcedId)

# 工具列表包含局部检索器、全局检索器和溯源查验工具
tools = [local_retriever_tool, global_retriever_tool, get_source_tool]

# Agent state ------------------------------------------------------------------
# 调试时如果要查看每个节点输入输出的状态，可以用这个函数插入打印的语句
def my_add_messages(left,right):
    print("\nLeft:\n")
    print(left)
    print("\nRight\n")
    print(right)
    return add_messages(left,right)
    
class AgentState(TypedDict):
    # The add_messages function defines how an update should be processed
    # Default is to replace. add_messages says "append"
    # messages: Annotated[Sequence[BaseMessage], my_add_messages]
    messages: Annotated[Sequence[BaseMessage], add_messages]

# Nodes and Edges --------------------------------------------------------------

### Edges

def grade_documents(state) -> Literal["generate", "rewrite", "reduce"]:
    """
    分流边：对工具调用的结果进行分流处理。
    如果是全局检索，转到reduce结点生成回复。
    如果是局部检索并且检索结果与问题相关，转到generate结点生成回复。
    如果是局部检索并且检索结果与问题不相关，转到rewrite结点重构问题。

    Args:
        state (messages): The current state

    Returns:
        str: A decision for which node to go to next
    """

    messages = state["messages"]
    # 倒数第2条消息是LLM发出工具调用请求的AIMessage。
    retrieve_message = messages[-2]
    
    # 如果是全局查询直接转去reduce结点。
    if retrieve_message.additional_kwargs["tool_calls"][0]["function"]["name"] == 'global_retriever_tool':
        print("---Global retrieve---")
        return "reduce"

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

### Nodes

def agent(state):
    """
    Invokes the agent model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply end.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with the agent response appended to messages
    """
    print("---CALL AGENT---")
    messages = state["messages"]
    model = llm

    model = model.bind_tools(tools)
    response = model.invoke(messages)
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}


def rewrite(state):
    """
    Transform the query to produce a better question.

    Args:
        state (messages): The current state

    Returns:
        dict: The updated state with re-phrased question
    """

    print("---TRANSFORM QUERY---")
    messages = state["messages"]
    # 在对话历史中，问题消息是倒数第3条。
    question = messages[-3].content

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


def reduce(state):
    """
    Generate answer for global retrieve - REDUCE stage

    Args:
        state (messages): The current state

    Returns:
         dict: The updated state with final answer
    """
    print("---REDUCE---")
    messages = state["messages"]
    
    # 在对话历史中，问题消息是倒数第3条。
    question = messages[-3].content
    # 检索结果在最后一条消息中（这是global_retriever_tool返回的MAP阶段结果）
    last_message = messages[-1]
    map_results = last_message.content

    # Reduce阶段生成最终结果的prompt与chain
    reduce_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                REDUCE_SYSTEM_PROMPT,
            ),
            (
                "human",
                """
            ---分析报告--- 
            {report_data}

            用户的问题是：
            {question}
                """,
            ),
        ]
    )
    reduce_chain = reduce_prompt | llm | StrOutputParser()

    # 再用LLM从每个社区摘要生成的中间结果生成最终的答复
    response = reduce_chain.invoke(
        {
            "report_data": map_results,
            "question": question,
            "response_type": response_type,
        }
    )
    # 明确返回一条AIMessage。
    return {"messages": [AIMessage(content = response)]}

def generate(state):
    """
    Generate answer

    Args:
        state (messages): The current state

    Returns:
         dict: The updated state with re-phrased question
    """
    print("---GENERATE---")
    messages = state["messages"]
    # 在对话历史中，问题消息是倒数第3条。
    question = messages[-3].content
    
    last_message = messages[-1]

    docs = last_message.content

    # 直接使用QueryAbout中的local_retriever函数
    response = local_retriever(question)
    
    # 这里有个Bug，response是String，generate节点返回的消息会自动判定为HumanMessage，其实是AIMessage。
    # 明确返回一条AIMessage。
    return {"messages": [AIMessage(content = response)]}
  
# Graph ------------------------------------------------------------------------
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# 管理对话历史
memory = MemorySaver()

# Define a new graph
workflow = StateGraph(AgentState)

# Define the nodes we will cycle between
workflow.add_node("agent", agent)  # agent
retrieve = ToolNode(tools)
workflow.add_node("retrieve", retrieve)  # retrieval
workflow.add_node("rewrite", rewrite)  # Re-writing the question
workflow.add_node(
    "generate", generate
)  # Generating a response after we know the documents are relevant
# Call agent node to decide to retrieve or not
# 增加一个全局查询的reduce结点
workflow.add_node(
    "reduce", reduce
)

# 定义结点之间的连接
# 从agent结点开始
workflow.add_edge(START, "agent")

# Decide whether to retrieve
workflow.add_conditional_edges(
    "agent",
    # Assess agent decision
    tools_condition,          # tools_condition()的输出是"tools"或END
    {
        # Translate the condition outputs to nodes in our graph
        "tools": "retrieve",  # 转到retrieve结点，执行局部检索或全局检索
        END: END,             # 直接结束
    },
)

# 检索结点执行结束后调边grade_documents，决定流转到哪个结点: generate、rewrite、reduce。
workflow.add_conditional_edges(
    "retrieve",
    # Assess agent decision
    grade_documents,
)
# 如果是局部查询生成，直接结束
workflow.add_edge("generate", END)
# 如果是重构问题，转到agent结点重新开始。
workflow.add_edge("rewrite", "agent")
# 增加一条全局查询到结束的边
workflow.add_edge("reduce", END)

# Compile
# Finally, we compile it!
# This compiles it into a LangChain Runnable,
# meaning you can use it as you would any other runnable
agent = workflow.compile(checkpointer=memory)

# Run --------------------------------------------------------------------------
# 限制rewrite的次数，以免陷入无限的循环
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
def get_source(sourcedId):
    """根据给定的ID查看文本块或社区，输出成HTML文本。"""
    ChunkCypher = """
    match (n:`__Chunk__`) where n.id=$id return n.fileName, n.text
    """
    CommunityCypher = """
    match (n:`__Community__`) where n.id=$id return n.summary, n.full_content    
    """
    
    temp = sourcedId.split(",")
    if temp[0] == '2':
        result = db_query(ChunkCypher, params={"id": temp[-1]})
    else:
        result = db_query(CommunityCypher, params={"id": temp[-1]})
    
    if result.shape[0] > 0:
        resp = result.iloc[0, 0] + "\n\n" + result.iloc[0, 1]
        resp = resp.replace("\n", "<br/>")
    else:
        resp = "在知识图谱中没有检索到该语料。"
    return resp

def user_config(sessionId):
    """用户session对象，以session ID为thread_id"""
    config = {"configurable": {"thread_id": sessionId, "recursion_limit": recursion_limit}}
    return config

def format_messages(messages):
    """格式化输出成HTML文本"""
    resp = ""
    for message in messages:
        # 函数调用的AIMessage和ToolMessage不输出显示
        if isinstance(message, AIMessage) and len(message.content) > 0:
            resp = resp + "AI: " + message.content + "\n\n"
        if isinstance(message, HumanMessage):
            resp = resp + "User: " + message.content + "\n"
    resp = resp.replace("\n", "<br/>")
    return resp

def add_links_to_text(text):
    """为数据引用加上溯源的链接"""
    # 定义正则表达式模式
    points_pattern = re.compile(r"\{'points':\[(.*?)\]\}")
    chunks_pattern = re.compile(r"'Chunks':\[(.*?)\]")
    
    # 替换points中的id为链接
    def replace_points(match):
        points = match.group(1)
        points_with_links = re.sub(r"\((\d+),'([0-9a-fA-F-]+)'\)", 
                                  lambda m: f"<a href='javascript:showInfo(\"1,{m.group(1)},{m.group(2)}\")'>({m.group(1)},'{m.group(2)}')</a>", 
                                  points)
        return f"{{'points':[{points_with_links}]}}"
    
    # 替换chunks中的id为链接
    def replace_chunks(match):
        chunks = match.group(1)
        chunk_ids = re.findall(r"'([^']+)'", chunks)
        for chunk_id in chunk_ids:
            chunks = chunks.replace(f"'{chunk_id}'", f"<a href='javascript:showInfo(\"2,{chunk_id}\")'>'{chunk_id}'</a>")
        return f"'Chunks':[{chunks}]"
    
    # 替换文本中的points和chunks
    result = points_pattern.sub(replace_points, text)
    result = chunks_pattern.sub(replace_chunks, result)
    
    return result

def ask_agent_with_source(query, sessionId):
    """执行一轮对话，包含溯源查验功能"""
    config = user_config(sessionId)
    inputs = {"messages": [("user", query)]}
    try:
        agent.invoke(inputs, config=config)
        messages = agent.get_state(config).values["messages"]
        # 对话记录格式化输出成文本
        response = format_messages(messages)
        # 为数据引用加上溯源的链接
        response = add_links_to_text(response)
    except Exception as e:
        response = repr(e)
    return response

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
