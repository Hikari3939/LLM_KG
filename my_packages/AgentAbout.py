import os
import pprint
from typing import Literal
from dotenv import load_dotenv
from langchain_core.tools import tool
from langgraph.graph import MessagesState
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import PromptTemplate
from langgraph.graph import END, StateGraph, START
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

from my_packages.QueryAbout import local_retriever, global_retriever

# 环境变量配置
load_dotenv(".env")
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# LLM配置
llm = ChatDeepSeek(model='deepseek-chat')

# LLM响应类型配置
response_type: str = "多个段落"


### Tools
# 创建局部检索器工具
@tool
def local_retriever_tool(query: str) -> str:
    """检索以脑卒中为核心，涵盖和脑卒中相关的药物、病症等知识图谱中的具体信息，
    适用于查找具体的实体、关系、属性等。"""
    return local_retriever(query)

# 创建全局检索器工具
@tool
def global_retriever_tool(query: str) -> str:
    """回答有关以脑卒中为核心，涵盖和脑卒中相关的药物、病症等知识图谱的全局性、综合性问题，
    通过分析多个社区摘要来提供全面答案。适用于综合分析、整体概述等宏观问题。"""
    return global_retriever(query)

tools = [local_retriever_tool, global_retriever_tool]


### Nodes
def query_or_response(state):
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
                3. **保持原意**：确保重构后的问题严格保持用户的原始意图
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
                
                {context}
                

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
        {{'communityIds': [逗号分隔的摘要ID列表]}}
        例如：
        {{'communityIds':['0-0','0-1']}}
        {{'communityIds':['0-0', '0-1', '0-3']}}
        其中'0-0'、'0-1'、'0-3'是摘要来源的communityId。
        - 摘要引用的说明放在引用之后，不要单独作为一段。
        例如： 
        #############################
        "X是Y公司的所有者，他也是X公司的首席执行官{{'communityIds':['0-0']}}，
        受到许多不法行为指控{{'communityIds':['0-0', '0-1', '0-3']}}。"  
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
    if retrieve_message.additional_kwargs["tool_calls"][0]["function"]["name"]== 'global_retriever_tool':
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
def create_workflow():
    '''创建工作流'''
    workflow = StateGraph(MessagesState)

    # 定义节点
    # query_or_response
    workflow.add_node("query_or_response", query_or_response)
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
    # Start from the "query_or_response" node
    workflow.add_edge(START, "query_or_response")
    # Decide whether to retrieve
    workflow.add_conditional_edges(
        "query_or_response",
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
    # 如果是重构问题，转到query_or_response结点重新开始。
    workflow.add_edge("rewrite", "query_or_response")
    # 如果是局部查询或全局查询生成，直接结束
    workflow.add_edge("generate", END)
    workflow.add_edge("reduce", END)
    
    return workflow


### Run
def create_agent(memory: InMemorySaver):
    '''创建agent实例'''
    agent = create_workflow().compile(checkpointer=memory)
    return agent

def user_config(session_id='3939', recursion_limit=3):
    """创建用户的session配置，以session_id作为thread_id"""
    config = {"configurable": {"thread_id": session_id, "recursion_limit": recursion_limit}}
    return config

def ask_agent(query, agent, config):
    '''向agent发送消息'''
    inputs = {"messages": [("user", query)]}
    for output in agent.stream(inputs, config=config):
        for key, value in output.items():
            print(f"---Output from node '{key}'---")
            pprint.pprint(value, indent=2, width=80, depth=None)
        print("")

def get_answer(memory: InMemorySaver, config):
    '''获取agent的回答'''
    result = memory.get(config)
    chat_history = result["channel_values"]["messages"]
    answer = chat_history[-1].content
    return answer
