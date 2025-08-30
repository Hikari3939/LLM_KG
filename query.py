import os
from dotenv import load_dotenv
from langchain_neo4j import Neo4jGraph, GraphCypherQAChain
from langchain_deepseek import ChatDeepSeek

# 加载环境变量
load_dotenv(".env")
NEO4J_URI = os.environ["NEO4J_URI"]
NEO4J_USERNAME = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]

DEEPSEEK_API_KEY = os.environ["DEEPSEEK_API_KEY"]

# 指定模型名称
INSTRUCT_MODEL = 'deepseek-chat'

if __name__ == '__main__':
    # 连接neo4j数据库
    graph = Neo4jGraph(
        url=NEO4J_URI, 
        username=NEO4J_USERNAME, 
        password=NEO4J_PASSWORD,
        enhanced_schema=True
    )
    print("数据库成功连接")

    # 刷新数据库结构信息
    graph.refresh_schema()

    # 连接大模型
    llm = ChatDeepSeek(
        model = INSTRUCT_MODEL,
        temperature=0
    )
    
    #构建查询工具链
    chain = GraphCypherQAChain.from_llm(
        graph=graph, llm=llm, verbose=True, allow_dangerous_requests=True
    )

    while True:
        questions = input("\n请输入问题：")
        if questions!="exit":
            answer = chain.invoke({"query":questions})
            print(answer['result'])
        else:
            break

    # 关闭数据库
    graph.close()