# 项目说明

## 文件结构

- data      `原始数据`
- create.py     `知识图谱构建模块`
- loader.py     `文档分割模块`
- query.py      `知识图谱问答模块`

> 需要单独建立.env文件，内容如下：

    # 数据库配置
    NEO4J_URI="Your neo4j URI"
    NEO4J_USERNAME="Your username"
    NEO4J_PASSWORD="Your password"

    # api-key
    DEEPSEEK_API_KEY = "Your deepseek api-key"


## 构建部分

首先，我们使用Langchain工具链所提供的Neo4jGraph模块连接neo4j数据库并清空数据库内容；之后，使用TextLoader和PyMuPDFLoader加载文件，并使用RecursiveCharacterTextSplitter模块进行分割；接下来，使用ChatDeepSeek模块与大模型建立连接；最后，分批次使用LLMGraphTransformer模块将分割后的文本块转换为图文档并存入知识图谱中。

## 问答部分

首先仍然使用Neo4jGraph模块连接数据库并更新数据库结构，接下来连接DeepSeek大模型，利用GraphCypherQAChain模块构建工具链，使用LLM将问题转化为Cypher查询语言，并根据查询结果输出问题的答案。

## 评价

一坨，别看