# 项目说明

## 文件结构

- data                `原始数据`
  - example.txt         `数据文件`
- my_packages         `项目模块`
  - DatabaseAbout.py    `数据库操作模块`
  - DataLoader.py       `原始数据加载模块`
  - MyNeo4j.py          `重载的Neo4jGraph类`
- create.py           `知识图谱初步构建`
- process.py          `知识图谱完善`
- query.py            `知识图谱查询`


## 问答部分

首先使用重载后的MyNeo4jGraph模块连接数据库并更新数据库结构，接下来连接DeepSeek大模型，利用GraphCypherQAChain模块构建工具链，使用LLM将问题转化为Cypher查询语言，并根据查询结果输出问题的答案。


# 运行依赖

- 拉取后需在项目主目录单独建立.env文件以连接数据库和DeepSeek，内容如下：

        # 数据库配置
        NEO4J_URI="Your neo4j URI"
        NEO4J_USERNAME="Your username"
        NEO4J_PASSWORD="Your password"

        # api-key
        DEEPSEEK_API_KEY = "Your deepseek api-key"

- 使用指令：`pip install -r requirements.txt` 可以安装该项目需要的所有资源包

- Neo4j需要安装APOC和GDS插件