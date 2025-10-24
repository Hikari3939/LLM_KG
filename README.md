# 脑卒中知识图谱构建与问答系统实现

## 运行依赖

- 拉取后需在项目主目录单独建立.env文件，内容如下：

        # 数据库配置
        NEO4J_URI="Your neo4j URI"
        NEO4J_USERNAME="Your username"
        NEO4J_PASSWORD="Your password"

        # api-key
        DEEPSEEK_API_KEY="Your deepseek api-key"
        LANGSMITH_API_KEY="Your LangSmith api-key"

- 需要在项目主目录建立data文件夹并放入txt文件作为数据源，可嵌套，但不能使用其它文件格式

- 使用指令：`pip install -r requirements.txt` 安装该项目需要的所有资源包

- Neo4j需要安装APOC和GDS插件
  
- 需要安装GPU对应版本的cuda和cuda对应版本的pytorch

## 参考

- [GraphRAG实战](https://github.com/icejean/GraphRAG)
