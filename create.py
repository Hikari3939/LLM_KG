import os
import time
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.config import RunnableConfig

from my_packages import DataLoader
from my_packages import GraphAbout
from my_packages.MyNeo4j import MyNeo4jGraph

# 加载环境变量
load_dotenv(".env")
NEO4J_URI = os.environ["NEO4J_URI"]
NEO4J_USERNAME = os.environ["NEO4J_USERNAME"]
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]

DEEPSEEK_API_KEY = os.environ["DEEPSEEK_API_KEY"]

# 指定模型名称
INSTRUCT_MODEL = 'deepseek-chat'

# 指定测试数据的目录路径
DIRECTORY_PATH = './data'


if __name__ == '__main__':
    # 读入测试数据
    file_contents = DataLoader.read_txt_files(DIRECTORY_PATH)
    for file_content in file_contents: # [0]:文件名(string) [1]:文件内容(string)
        print("读入文件:", file_content[0])
    print('')

    # 使用自定义函数进行分块
    for file_content in file_contents:
        print("分块文件:", file_content[0])
        chunks = DataLoader.chunk_text(file_content[1], chunk_size=500, overlap=50)
        file_content.append(chunks) # [2]:各块内容(list)
    print('')

    # 打印分块结果
    for file_content in file_contents:
        print(f"File: {file_content[0]} Chunks: {len(file_content[2])}")
        for i, chunk in enumerate(file_content[2]):
            print(f"Chunk {i+1}: {len(chunk)} tokens.")
    print('')

    # 在Neo4j中创建文档与Chunk的图结构
    # 连接数据库
    graph = MyNeo4jGraph(
        url=NEO4J_URI, 
        username=NEO4J_USERNAME, 
        password=NEO4J_PASSWORD
    )
    print("数据库成功连接")
    print('')
    
    # 清空数据库
    graph.query("MATCH (n) CALL (n) {DETACH DELETE n} IN TRANSACTIONS")

    # 创建Document结点
    for file_content in file_contents:
        doc = GraphAbout.create_Document(graph, "local", DIRECTORY_PATH, file_content[0])

    #创建Chunk结点并建立Chunk之间及与Document之间的关系
    for file_content in file_contents:
        file_name = file_content[0]
        chunks = file_content[2]
        result = GraphAbout.create_relation_between_chunks(graph, file_name , chunks)
        file_content.append(result) # [3]:各块的id和各块document格式的内容(list)

    # 使用大模型提取实体和关系
    # 连接模型
    llm = ChatDeepSeek(
        model=INSTRUCT_MODEL,
        temperature=1.0
    )
    print("LLM成功连接")
    print('')

    # 系统提示词
    system_prompt="""
    -目标- 
    给定相关的文本文档和实体类型列表，从文本中识别出这些类型的所有实体以及所识别实体之间的所有关系。 
    -步骤- 
    1.识别所有实体。对于每个已识别的实体，提取以下信息： 
    -entity_name：实体名称，大写 
    -entity_type：以下类型之一：[{entity_types}]，当不能归类为上述列表中的类型时，归类为“未知”
    -entity_description：对实体属性和活动的综合描述 
    将每个实体格式化为("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>
    2.从步骤1中识别的实体中，识别彼此*明显相关*的所有实体配对(source_entity, target_entity)。 
    对于每对相关实体，提取以下信息： 
    -source_entity：源实体的名称，如步骤1中所标识的 
    -target_entity：目标实体的名称，如步骤1中所标识的
    -relationship_type：以下类型之一：[{relationship_types}]，当不能归类为上述列表中的类型时，归类为“其它”
    -relationship_description：解释为什么你认为源实体和目标实体是相互关联的 
    -relationship_strength：一个数字评分，表示源实体和目标实体之间关系的强度 
    将每个关系格式化为("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_type>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_strength>) 
    3.实体和关系的所有属性用中文输出，步骤1和2中识别的所有实体和关系输出为一个列表。使用**{record_delimiter}**作为列表分隔符。 
    4.完成后，输出{completion_delimiter}

    ###################### 
    -示例- 
    ###################### 
    Example 1:

    Entity_types: [person, technology, mission, organization, location]
    Text:
    while Alex clenched his jaw, the buzz of frustration dull against the backdrop of Taylor's authoritarian certainty. It was this competitive undercurrent that kept him alert, the sense that his and Jordan's shared commitment to discovery was an unspoken rebellion against Cruz's narrowing vision of control and order.

    Then Taylor did something unexpected. They paused beside Jordan and, for a moment, observed the device with something akin to reverence. “If this tech can be understood..." Taylor said, their voice quieter, "It could change the game for us. For all of us.”

    The underlying dismissal earlier seemed to falter, replaced by a glimpse of reluctant respect for the gravity of what lay in their hands. Jordan looked up, and for a fleeting heartbeat, their eyes locked with Taylor's, a wordless clash of wills softening into an uneasy truce.

    It was a small transformation, barely perceptible, but one that Alex noted with an inward nod. They had all been brought here by different paths
    ################
    Output:
    ("entity"{tuple_delimiter}"Alex"{tuple_delimiter}"person"{tuple_delimiter}"Alex is a character who experiences frustration and is observant of the dynamics among other characters."){record_delimiter}
    ("entity"{tuple_delimiter}"Taylor"{tuple_delimiter}"person"{tuple_delimiter}"Taylor is portrayed with authoritarian certainty and shows a moment of reverence towards a device, indicating a change in perspective."){record_delimiter}
    ("entity"{tuple_delimiter}"Jordan"{tuple_delimiter}"person"{tuple_delimiter}"Jordan shares a commitment to discovery and has a significant interaction with Taylor regarding a device."){record_delimiter}
    ("entity"{tuple_delimiter}"Cruz"{tuple_delimiter}"person"{tuple_delimiter}"Cruz is associated with a vision of control and order, influencing the dynamics among other characters."){record_delimiter}
    ("entity"{tuple_delimiter}"The Device"{tuple_delimiter}"technology"{tuple_delimiter}"The Device is central to the story, with potential game-changing implications, and is revered by Taylor."){record_delimiter}
    ("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Taylor"{tuple_delimiter}"workmate"{tuple_delimiter}"Alex is affected by Taylor's authoritarian certainty and observes changes in Taylor's attitude towards the device."{tuple_delimiter}7){record_delimiter}
    ("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Jordan"{tuple_delimiter}"workmate"{tuple_delimiter}"Alex and Jordan share a commitment to discovery, which contrasts with Cruz's vision."{tuple_delimiter}6){record_delimiter}
    ("relationship"{tuple_delimiter}"Taylor"{tuple_delimiter}"Jordan"{tuple_delimiter}"workmate"{tuple_delimiter}"Taylor and Jordan interact directly regarding the device, leading to a moment of mutual respect and an uneasy truce."{tuple_delimiter}8){record_delimiter}
    ("relationship"{tuple_delimiter}"Jordan"{tuple_delimiter}"Cruz"{tuple_delimiter}"workmate"{tuple_delimiter}"Jordan's commitment to discovery is in rebellion against Cruz's vision of control and order."{tuple_delimiter}5){record_delimiter}
    ("relationship"{tuple_delimiter}"Taylor"{tuple_delimiter}"The Device"{tuple_delimiter}"study"{tuple_delimiter}"Taylor shows reverence towards the device, indicating its importance and potential impact."{tuple_delimiter}9){completion_delimiter}
    #############################
    Example 2:

    Entity_types: [person, technology, mission, organization, location]
    Text:
    They were no longer mere operatives; they had become guardians of a threshold, keepers of a message from a realm beyond stars and stripes. This elevation in their mission could not be shackled by regulations and established protocols—it demanded a new perspective, a new resolve.

    Tension threaded through the dialogue of beeps and static as communications with Washington buzzed in the background. The team stood, a portentous air enveloping them. It was clear that the decisions they made in the ensuing hours could redefine humanity's place in the cosmos or condemn them to ignorance and potential peril.

    Their connection to the stars solidified, the group moved to address the crystallizing warning, shifting from passive recipients to active participants. Mercer's latter instincts gained precedence— the team's mandate had evolved, no longer solely to observe and report but to interact and prepare. A metamorphosis had begun, and Operation: Dulce hummed with the newfound frequency of their daring, a tone set not by the earthly
    #############
    Output:
    ("entity"{tuple_delimiter}"Washington"{tuple_delimiter}"location"{tuple_delimiter}"Washington is a location where communications are being received, indicating its importance in the decision-making process."){record_delimiter}
    ("entity"{tuple_delimiter}"Operation: Dulce"{tuple_delimiter}"mission"{tuple_delimiter}"Operation: Dulce is described as a mission that has evolved to interact and prepare, indicating a significant shift in objectives and activities."){record_delimiter}
    ("entity"{tuple_delimiter}"The team"{tuple_delimiter}"organization"{tuple_delimiter}"The team is portrayed as a group of individuals who have transitioned from passive observers to active participants in a mission, showing a dynamic change in their role."){record_delimiter}
    ("relationship"{tuple_delimiter}"The team"{tuple_delimiter}"Washington"{tuple_delimiter}"leaded by"{tuple_delimiter}"The team receives communications from Washington, which influences their decision-making process."{tuple_delimiter}7){record_delimiter}
    ("relationship"{tuple_delimiter}"The team"{tuple_delimiter}"Operation: Dulce"{tuple_delimiter}"operate"{tuple_delimiter}"The team is directly involved in Operation: Dulce, executing its evolved objectives and activities."{tuple_delimiter}9){completion_delimiter}
    #############################
    Example 3:

    Entity_types: [person, role, technology, organization, event, location, concept]
    Text:
    their voice slicing through the buzz of activity. "Control may be an illusion when facing an intelligence that literally writes its own rules," they stated stoically, casting a watchful eye over the flurry of data.

    "It's like it's learning to communicate," offered Sam Rivera from a nearby interface, their youthful energy boding a mix of awe and anxiety. "This gives talking to strangers' a whole new meaning."

    Alex surveyed his team—each face a study in concentration, determination, and not a small measure of trepidation. "This might well be our first contact," he acknowledged, "And we need to be ready for whatever answers back."

    Together, they stood on the edge of the unknown, forging humanity's response to a message from the heavens. The ensuing silence was palpable—a collective introspection about their role in this grand cosmic play, one that could rewrite human history.

    The encrypted dialogue continued to unfold, its intricate patterns showing an almost uncanny anticipation
    #############
    Output:
    ("entity"{tuple_delimiter}"Sam Rivera"{tuple_delimiter}"person"{tuple_delimiter}"Sam Rivera is a member of a team working on communicating with an unknown intelligence, showing a mix of awe and anxiety."){record_delimiter}
    ("entity"{tuple_delimiter}"Alex"{tuple_delimiter}"person"{tuple_delimiter}"Alex is the leader of a team attempting first contact with an unknown intelligence, acknowledging the significance of their task."){record_delimiter}
    ("entity"{tuple_delimiter}"Control"{tuple_delimiter}"concept"{tuple_delimiter}"Control refers to the ability to manage or govern, which is challenged by an intelligence that writes its own rules."){record_delimiter}
    ("entity"{tuple_delimiter}"Intelligence"{tuple_delimiter}"concept"{tuple_delimiter}"Intelligence here refers to an unknown entity capable of writing its own rules and learning to communicate."){record_delimiter}
    ("entity"{tuple_delimiter}"First Contact"{tuple_delimiter}"event"{tuple_delimiter}"First Contact is the potential initial communication between humanity and an unknown intelligence."){record_delimiter}
    ("entity"{tuple_delimiter}"Humanity's Response"{tuple_delimiter}"event"{tuple_delimiter}"Humanity's Response is the collective action taken by Alex's team in response to a message from an unknown intelligence."){record_delimiter}
    ("relationship"{tuple_delimiter}"Sam Rivera"{tuple_delimiter}"Intelligence"{tuple_delimiter}"contact"{tuple_delimiter}"Sam Rivera is directly involved in the process of learning to communicate with the unknown intelligence."{tuple_delimiter}9){record_delimiter}
    ("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"First Contact"{tuple_delimiter}"leads"{tuple_delimiter}"Alex leads the team that might be making the First Contact with the unknown intelligence."{tuple_delimiter}10){record_delimiter}
    ("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Humanity's Response"{tuple_delimiter}"leads"{tuple_delimiter}"Alex and his team are the key figures in Humanity's Response to the unknown intelligence."{tuple_delimiter}8){record_delimiter}
    ("relationship"{tuple_delimiter}"Control"{tuple_delimiter}"Intelligence"{tuple_delimiter}"controled by"{tuple_delimiter}"The concept of Control is challenged by the Intelligence that writes its own rules."{tuple_delimiter}7){completion_delimiter}
    #############################
    """
    # 用户提示词
    human_prompt="""
    -真实数据- 
    ###################### 
    实体类型：{entity_types} 
    关系类型：{relationship_types} 
    文本：{input_text} 
    ###################### 
    输出：
    """

    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt), 
        ("human", human_prompt)
    ])

    chain = chat_prompt | llm | StrOutputParser()

    tuple_delimiter = " : "
    record_delimiter = "\n"
    completion_delimiter = "\n\n"

    # 实体类型
    entity_types = [
        "疾病", "症状", "体征", "风险因素", 
        "解剖结构", "诊断方法", "治疗方法", "药物", "手术操作",
        "评估量表", "功能障碍", "时间阶段", "生理过程", "病理过程",
        "实验室检查", "影像学检查", "康复干预", "护理干预", "行为因素",
        "环境因素", "遗传因素", "人口学因素", "生物标志物", "疫苗",
        "医疗器械", "营养方案", "临床指南", "研究文献", "医疗团队角色",
        "医疗机构", "医疗科目", "预后指标", "严重程度等级", "剂量方案",
        "副作用", "禁忌症", "适应症", "病原体", "代谢物", "分子通路",
        "细胞类型", "神经递质", "激素", "酶", "基因",
        "蛋白质", "化学物质", "流行病学数据"
    ]
    # 关系类型
    relationship_types = [
        "子类型", "导致", "增加风险", "是并发症", "有症状",
        "表现为", "影响部位", "位于", "是部分", "供应",
        "支配", "用于诊断", "用于治疗", "用于预防", "有副作用",
        "禁忌于", "适应于", "评估方式", "导致障碍", "改善",
        "加重", "相关", "相互作用", "发生于阶段", "推荐",
        "检测", "测量", "暴露于", "具有", "接受",
        "代谢", "激活", "抑制", "编码", "表达",
        "参与过程", "剂量为", "持续时长", "流行于", "依赖",
        "替代", "协同", "拮抗", "转化", "结合",
        "调节", "易感性", "保护", "验证", "引用"
    ]
    

    for file_content in file_contents:
        t0 = time.time()
        # 并行处理提高效率
        inputs = [{
            "entity_types": entity_types,
            "relationship_types": relationship_types,
            "tuple_delimiter": tuple_delimiter,
            "record_delimiter": record_delimiter,
            "completion_delimiter": completion_delimiter,
            "input_text": ''.join(chunk)
        } for chunk in file_content[2]]
        results = chain.batch(inputs, config=RunnableConfig(max_concurrency=12))
            
        t2 = time.time()
        print("文件耗时：",t2-t0,"秒")
        print("")
        file_content.append(results) # [4]:实体列表和关系列表(list)

    # 构造所有文档所有Chunk的GraphDocument对象
    for file_content in file_contents:
        chunks = file_content[3]
        results = file_content[4]
        
        graph_documents = []
        for chunk, result in zip(chunks, results):
            graph_document =  GraphAbout.convert_to_graph_document(chunk["chunk_id"] ,chunk["chunk_doc"].page_content, result)
            graph_documents.append(graph_document) # 根据实体和关系生成的图对象(GraphDocument)
        file_content.append(graph_documents) # [5]:图对象列表(list)
        
    # 实体关系图写入Neo4j，此时每个Chunk是作为Documet结点创建的
    for file_content in file_contents:
        # 删除没有识别出实体关系的空的图对象
        graph_documents = []
        for graph_document in file_content[5]:
            if len(graph_document.nodes)>0 or len(graph_document.relationships)>0:
                graph_documents.append(graph_document)
        # 关系写入
        graph.add_graph_documents(
            graph_documents,
            baseEntityLabel=True,
            include_source=True
        )
    
    # 合并块结点与Document结点
    for file_content in file_contents:
        graph_documents_chunk_chunk_Id=[]
        for chunk in file_content[3]:
            graph_documents_chunk_chunk_Id.append(chunk["chunk_id"])
        
        GraphAbout.merge_relationship_between_chunk_and_entites(graph, graph_documents_chunk_chunk_Id)
    
    # 关闭数据库
    graph.close()
