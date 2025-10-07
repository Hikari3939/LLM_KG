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
    -entity_type：以下类型之一：[{entity_types}]，***当不能归类为上述列表中的类型时，归类为“其他”***
    -entity_description：对实体属性和活动的综合描述 
    将每个实体格式化为("entity"{tuple_delimiter}<entity_name>{tuple_delimiter}<entity_type>{tuple_delimiter}<entity_description>
    2.从步骤1中识别的实体中，识别彼此*明显相关*的所有实体配对(source_entity, target_entity)。 
    对于每对相关实体，提取以下信息： 
    -source_entity：源实体的名称，如步骤1中所标识的 
    -target_entity：目标实体的名称，如步骤1中所标识的
    -relationship_type：以下类型之一：[{relationship_types}]，***当不能归类为上述列表中的类型时，归类为“其他”***
    -relationship_description：解释为什么你认为源实体和目标实体是相互关联的 
    -relationship_strength：一个数字评分，表示源实体和目标实体之间关系的强度 
    将每个关系格式化为("relationship"{tuple_delimiter}<source_entity>{tuple_delimiter}<target_entity>{tuple_delimiter}<relationship_type>{tuple_delimiter}<relationship_description>{tuple_delimiter}<relationship_strength>) 
    3.实体和关系的所有属性用中文输出，步骤1和2中识别的所有实体和关系输出为一个列表。使用**{record_delimiter}**作为列表分隔符。 
    4.完成后，输出{completion_delimiter}
    -注意- 
    ***严禁使用列表中不存在的实体类型和关系类型***。如果遇到不能归类的类型，请将其归类为“其他”。

    ###################### 
    -示例- 
    ###################### 
    Example 1:
    
    Entity_types: 
    [person, mission, organization, event, location, concept]
    
    Relationship_types: 
    [workmate, study, contact, leads, leaded by, operate]
    
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
    ("entity"{tuple_delimiter}"The Device"{tuple_delimiter}"others"{tuple_delimiter}"The Device is central to the story, with potential game-changing implications, and is revered by Taylor."){record_delimiter}
    ("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Taylor"{tuple_delimiter}"workmate"{tuple_delimiter}"Alex is affected by Taylor's authoritarian certainty and observes changes in Taylor's attitude towards the device."{tuple_delimiter}7){record_delimiter}
    ("relationship"{tuple_delimiter}"Alex"{tuple_delimiter}"Jordan"{tuple_delimiter}"workmate"{tuple_delimiter}"Alex and Jordan share a commitment to discovery, which contrasts with Cruz's vision."{tuple_delimiter}6){record_delimiter}
    ("relationship"{tuple_delimiter}"Taylor"{tuple_delimiter}"Jordan"{tuple_delimiter}"workmate"{tuple_delimiter}"Taylor and Jordan interact directly regarding the device, leading to a moment of mutual respect and an uneasy truce."{tuple_delimiter}8){record_delimiter}
    ("relationship"{tuple_delimiter}"Jordan"{tuple_delimiter}"Cruz"{tuple_delimiter}"workmate"{tuple_delimiter}"Jordan's commitment to discovery is in rebellion against Cruz's vision of control and order."{tuple_delimiter}5){record_delimiter}
    ("relationship"{tuple_delimiter}"Taylor"{tuple_delimiter}"The Device"{tuple_delimiter}"study"{tuple_delimiter}"Taylor shows reverence towards the device, indicating its importance and potential impact."{tuple_delimiter}9){completion_delimiter}
    #############################
    Example 2:

    Entity_types: 
    [person, mission, organization, event, location, concept]
    
    Relationship_types: 
    [workmate, study, contact, leads, leaded by, operate]
    
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
    ("relationship"{tuple_delimiter}"Control"{tuple_delimiter}"Intelligence"{tuple_delimiter}"others"{tuple_delimiter}"The concept of Control is challenged by the Intelligence that writes its own rules."{tuple_delimiter}7){completion_delimiter}
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
        # 临床核心实体
        {"name": "疾病", "description": "明确的病理状态，如脑卒中、高血压等"},
        {"name": "症状", "description": "患者主观报告的不适感，如头痛、眩晕"},
        {"name": "体征", "description": "客观可观察或测量的临床表现，如偏瘫、言语障碍"},
        {"name": "功能障碍", "description": "身体或认知功能受损，如运动障碍、认知缺陷"},
        {"name": "并发症", "description": "疾病过程中继发的其他健康问题，如肺炎、深静脉血栓"},
        
        # 诊断与评估实体
        {"name": "诊断方法", "description": "用于确定疾病的通用方法，如临床评估、影像学分析"},
        {"name": "实验室检查", "description": "通过生物样本检测获得的指标，如血常规、凝血功能"},
        {"name": "影像学检查", "description": "利用影像技术进行的检查，如CT、MRI"},
        {"name": "评估量表", "description": "标准化工具用于量化症状或功能，如NIHSS、格拉斯哥昏迷量表"},
        {"name": "生理指标", "description": "可测量的生理参数，如血压、心率"},
        {"name": "生物标志物", "description": "客观指示病理状态的生物分子，如C反应蛋白、同型半胱氨酸"},
        
        # 治疗与干预实体
        {"name": "治疗方法", "description": "广义治疗策略，如药物治疗、康复治疗"},
        {"name": "药物", "description": "用于治疗或预防疾病的化学物质，如阿司匹林、他汀类药物"},
        {"name": "手术操作", "description": "外科介入程序，如血栓切除术、血管成形术"},
        {"name": "康复干预", "description": "旨在恢复功能的非药物干预，如物理治疗、作业治疗"},
        {"name": "护理干预", "description": "护理人员提供的措施，如体位管理、健康教育"},
        {"name": "剂量方案", "description": "药物使用的具体剂量和频率，如每日一次、静脉滴注"},
        {"name": "副作用", "description": "治疗引起的不良反应，如出血、过敏"},
        {"name": "禁忌症", "description": "不适合特定治疗的情况，如活动性出血患者禁用抗凝药"},
        {"name": "适应症", "description": "适合特定治疗的情况，如急性缺血性脑卒中适用溶栓治疗"},
        
        # 风险与保护因素
        {"name": "行为因素", "description": "与生活方式相关的因素，如吸烟、饮酒"},
        {"name": "环境因素", "description": "外部环境暴露，如空气污染、职业风险"},
        {"name": "遗传因素", "description": "遗传背景相关的风险，如家族史、基因突变"},
        {"name": "人口学因素", "description": "人口统计特征，如年龄、性别"},
        {"name": "生理风险因素", "description": "内在生理状态，如肥胖、高血压"},
        
        # 生物医学基础实体
        {"name": "解剖结构", "description": "身体组织结构，如大脑、血管"},
        {"name": "细胞类型", "description": "特定细胞类别，如神经元、胶质细胞"},
        {"name": "生理过程", "description": "正常身体功能过程，如血液循环、神经传导"},
        {"name": "病理过程", "description": "疾病相关机制，如缺血、炎症"},
        {"name": "分子通路", "description": "生物分子相互作用的路径，如凝血通路、凋亡通路"},
        {"name": "神经递质", "description": "神经信号传递分子，如多巴胺、谷氨酸"},
        {"name": "激素", "description": "内分泌调节分子，如肾上腺素、皮质醇"},
        {"name": "酶", "description": "催化生物反应的蛋白质，如凝血酶、纤溶酶"},
        {"name": "基因", "description": "遗传信息单位，如APOE基因、NOTCH3基因"},
        {"name": "蛋白质", "description": "功能生物分子，如淀粉样蛋白、tau蛋白"},
        {"name": "化学物质", "description": "一般化学实体，如葡萄糖、氧气"},
        {"name": "代谢物", "description": "代谢过程中产生的小分子，如乳酸、酮体"},
        
        # 医疗资源与管理实体
        {"name": "医疗团队角色", "description": "医疗专业人员角色，如神经科医生、护士"},
        {"name": "医疗机构", "description": "提供医疗服务的场所，如医院、康复中心"},
        {"name": "医疗科目", "description": "医学专业领域，如神经内科、放射科"},
        {"name": "医疗器械", "description": "用于诊断或治疗的设备，如输液泵、监护仪"},
        {"name": "疫苗", "description": "预防性生物制品，如流感疫苗"},
        {"name": "营养方案", "description": "饮食管理计划，如低盐饮食、肠内营养"},
        {"name": "临床指南", "description": "基于证据的实践推荐，如AHA/ASA脑卒中指南"},
        {"name": "研究文献", "description": "科学出版物，如随机对照试验、综述文章"},
        
        # 预后与流行病学实体
        {"name": "预后指标", "description": "预测疾病结局的参数，如死亡率、功能恢复率"},
        {"name": "时间阶段", "description": "疾病或治疗的时间划分，如急性期、恢复期"},
        {"name": "流行病学数据", "description": "疾病分布和决定因素的数据，如发病率、患病率"}
    ]
    # 关系类型
    relationship_types = [
        # 层次与分类关系
        {"name": "子类型", "description": "表示概念间的继承关系，如缺血性脑卒中是脑卒中的子类型"},
        {"name": "属于", "description": "表示实例与类别的关系，如某患者属于高血压人群"},
        {"name": "部分整体", "description": "表示组成部分与整体的关系，如大脑中动脉是脑动脉的部分"},
        
        # 因果与风险关系
        {"name": "导致", "description": "直接因果关系，如高血压导致脑卒中"},
        {"name": "增加风险", "description": "增加患病概率的关系，如吸烟增加脑卒中风险"},
        {"name": "减少风险", "description": "降低患病概率的关系，如锻炼减少脑卒中风险"},
        {"name": "加重", "description": "使症状或病情恶化，如感染加重神经功能缺损"},
        {"name": "改善", "description": "使症状或病情好转，如康复治疗改善运动功能"},
        
        # 临床关联关系
        {"name": "有症状", "description": "疾病与症状的关联，如脑卒中有偏瘫症状"},
        {"name": "表现为", "description": "疾病或病理过程的临床表现"},
        {"name": "是并发症", "description": "疾病引发的继发性问题，如肺炎是脑卒中的并发症"},
        {"name": "导致障碍", "description": "疾病导致的功能障碍，如脑卒中导致言语障碍"},
        
        # 解剖定位关系
        {"name": "位于", "description": "解剖结构的位置关系"},
        {"name": "影响部位", "description": "疾病或病理过程影响的解剖部位"},
        {"name": "供应", "description": "血管供应关系，如大脑中动脉供应基底节区"},
        {"name": "支配", "description": "神经支配关系"},
        
        # 诊断评估关系
        {"name": "用于诊断", "description": "检查方法用于疾病诊断"},
        {"name": "检测", "description": "检查方法检测特定指标"},
        {"name": "测量", "description": "量化测量特定参数"},
        {"name": "评估方式", "description": "评估工具与评估对象的关系"},
        {"name": "验证", "description": "方法或工具的验证关系"},
        
        # 治疗干预关系
        {"name": "用于治疗", "description": "治疗方法针对特定疾病"},
        {"name": "用于预防", "description": "干预措施用于疾病预防"},
        {"name": "适应于", "description": "治疗的适应症关系"},
        {"name": "禁忌于", "description": "治疗的禁忌症关系"},
        {"name": "有副作用", "description": "治疗引起的不良反应"},
        {"name": "剂量为", "description": "药物的剂量方案"},
        {"name": "给药方式", "description": "药物的给药途径"},
        
        # 药物相互作用
        {"name": "协同", "description": "药物间的协同作用"},
        {"name": "拮抗", "description": "药物间的拮抗作用"},
        {"name": "相互作用", "description": "药物间的相互影响"},
        
        # 生物医学机制
        {"name": "参与过程", "description": "分子或细胞参与生物过程"},
        {"name": "激活", "description": "生物分子的激活作用"},
        {"name": "抑制", "description": "生物分子的抑制作用"},
        {"name": "代谢", "description": "物质的代谢转化"},
        {"name": "编码", "description": "基因编码蛋白质"},
        {"name": "表达", "description": "基因或蛋白质的表达"},
        {"name": "调节", "description": "生物调节作用"},
        {"name": "结合", "description": "分子间的结合作用"},
        
        # 时间与阶段关系
        {"name": "发生于", "description": "事件发生的时间阶段"},
        {"name": "持续时长", "description": "过程或治疗的持续时间"},
        
        # 流行病学关系
        {"name": "流行于", "description": "疾病在人群或地区的分布"},
        {"name": "易感性", "description": "人群对疾病的易感程度"},
        
        # 资源应用关系
        {"name": "使用", "description": "使用医疗资源或工具"},
        {"name": "推荐", "description": "指南或专家推荐"},
        {"name": "引用", "description": "文献或证据引用"},
        {"name": "基于", "description": "基于证据或原理"},
        
        # 预后结局关系
        {"name": "预后指标", "description": "与预后相关的指标"},
        {"name": "预测", "description": "预测疾病结局"}
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
        file_content.append(results) # [4]:实体列表和关系列表(list)
    print("LLM处理完成")
    print("")

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
    print("知识图谱初步构建完成")
    print("")

    # 关闭数据库
    graph.close()
