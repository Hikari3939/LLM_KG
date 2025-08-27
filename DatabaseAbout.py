# 在Neo4j中创建文档与Chunk的图结构----------------------------------------------
# https://python.langchain.com/v0.2/api_reference/community/graphs/langchain_community.graphs.neo4j_graph.Neo4jGraph.html#langchain_community.graphs.neo4j_graph.Neo4jGraph.add_graph_documents
from langchain_core.documents import Document
import hashlib
import logging
from typing import List

# 创建Document结点，与Chunk之间按属性名fileName匹配。
def create_Document(graph, type, uri, file_name):
    query = """
    MERGE(d:`__Document__` {fileName :$file_name}) SET d.type=$type,
          d.uri=$uri
    RETURN d;
    """
    doc = graph.query(query,{"file_name":file_name,"type":type,"uri":uri})
    return doc

#创建Chunk结点并建立Chunk之间及与Document之间的关系
#这个程序直接从Neo4j KG Builder拷贝引用，为了增加tokens属性稍作修改。
#https://github.com/neo4j-labs/llm-graph-builder/blob/main/backend/src/make_relationships.py
def create_relation_between_chunks(graph, file_name, chunks: List)->list:
    logging.info("creating FIRST_CHUNK and NEXT_CHUNK relationships between chunks")
    current_chunk_id = ""
    lst_chunks_including_hash = []
    batch_data = []
    relationships = []
    offset=0
    for i, chunk in enumerate(chunks):
        page_content = ''.join(chunk)
        page_content_sha1 = hashlib.sha1(page_content.encode()) # chunk.page_content.encode()
        previous_chunk_id = current_chunk_id
        current_chunk_id = page_content_sha1.hexdigest()
        position = i + 1 
        if i>0:
            last_page_content = ''.join(chunks[i-1])
            offset += len(last_page_content)  # chunks[i-1].page_content
        if i == 0:
            firstChunk = True
        else:
            firstChunk = False  
        metadata = {"position": position,"length": len(page_content), "content_offset":offset, "tokens":len(chunk)}
        chunk_document = Document(
            page_content=page_content, metadata=metadata
        )
        
        chunk_data = {
            "id": current_chunk_id,
            "pg_content": chunk_document.page_content,
            "position": position,
            "length": chunk_document.metadata["length"],
            "f_name": file_name,
            "previous_id" : previous_chunk_id,
            "content_offset" : offset,
            "tokens" : len(chunk)
        }
        
        batch_data.append(chunk_data)
        
        lst_chunks_including_hash.append({'chunk_id': current_chunk_id, 'chunk_doc': chunk_document})
        
        # create relationships between chunks
        if firstChunk:
            relationships.append({"type": "FIRST_CHUNK", "chunk_id": current_chunk_id})
        else:
            relationships.append({
                "type": "NEXT_CHUNK",
                "previous_chunk_id": previous_chunk_id,  # ID of previous chunk
                "current_chunk_id": current_chunk_id
            })
          
    query_to_create_chunk_and_PART_OF_relation = """
        UNWIND $batch_data AS data
        MERGE (c:`__Chunk__` {id: data.id})
        SET c.text = data.pg_content, c.position = data.position, c.length = data.length, c.fileName=data.f_name, 
            c.content_offset=data.content_offset, c.tokens=data.tokens
        WITH data, c
        MATCH (d:`__Document__` {fileName: data.f_name})
        MERGE (c)-[:PART_OF]->(d)
    """
    graph.query(query_to_create_chunk_and_PART_OF_relation, params={"batch_data": batch_data})
    
    query_to_create_FIRST_relation = """ 
        UNWIND $relationships AS relationship
        MATCH (d:`__Document__` {fileName: $f_name})
        MATCH (c:`__Chunk__` {id: relationship.chunk_id})
        FOREACH(r IN CASE WHEN relationship.type = 'FIRST_CHUNK' THEN [1] ELSE [] END |
                MERGE (d)-[:FIRST_CHUNK]->(c))
        """
    graph.query(query_to_create_FIRST_relation, params={"f_name": file_name, "relationships": relationships})   
    
    query_to_create_NEXT_CHUNK_relation = """ 
        UNWIND $relationships AS relationship
        MATCH (c:`__Chunk__` {id: relationship.current_chunk_id})
        WITH c, relationship
        MATCH (pc:`__Chunk__` {id: relationship.previous_chunk_id})
        FOREACH(r IN CASE WHEN relationship.type = 'NEXT_CHUNK' THEN [1] ELSE [] END |
                MERGE (c)<-[:NEXT_CHUNK]-(pc))
        """
    graph.query(query_to_create_NEXT_CHUNK_relation, params={"relationships": relationships})   
    
    return lst_chunks_including_hash

# 提取的实体关系写入Neo4j-------------------------------------------------------
# 由answer.content生成一个GraphDocument对象
# 每个GraphDocument对象里增加一个metadata属性chunk_id，以便与前面建立的Chunk结点关联
import re
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship
from langchain_core.documents import Document
from MyNeo4j import MyNeo4jGraph

# 将每个块提取的实体关系文本转换为LangChain的GraphDocument对象
def convert_to_graph_document(chunk_id, input_text, result):
    # 提取节点和关系
    node_pattern = re.compile(r'\("entity" : "(.+?)" : "(.+?)" : "(.+?)"\)')
    relationship_pattern = re.compile(r'\("relationship" : "(.+?)" : "(.+?)" : "(.+?)" : "(.+?)" : (.+?)\)')

    nodes = {}
    relationships = []

    # 解析并创建节点
    for match in node_pattern.findall(result):
        node_id, node_type, description = match
        if node_id not in nodes:
            nodes[node_id] = Node(id=node_id, type=node_type, properties={'description': description})

    # 解析并处理关系
    for match in relationship_pattern.findall(result):
        source_id, target_id, type, description, weight = match
        # 确保source节点存在
        if source_id not in nodes:
            nodes[source_id] = Node(id=source_id, type="未知", properties={'description': 'No additional data'})
        # 确保target节点存在
        if target_id not in nodes:
            nodes[target_id] = Node(id=target_id, type="未知", properties={'description': 'No additional data'})
        relationships.append(Relationship(source=nodes[source_id], target=nodes[target_id], type=type,
            properties={"description":description, "weight":float(weight)}))

    # 创建图对象
    graph_document = GraphDocument(
        nodes=list(nodes.values()),
        relationships=relationships,
        # page_content不能为空。
        source=Document(page_content=input_text, metadata={"chunk_id": chunk_id})
    )
    return graph_document

# 合并Chunk结点与add_graph_documents()创建的相应Document结点，
# 迁移所有的实体关系到Chunk结点，并删除相应的Document结点。
# 完成Document->Chunk->Entity的结构。
def merge_relationship_between_chunk_and_entites(graph: MyNeo4jGraph, graph_documents_chunk_chunk_Id : list):
    batch_data = []
    logging.info("Create MENTIONS relationship between chunks and entities")
    for graph_doc_chunk_id in graph_documents_chunk_chunk_Id:
        query_data={
            'chunk_id': graph_doc_chunk_id,
        }
        batch_data.append(query_data)

    if batch_data:
        unwind_query = """
          UNWIND $batch_data AS data
          MATCH (c:`__Chunk__` {id: data.chunk_id}), (d:Document{chunk_id:data.chunk_id})
          WITH c, d
          MATCH (d)-[r:MENTIONS]->(e)
          MERGE (c)-[newR:MENTIONS]->(e)
          ON CREATE SET newR += properties(r)
          DETACH DELETE d
                """
        graph.query(unwind_query, params={"batch_data": batch_data})
