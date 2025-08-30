# 重载Neo4jGraph类，使节点合并时对description属性进行拼接而非直接替代
from langchain_neo4j.graphs.graph_document import GraphDocument
from typing import Any, Dict, List, Optional
from langchain_neo4j import Neo4jGraph
from hashlib import md5

BASE_ENTITY_LABEL = "__Entity__"

include_docs_query = (
    "MERGE (d:Document {id:$document.metadata.id}) "
    "SET d.text = $document.page_content "
    "SET d += $document.metadata "
    "WITH d "
)

# 修改后的节点合并函数
def my_get_node_import_query(baseEntityLabel: bool, include_source: bool) -> str:
    if baseEntityLabel:
        return (
            f"{include_docs_query if include_source else ''}"
            "UNWIND $data AS row "
            f"MERGE (source:`{BASE_ENTITY_LABEL}` {{id: row.id}}) "
            
            # 对description属性进行拼接处理
            f"{'WITH d, source, row, ' if include_source else 'WITH source, row, '}"
            # 创建新属性映射
            "apoc.map.setKey( "
            "   row.properties, "  # 原始属性
            "   'description', "  # 指定要修改的属性名
            "   COALESCE( "
            "       CASE "
                        # 当原description和新description都不为空时，用分隔符连接
            "           WHEN (source.description IS NOT NULL AND source.description <> '') "
            "                 AND (row.properties.description IS NOT NULL AND row.properties.description <> '') "
            "           THEN source.description + '；' + row.properties.description "
                        # 当只有原description有值时，直接使用原值
            "           WHEN (source.description IS NOT NULL AND source.description <> '') "
            "                 AND (row.properties.description IS NULL OR row.properties.description = '') "
            "           THEN source.description "
                        # 当只有新description有值时，直接使用新值
            "           WHEN (source.description IS NULL OR source.description = '') "
            "                 AND (row.properties.description IS NOT NULL AND row.properties.description <> '') "
            "           THEN row.properties.description "
                        # 其他情况（两者都为空）返回空字符串
            "           ELSE '' "
            "        END, "
            "       '' "
            "    ) "  # 外层COALESCE确保最终结果不为NULL
            ") AS mergedProps "
            
            "SET source += mergedProps "  # 更新节点属性
                        
            f"{'MERGE (d)-[:MENTIONS]->(source) ' if include_source else ''}"
            
            # 检查并处理标签
            "WITH source, row, "
            
            # 标签合并逻辑
            "CASE "
            # row.type 为'未知'
            f" WHEN row.type = '未知' AND size([l IN labels(source) WHERE l <> '{BASE_ENTITY_LABEL}' AND l <> '未知']) > 0 "
            "    THEN labels(source) "  # 仅保留 source 标签
            "  WHEN '未知' IN labels(source) "  # source 标签中含有'未知'
            f"   THEN ['{BASE_ENTITY_LABEL}'] + [row.type] "  # 仅保留 row.type 标签
            "  ELSE labels(source) + [row.type] "
            "END AS validLabels "

            # 设置标签
            "CALL apoc.create.setLabels(source, "
            f"  [l IN validLabels WHERE l <> '']) "  # 保留有效标签
            "YIELD node "

            "RETURN distinct 'done' AS result"
        )
    else:
        return (
            f"{include_docs_query if include_source else ''}"
            "UNWIND $data AS row "
            "CALL apoc.merge.node([row.type], {id: row.id}, "
            "row.properties, {}) YIELD node "
            f"{'MERGE (d)-[:MENTIONS]->(node) ' if include_source else ''}"
            "RETURN distinct 'done' AS result"
        )

def _get_rel_import_query(baseEntityLabel: bool) -> str:
    if baseEntityLabel:
        return (
            "UNWIND $data AS row "
            f"MERGE (source:`{BASE_ENTITY_LABEL}` {{id: row.source}}) "
            f"MERGE (target:`{BASE_ENTITY_LABEL}` {{id: row.target}}) "
            "WITH source, target, row "
            "CALL apoc.merge.relationship(source, row.type, "
            "{}, row.properties, target) YIELD rel "
            "RETURN distinct 'done'"
        )
    else:
        return (
            "UNWIND $data AS row "
            "CALL apoc.merge.node([row.source_label], {id: row.source},"
            "{}, {}) YIELD node as source "
            "CALL apoc.merge.node([row.target_label], {id: row.target},"
            "{}, {}) YIELD node as target "
            "CALL apoc.merge.relationship(source, row.type, "
            "{}, row.properties, target) YIELD rel "
            "RETURN distinct 'done'"
        )

def _remove_backticks(text: str) -> str:
    return text.replace("`", "")

class MyNeo4jGraph(Neo4jGraph):
    def __init__(
        self, 
        url: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None,
        timeout: Optional[float] = None,
        sanitize: bool = False,
        refresh_schema: bool = True,
        *,
        driver_config: Optional[Dict] = None,
        enhanced_schema: bool = False,
    ):
        
        super().__init__(
            url, username, password, 
            database, timeout, sanitize, refresh_schema, 
            driver_config=driver_config, 
            enhanced_schema=enhanced_schema
        )
    
    def add_graph_documents(
        self,
        graph_documents: List[GraphDocument],
        include_source: bool = False,
        baseEntityLabel: bool = False,
    ) -> None:
        """
        This method constructs nodes and relationships in the graph based on the
        provided GraphDocument objects.

        Parameters:
        - graph_documents (List[GraphDocument]): A list of GraphDocument objects
        that contain the nodes and relationships to be added to the graph. Each
        GraphDocument should encapsulate the structure of part of the graph,
        including nodes, relationships, and optionally the source document information.
        - include_source (bool, optional): If True, stores the source document
        and links it to nodes in the graph using the MENTIONS relationship.
        This is useful for tracing back the origin of data. Merges source
        documents based on the `id` property from the source document metadata
        if available; otherwise it calculates the MD5 hash of `page_content`
        for merging process. Defaults to False.
        - baseEntityLabel (bool, optional): If True, each newly created node
        gets a secondary __Entity__ label, which is indexed and improves import
        speed and performance. Defaults to False.

        Raises:
            RuntimeError: If the connection has been closed.
        """
        self._check_driver_state()
        if baseEntityLabel:  # Check if constraint already exists
            constraint_exists = any(
                [
                    el["labelsOrTypes"] == [BASE_ENTITY_LABEL]
                    and el["properties"] == ["id"]
                    for el in self.structured_schema.get("metadata", {}).get(
                        "constraint", []
                    )
                ]
            )

            if not constraint_exists:
                # Create constraint
                self.query(
                    f"CREATE CONSTRAINT IF NOT EXISTS FOR (b:{BASE_ENTITY_LABEL}) "
                    "REQUIRE b.id IS UNIQUE;"
                )
                self.refresh_schema()  # Refresh constraint information

        # Check each graph_document has a source when include_source is true
        if include_source:
            for doc in graph_documents:
                if doc.source is None:
                    raise TypeError(
                        "include_source is set to True, "
                        "but at least one document has no `source`."
                    )

        node_import_query = my_get_node_import_query(baseEntityLabel, include_source)
        rel_import_query = _get_rel_import_query(baseEntityLabel)
        for document in graph_documents:
            node_import_query_params: dict[str, Any] = {
                "data": [el.__dict__ for el in document.nodes]
            }
            if include_source and document.source:
                if not document.source.metadata.get("id"):
                    document.source.metadata["id"] = md5(
                        document.source.page_content.encode("utf-8")
                    ).hexdigest()
                node_import_query_params["document"] = document.source.__dict__

            # Remove backticks from node types
            for node in document.nodes:
                node.type = _remove_backticks(node.type)
            # Import nodes
            self.query(node_import_query, node_import_query_params)
            # Import relationships
            self.query(
                rel_import_query,
                {
                    "data": [
                        {
                            "source": el.source.id,
                            "source_label": _remove_backticks(el.source.type),
                            "target": el.target.id,
                            "target_label": _remove_backticks(el.target.type),
                            "type": _remove_backticks(
                                el.type.replace(" ", "_").upper()
                            ),
                            "properties": el.properties,
                        }
                        for el in document.relationships
                    ]
                },
            )
