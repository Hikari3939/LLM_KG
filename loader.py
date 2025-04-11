import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, PyMuPDFLoader

# 读入测试文件。
def load_documents(directory):
    # 存放结果的列表
    results = []
    # 遍历指定目录下的所有文件和文件夹
    for filename in os.listdir(directory):
        # 加载.txt文件
        if filename.endswith(".txt"):
            # 构建完整的文件路径
            file_path = os.path.join(directory, filename)
            # 打开并读取文件内容
            loader = TextLoader(file_path, encoding='utf-8')
            document = loader.load()
            # 将文件名和内容以列表形式添加到结果列表
            results.append((filename, document))
        # 加载.pdf文件
        if filename.endswith(".pdf"):
            # 构建完整的文件路径
            file_path = os.path.join(directory, filename)
            # 打开并读取文件内容
            loader = PyMuPDFLoader(file_path)
            document = loader.load()
            # 将文件名和内容以列表形式添加到结果列表
            results.append((filename, document))
    
    return results

# 分割目标文件夹中所有测试数据文件
def split_folder(directory, chunk_size, chunk_overlap):
    # 存放结果的列表
    all_documents = []
    files = load_documents(directory)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap, 
        separators=["\n\n", "\n", ".", ",", " ", ""], 
    )

    for filename, docs in files:
        # 分割文件
        print(f"Processing: {filename}")
        
        split_docs = text_splitter.split_documents(docs) # 返回一个分割后的列表
        all_documents.append((filename, split_docs))
    
    return all_documents
