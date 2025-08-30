# 读入测试数据
import os
import codecs

# 读入测试文件。
def read_txt_files(directory):
    # 存放结果的列表
    results = []
    # 遍历指定目录下的所有文件和文件夹
    for filename in os.listdir(directory):
        # 检查文件扩展名是否为.txt
        if filename.endswith(".txt"):
            # 构建完整的文件路径
            file_path = os.path.join(directory, filename)
            # 打开并读取文件内容
            with codecs.open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # 将文件名和内容以列表形式添加到结果列表
            results.append([filename, content])
    
    return results

# 文本分块
import hanlp

# 单任务模型，分词，token的计数是计算词，包括标点符号。
tokenizer = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)

# 划分段落。
def split_into_paragraphs(text):
    text = text.replace('\r', '\n')
    
    paragraphs = text.split('\n')
    while '' in paragraphs:
        paragraphs.remove('')
        
    return paragraphs

# 判断token是否为句子结束符，视情况再增加。
def is_sentence_end(token):
    return token in ['。', '！', '？']

# 向后查找到句子结束符，用于动态调整chunk划分以保证chunk以完整的句子结束
def find_sentence_boundary_forward(tokens, chunk_size):
    end = len(tokens)  # 默认的end值设置为tokens的长度
    for i in range(chunk_size, len(tokens)):  # 从chunk_size开始向后查找
        if is_sentence_end(tokens[i]):
            end = i + 1  # 包含句尾符号
            break
    return end  

# 从位置start开始向前寻找上一句的句子结束符，以保证分块重叠的部分从一个完整的句子开始。
def find_sentence_boundary_backward(tokens, start):
    for i in range(start - 1, -1, -1):
        if is_sentence_end(tokens[i]):
            return i + 1  # 包含句尾符号
    return 0  # 找不到
  
# 文本分块，文本块的参考大小为chunk_size，文本块之间重叠部分的参考大小为overlap。
# 为了保证文本块之间重叠的部分及文本块末尾截断的部分都是完整的句子，
# 文本块的大小和重叠部分的大小都是根据当前文本块的内容动态调整的，是浮动的值。
def chunk_text(text, chunk_size=300, overlap=50):
    if chunk_size <= overlap:  # 参数检查
        raise ValueError("chunk_size must be greater than overlap.")
    # 先划分为段落，段落保存了语义上的信息，整个段落去处理。  
    paragraphs = split_into_paragraphs(text)
    chunks = []
    buffer = []
    # 逐个段落处理
    i = 0
    while i < len(paragraphs):
        # 注满buffer，直到大于chunk_szie，整个段落读入，段落保存了语义上的信息。
        while len(buffer) < chunk_size and i < len(paragraphs):
            tokens = tokenizer(paragraphs[i])
            buffer.extend(tokens)
            i += 1
        # 当前buffer分块
        while len(buffer) >= chunk_size:
            # 保证从完整的句子处截断。
            end = find_sentence_boundary_forward(buffer, chunk_size)
            chunk = buffer[:end]
            chunks.append(chunk)  # 保留token的状态以便后面计数
            # 保证重叠的部分从完整的句子开始。
            start_next = find_sentence_boundary_backward(buffer, end - overlap)
            if start_next==0:  # 找不到了上一句的句子结束符，调整重叠范围再找一次。
                start_next = find_sentence_boundary_backward(buffer, end-1)
            if start_next==0:  # 真的找不到，放弃块首的完整句子重叠。
                start_next = end - overlap
            buffer=buffer[start_next:]
        
    if buffer:  # 如果缓冲区还有剩余的token
        if len(chunks) > 0:
            # 检查一下剩余部分是否已经包含在最后一个分块之中，它只是留作块间重叠。
            last_chunk = chunks[len(chunks)-1]
            rest = ''.join(buffer)
            temp = ''.join(last_chunk[len(last_chunk)-len(rest):])
            if temp!=rest:   # 如果不是留作重叠，则是最后的一个分块。
                chunks.append(buffer)
        else:
            chunks.append(buffer)
    
    return chunks
