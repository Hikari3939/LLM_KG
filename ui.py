import streamlit as st
import time
import re
import uuid
import datetime

# 模块导入
try:
    from agent import ask_agent_with_source, get_source, clear_session, agent, user_config
    from langchain_core.messages import HumanMessage, AIMessage, RemoveMessage
    AGENT_AVAILABLE = True
except ImportError as e:
    st.error(f"无法导入agent模块: {e}")
    AGENT_AVAILABLE = False
    # 占位函数
    def ask_agent_with_source(prompt, session_id):
        return "Agent模块不可用，请检查依赖安装"
    def get_source(source_id):
        return "Agent模块不可用，请检查依赖安装"
    def clear_session():
        pass
    def sync_messages_to_agent(messages, session_id):
        pass

# 页面配置
def setup_page_config():
    """配置页面基本设置"""
    try:
        st.set_page_config(
            page_title="GraphRAG - 知识图谱智能问答",
            layout="wide",
            initial_sidebar_state="expanded",
            menu_items={
                'Get Help': 'https://github.com/your-repo',
                'Report a bug': 'https://github.com/your-repo/issues',
                'About': "基于知识图谱的智能问答系统"
            }
        )
    except Exception as e:
        st.error(f"页面配置错误: {e}")

def setup_custom_styles():
    """设置自定义样式"""
    st.markdown("""
    <style>
        /* 全局浅蓝色主题 */
        :root {
            --primary-color: #3b82f6;
            --primary-dark: #2563eb;
            --primary-light: #60a5fa;
            --secondary-color: #0ea5e9;
            --accent-color: #06b6d4;
            --text-primary: #1e40af;
            --text-secondary: #1e3a8a;
            --bg-primary: #f0f9ff;
            --bg-secondary: #e0f2fe;
        }
        
        /* 完全隐藏整个header区域 */
        header {
            display: none !important;
        }
        
        /* 侧边栏始终显示 */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f0f9ff 0%, #e0f2fe 100%);
            min-width: 300px !important;
            max-width: 350px !important;
            border-right: 2px solid var(--primary-color);
            transform: translateX(0) !important;
            transition: none !important;
            visibility: visible !important;
            display: block !important;
        }
        
        /* 调整主内容区域，为固定的侧边栏留出空间 */
        .main .block-container {
            padding-top: 1rem;
            padding-left: 2rem;
            padding-right: 1rem;
            margin-left: 350px !important;
            background: var(--bg-primary);
        }
        
        /* 历史对话列表样式 */
        .stSidebar .stButton > button {
            margin-bottom: 4px !important;
            padding: 6px 12px !important;
            font-size: 0.9rem !important;
        }
        
        .stSidebar .stCaption {
            margin-top: 2px !important;
            margin-bottom: 8px !important;
            font-size: 0.8rem !important;
        }
        
        .stSidebar hr {
            margin: 8px 0 !important;
        }
        
        /* 隐藏Streamlit默认元素 */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        .stDeployButton {visibility: hidden;}
        
        .sidebar-title {
            text-align: center;
            margin-bottom: 1rem;
            font-size: 1.8rem;
            font-weight: 800;
            background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 50%, var(--accent-color) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            text-shadow: 0 4px 8px rgba(99, 102, 241, 0.3);
            letter-spacing: 1px;
            position: relative;
            animation: titleGlow 3s ease-in-out infinite alternate;
            padding: 8px 16px;
            border-radius: 12px;
            background-color: rgba(99, 102, 241, 0.05);
        }
        
        @keyframes titleGlow {
            0% {
                text-shadow: 0 4px 8px rgba(99, 102, 241, 0.3), 0 0 20px rgba(99, 102, 241, 0.2);
            }
            100% {
                text-shadow: 0 4px 8px rgba(99, 102, 241, 0.5), 0 0 30px rgba(139, 92, 246, 0.3);
            }
        }
        
        .sidebar-subtitle {
            text-align: center;
            margin-bottom: 1.5rem;
            font-size: 1rem;
            color: var(--text-secondary);
        }
        
        /* 消息样式 */
        .message-user {
            background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
            color: white;
            padding: 12px 16px;
            border-radius: 16px 16px 4px 16px;
            margin: 8px 0;
            max-width: 70%;
            margin-left: auto;
            margin-right: 0;
            word-wrap: break-word;
            box-shadow: 0 2px 8px rgba(99, 102, 241, 0.2);
        }
        
        .message-ai {
            background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
            color: var(--text-primary);
            padding: 12px 16px;
            border-radius: 16px 16px 16px 4px;
            margin: 8px 0;
            max-width: 70%;
            margin-left: 0;
            margin-right: auto;
            word-wrap: break-word;
            border: 1px solid var(--primary-light);
            box-shadow: 0 2px 8px rgba(99, 102, 241, 0.1);
        }
        
        .identifier-highlight {
            color: var(--primary-color);
            font-weight: bold;
            font-family: 'Courier New', monospace;
            font-size: 0.95em;
        }
        
        /* 按钮样式 */
        .stButton > button {
            background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 100%);
            color: white;
            border: none;
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.2s ease;
        }
        
        .stButton > button:hover {
            background: linear-gradient(135deg, var(--primary-dark) 0%, var(--accent-color) 100%);
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
        }
        
        /* 删除按钮特殊样式 */
        .stButton > button[title="删除对话"] {
            background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
            color: white;
            font-size: 16px;
            font-weight: bold;
            min-width: 32px;
            height: 32px;
            padding: 0;
        }
        
        .stButton > button[title="删除对话"]:hover {
            background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%);
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(239, 68, 68, 0.3);
        }
        
        /* 输入框样式 */
        .stChatInput > div > div > div > input {
            border: 2px solid var(--primary-light);
            border-radius: 12px;
            background: white;
        }
        
        .stChatInput > div > div > div > input:focus {
            border-color: var(--primary-color);
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
        }
        
        /* 欢迎界面样式 */
        .welcome-container {
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 50vh;
            padding: 2rem;
        }
        
        .welcome-content {
            text-align: center;
            max-width: 500px;
        }
        
        /* 标题样式 */
        h1, h2, h3, h4, h5, h6 {
            color: var(--primary-color);
        }
        
        /* 欢迎界面主标题特殊样式 */
        .welcome-content h1 {
            font-size: 2.5rem;
            font-weight: 800;
            background: linear-gradient(135deg, var(--primary-color) 0%, var(--secondary-color) 50%, var(--accent-color) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            text-shadow: 0 4px 8px rgba(99, 102, 241, 0.3);
            letter-spacing: 2px;
            margin-bottom: 1rem;
            animation: welcomeTitleGlow 4s ease-in-out infinite alternate;
        }
        
        @keyframes welcomeTitleGlow {
            0% {
                text-shadow: 0 4px 8px rgba(99, 102, 241, 0.3), 0 0 20px rgba(99, 102, 241, 0.2);
                transform: scale(1);
            }
            100% {
                text-shadow: 0 6px 12px rgba(99, 102, 241, 0.5), 0 0 40px rgba(139, 92, 246, 0.4);
                transform: scale(1.02);
            }
        }
        
        /* 欢迎界面副标题样式 */
        .welcome-subtitle {
            font-size: 1.2rem;
            font-weight: 600;
            background: linear-gradient(135deg, var(--text-secondary) 0%, var(--primary-color) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            text-shadow: 0 2px 4px rgba(99, 102, 241, 0.2);
            letter-spacing: 0.5px;
            margin-bottom: 1.5rem;
            animation: subtitleFade 2s ease-in-out infinite alternate;
        }
        
        /* 欢迎界面提示文字样式 */
        .welcome-hint {
            font-size: 1rem;
            font-weight: 500;
            color: var(--text-secondary);
            background: linear-gradient(135deg, var(--text-secondary) 0%, var(--primary-light) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
            text-shadow: 0 1px 2px rgba(99, 102, 241, 0.1);
            letter-spacing: 0.3px;
            line-height: 1.6;
            padding: 12px 20px;
            border-radius: 12px;
            background-color: rgba(99, 102, 241, 0.05);
            border: 1px solid rgba(99, 102, 241, 0.1);
            animation: hintPulse 3s ease-in-out infinite;
        }
        
        @keyframes subtitleFade {
            0% {
                opacity: 0.8;
                transform: translateY(0);
            }
            100% {
                opacity: 1;
                transform: translateY(-2px);
            }
        }
        
        @keyframes hintPulse {
            0% {
                box-shadow: 0 0 0 0 rgba(99, 102, 241, 0.1);
                transform: scale(1);
            }
            50% {
                box-shadow: 0 0 0 8px rgba(99, 102, 241, 0.05);
                transform: scale(1.01);
            }
            100% {
                box-shadow: 0 0 0 0 rgba(99, 102, 241, 0.1);
                transform: scale(1);
            }
        }
        
        /* 响应式调整 */
        @media (max-width: 768px) {
            section[data-testid="stSidebar"] {
                min-width: 280px !important;
                max-width: 280px !important;
            }
            .main .block-container {
                margin-left: 280px !important;
                padding-left: 1rem !important;
            }
        }
    </style>
    """, unsafe_allow_html=True)

# 业务逻辑
def get_source_content(source_id):
    """调用溯源查询函数"""
    try:
        # 解析引用格式，提取实际的ID
        if source_id.startswith("{'points':"):
            # 提取points中的ID
            matches = re.findall(r"\((\d+),'([0-9a-fA-F-]+)'\)", source_id)
            if matches:
                # 使用第一个匹配的ID，格式为 "1,community_id"
                community_id = matches[0][1]
                actual_id = f"1,{community_id}"
                content = get_source(actual_id)
                return content
        elif source_id.startswith("'Chunks':"):
            # 提取chunks中的ID
            matches = re.findall(r"'([0-9a-fA-F]+)'", source_id)
            if matches:
                # 使用第一个匹配的ID，格式为 "2,chunk_id"
                chunk_id = matches[0]
                actual_id = f"2,{chunk_id}"
                content = get_source(actual_id)
                return content
        
        # 如果无法解析，直接使用原始ID
        content = get_source(source_id)
        return content
    except Exception as e:
        return f"溯源查询出现错误：{str(e)}"

def sync_messages_to_agent(messages, session_id):
    """同步历史消息到Agent"""
    if not AGENT_AVAILABLE or not messages:
        return
    
    try:
        config = user_config(session_id)
        
        # 清除当前session的所有消息
        current_state = agent.get_state(config)
        if current_state and current_state.values.get("messages"):
            existing_messages = current_state.values["messages"]
            for message in reversed(existing_messages):
                agent.update_state(config, {"messages": [RemoveMessage(id=message.id)]})
        
        # 将历史消息转换为agent格式并添加
        agent_messages = []
        for msg in messages:
            if msg["role"] == "user":
                agent_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                agent_messages.append(AIMessage(content=msg["content"]))
        
        if agent_messages:
            agent.update_state(config, {"messages": agent_messages})
            
    except Exception as e:
        print(f"同步消息到agent时出错: {e}")
        # 如果同步失败，不影响正常使用，只是没有历史上下文

# 状态管理
def initialize_session_state():
    """初始化会话状态"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'input_key' not in st.session_state:
        st.session_state.input_key = 0
    if 'selected_tab' not in st.session_state:
        st.session_state.selected_tab = "知识图谱对话"
    if 'current_source_id' not in st.session_state:
        st.session_state.current_source_id = ""
    if 'current_source_content' not in st.session_state:
        st.session_state.current_source_content = "暂无溯源内容。"
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if 'should_show_traceability' not in st.session_state:
        st.session_state.should_show_traceability = False
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'current_chat_id' not in st.session_state:
        st.session_state.current_chat_id = None

def handle_source_click(source_id):
    """处理溯源点击"""
    st.session_state.current_source_id = source_id
    st.session_state.current_source_content = get_source_content(source_id)
    st.session_state.should_show_traceability = True
    st.rerun()

def create_new_chat():
    """创建新对话"""
    # 生成新的session_id，确保完全独立的对话上下文
    new_session_id = str(uuid.uuid4())
    
    # 如果agent模块可用，清除之前的对话历史
    if AGENT_AVAILABLE:
        try:
            clear_session(st.session_state.session_id)
        except Exception as e:
            st.warning(f"清除对话历史时出现警告：{str(e)}")
    
    # 更新session_id为新的ID
    st.session_state.session_id = new_session_id
    
    # 创建新的对话记录
    new_chat_id = str(uuid.uuid4())
    new_chat = {
        'id': new_chat_id,
        'title': '新对话',
        'messages': [],
        'created_at': time.time()
    }
    st.session_state.chat_history.append(new_chat)
    st.session_state.current_chat_id = new_chat_id
    st.session_state.messages = []
    st.session_state.should_show_traceability = False

def load_chat(chat_id):
    """加载对话"""
    for chat in st.session_state.chat_history:
        if chat['id'] == chat_id:
            # 为每个对话生成独立的session_id，确保上下文隔离
            if 'session_id' not in chat:
                chat['session_id'] = str(uuid.uuid4())
            
            st.session_state.current_chat_id = chat_id
            st.session_state.session_id = chat['session_id']
            st.session_state.messages = chat['messages']
            st.session_state.should_show_traceability = False
            
            # 同步历史消息到agent的对话历史中
            if AGENT_AVAILABLE and chat['messages']:
                try:
                    sync_messages_to_agent(chat['messages'], chat['session_id'])
                except Exception as e:
                    st.warning(f"同步对话历史时出现警告：{str(e)}")
            
            st.rerun()
            break

def save_current_chat():
    """保存对话"""
    if st.session_state.current_chat_id and st.session_state.messages:
        for chat in st.session_state.chat_history:
            if chat['id'] == st.session_state.current_chat_id:
                chat['messages'] = st.session_state.messages
                chat['session_id'] = st.session_state.session_id  # 保存session_id
                # 更新标题为第一条用户消息
                if st.session_state.messages:
                    first_user_msg = next((msg for msg in st.session_state.messages if msg['role'] == 'user'), None)
                    if first_user_msg:
                        chat['title'] = first_user_msg['content'][:30] + '...' if len(first_user_msg['content']) > 30 else first_user_msg['content']
                break

def delete_chat(chat_id):
    """删除对话"""
    st.session_state.chat_history = [chat for chat in st.session_state.chat_history if chat['id'] != chat_id]
    if st.session_state.current_chat_id == chat_id:
        st.session_state.current_chat_id = None
        st.session_state.messages = []
    st.rerun()

# 界面组件
def render_sidebar():
    """渲染侧边栏"""
    with st.sidebar:
        # 侧边栏标题
        st.markdown('<div class="sidebar-title">GraphRAG</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-subtitle">知识图谱智能问答</div>', unsafe_allow_html=True)
        st.markdown("---")
        
        # 新建对话按钮
        if st.button("新建对话", use_container_width=True, type="primary"):
            create_new_chat()
        
        st.markdown("---")
        
        # 历史对话列表
        st.subheader("历史对话")
        
        if st.session_state.chat_history:
            for chat in reversed(st.session_state.chat_history):
                is_active = chat['id'] == st.session_state.current_chat_id
                
                # 创建列布局
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    # 显示对话标题和基本信息
                    display_title = f"{chat['title'][:20]}{'...' if len(chat['title']) > 20 else ''}"
                    
                    if st.button(
                        display_title,
                        key=f"load_{chat['id']}",
                        use_container_width=True,
                        type="primary" if is_active else "secondary"
                    ):
                        load_chat(chat['id'])
                
                with col2:
                    if st.button("×", key=f"delete_{chat['id']}", help="删除对话"):
                        delete_chat(chat['id'])
        else:
            st.info("暂无历史对话")
        
        # 使用说明
        with st.expander("使用说明", expanded=True):
            st.write("""
            **问答类型：**
            - 全局性查询：整体情况
            - 局部性查询：具体实体  
            - 普通咨询：系统介绍
            
            **溯源功能：**
            点击AI回复中的'查验引用'查看来源
            """)

def render_main_content():
    """渲染主内容"""
    # 检查是否需要显示溯源查验界面
    if st.session_state.should_show_traceability:
        render_traceability_tab()
    else:
        render_chat_interface()

def render_chat_interface():
    """渲染聊天界面"""
    # 显示对话历史
    if st.session_state.messages:
        # 创建聊天容器
        for i, message in enumerate(st.session_state.messages):
            if message["role"] == "user":
                st.markdown(f'<div class="message-user">{message["content"]}</div>', unsafe_allow_html=True)
            else:
                render_agent_message(message, i)
    else:
        # 简化的欢迎界面
        st.markdown("""
        <div class="welcome-container">
            <div class="welcome-content">
                <h1 style="text-align: center;">欢迎使用 GraphRAG</h1>
                <p class="welcome-subtitle" style="text-align: center;">基于知识图谱的智能问答系统</p>
                <div style="text-align: center; margin: 2rem 0;">
                    <p class="welcome-hint">在下方输入框中输入您的问题，开始与知识图谱对话</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # 输入区域
    user_input = st.chat_input("输入您的问题，按回车发送...", key=f"input_{st.session_state.input_key}")
    
    if user_input:
        # 如果没有当前对话，创建新对话
        if not st.session_state.current_chat_id:
            create_new_chat()
        else:
            save_current_chat()
        
        # 添加用户消息
        st.session_state.messages.append({"role": "user", "content": user_input})
        st.session_state.input_key += 1
        
        # 保存用户输入到session_state中，以便在重新运行后使用
        st.session_state.pending_user_input = user_input
        
        # 重新运行以显示用户消息
        st.rerun()
    
    # 检查是否有待处理的用户输入
    if hasattr(st.session_state, 'pending_user_input') and st.session_state.pending_user_input:
        user_input = st.session_state.pending_user_input
        # 清除待处理的输入
        del st.session_state.pending_user_input
        
        # 获取AI响应
        with st.spinner("知识图谱正在思考..."):
            try:
                response = ask_agent_with_source(user_input, st.session_state.session_id)
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                # 保存当前对话
                save_current_chat()
                
                # 重新运行以显示AI回复
                st.rerun()
            except Exception as e:
                st.error(f"处理请求时出现错误：{str(e)}")

def render_traceability_tab():
    """渲染溯源界面"""
    # 溯源查验界面
    st.title("溯源查验")
    st.markdown("查看知识图谱中的原始数据来源")
    
    # 返回按钮
    if st.button("返回对话", type="primary"):
        st.session_state.should_show_traceability = False
        st.rerun()
    
    st.markdown("---")
    
    # 溯源信息
    if st.session_state.current_source_id:
        st.success("溯源信息已加载")
        
        # 溯源ID
        st.subheader("溯源ID")
        st.code(st.session_state.current_source_id, language="text")
        
        # 溯源内容
        st.subheader("溯源内容")
        st.text_area(
            "溯源内容", 
            value=st.session_state.current_source_content, 
            height=400,
            disabled=True,
            label_visibility="collapsed"
        )
    else:
        st.info("暂无溯源信息")
        st.markdown("点击AI回复中的'查验引用'按钮查看知识图谱来源信息")

def render_agent_message(msg, index):
    """渲染AI消息"""
    display_content = msg["content"]
    
    # 定义各种标识符的正则表达式模式
    patterns = [
        # 完整的data结构
        (r"\{'data':\s*\{[^}]*\}\s*\{[^}]*\}\}", "data_structure"),
        # points格式
        (r"\{'points':\[(.*?)\]\}", "points"),
        # Chunks格式
        (r"'Chunks':\[(.*?)\]", "chunks"),
        # Entities格式
        (r"'Entities':\[(.*?)\]", "entities"),
        # Reports格式
        (r"'Reports':\[(.*?)\]", "reports"),
        # Relationships格式
        (r"'Relationships':\[(.*?)\]", "relationships"),
        # 单独的chunk ID
        (r"'([0-9a-fA-F]{40})'", "chunk_id"),
        # 数字ID
        (r"\b(\d+)\b", "number_id")
    ]
    
    # 收集所有匹配项
    all_matches = []
    for pattern, match_type in patterns:
        matches = list(re.finditer(pattern, display_content))
        for match in matches:
            all_matches.append((match.start(), match.end(), match.group(0), match_type))
    
    # 按位置排序
    all_matches.sort(key=lambda x: x[0])
    
    # 分割文本并处理引用
    last_end = 0
    parts = []
    for start, end, match_text, match_type in all_matches:
        # 添加前面的文本
        parts.append(display_content[last_end:start])
        
        # 根据类型添加不同的样式
        if match_type in ["data_structure", "points", "chunks", "entities", "reports", "relationships", "chunk_id", "number_id"]:
            # 所有标识符都使用蓝色加粗样式
            parts.append(f'<span class="identifier-highlight">{match_text}</span>')
        else:
            # 其他情况保持加粗
            parts.append(f'**{match_text}**')
        
        last_end = end
    
    # 添加最后一部分文本
    parts.append(display_content[last_end:])
    
    # 渲染消息内容
    formatted_content = "".join(parts)
    st.markdown(f'<div class="message-ai">{formatted_content}</div>', unsafe_allow_html=True)
    
    # 为溯源相关的引用添加按钮
    traceability_matches = [m for m in all_matches if m[3] in ["points", "chunks", "data_structure"]]
    for i, (start, end, match_text, match_type) in enumerate(traceability_matches):
        if st.button(f"查验引用 {i+1}", key=f"source_{index}_{i}"):
            handle_source_click(match_text)

if __name__ == "__main__":
    try:
        # 初始化配置
        setup_page_config()
        setup_custom_styles()
        initialize_session_state()
        
        # 渲染界面
        render_sidebar()
        render_main_content()
        
    except Exception as e:
        st.error(f"应用运行错误: {str(e)}")