from langgraph.checkpoint.memory import InMemorySaver
import streamlit as st
import time
import uuid
import re

from my_packages.AgentAbout import create_agent, user_config, ask_agent, get_answer
from my_packages.QueryAbout import get_source

# 页面配置
def setup_page_config():
    """配置页面基本设置"""
    try:
        st.set_page_config(
            page_title="GraphRAG - 知识图谱智能问答",
            layout="wide",
            initial_sidebar_state="expanded",
            menu_items={
                'Get Help': 'https://github.com/Hikari3939/LLM_KG',
                'Report a bug': 'https://github.com/Hikari3939/LLM_KG/issues',
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

# 状态管理
def initialize_session_state(agent, memory):
    """初始化会话状态"""
    # Agent部分
    if 'agent' not in st.session_state:
        st.session_state.agent = agent
    if 'memory' not in st.session_state:
        st.session_state.memory = memory
    # 对话部分
    if 'current_config' not in st.session_state:
        st.session_state.current_config = {}
    if 'current_messages' not in st.session_state:
        st.session_state.current_messages = []
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    # 溯源部分
    if 'current_source_type' not in st.session_state:
        st.session_state.current_source_type = ""
    if 'current_source_content' not in st.session_state:
        st.session_state.current_source_content = "暂无溯源内容。"
    if 'show_traceability' not in st.session_state:
        st.session_state.show_traceability = False

# 业务实现
def deal_input(user_input):
    '''处理用户输入'''
    # 如果没有当前对话，创建新对话
    if not st.session_state.current_config:
        create_new_chat()
    
    # 添加用户消息
    st.session_state.current_messages.append({"role": "user", "content": user_input})
    
    # 立即保存用户消息
    save_current_chat()
    
    # 获取AI响应
    with st.spinner("Agent正在思考..."):
        ask_agent(user_input, st.session_state.agent, st.session_state.current_config)
        response = get_answer(st.session_state.memory, st.session_state.current_config)
        st.session_state.current_messages.append({"role": "assistant", "content": response})

    # 保存当前对话
    save_current_chat()

def deal_trace(ai_message):
    """处理溯源"""
    # 定义标识符的正则表达式模式
    patterns = [
        # CommunityIds格式
        (r"'CommunityIds':\[(.*?)\]", "Community"),
        # Chunks格式
        (r"'Chunks':\[(.*?)\]", "Chunk"),
    ]
    
    # 收集所有匹配项
    all_matches = []
    for pattern, match_type in patterns:
        matches = re.findall(pattern, ai_message)
        if matches:
            all_matches.append((match_type, matches))
    
    return all_matches

def create_new_chat():
    """创建新对话"""
    # 生成新的config，确保完全独立的对话上下文
    new_session_id = str(uuid.uuid4())
    new_config = user_config(new_session_id)
        
    # 创建新的对话记录
    new_chat = {
        'config': new_config,
        'title': '新对话',
        'messages': [],
        'created_at': time.time()
    }
    st.session_state.current_config = new_config
    st.session_state.current_messages = []
    st.session_state.chat_history.append(new_chat)

def save_current_chat():
    """保存当前对话"""
    for chat in st.session_state.chat_history:
        if chat['config'] == st.session_state.current_config:
            chat['messages'] = st.session_state.current_messages
            # 更新标题为第一条用户消息
            if st.session_state.current_messages:
                first_user_msg = next((msg for msg in st.session_state.current_messages if msg['role'] == 'user'), None)
                if first_user_msg:
                    chat['title'] = first_user_msg['content'][:30] + '...' if len(first_user_msg['content']) > 30 else first_user_msg['content']
            break

def load_chat(chat_config):
    """加载对话"""
    for chat in st.session_state.chat_history:
        if chat['config'] == chat_config:            
            st.session_state.current_config = chat['config']
            st.session_state.current_messages = chat['messages']
            break

def delete_chat(chat_config):
    """删除对话"""
    # 删除历史
    st.session_state.chat_history = [
        chat for chat in st.session_state.chat_history if chat['config'] != chat_config
    ]
    # 删除检查点
    st.session_state.memory.delete_thread(chat_config['configurable']['thread_id'])
    
    if st.session_state.current_config == chat_config:
        st.session_state.current_config = {}
        st.session_state.current_messages = []

def trace_source(source_ids):
    """查询来源"""
    if isinstance(source_ids, list):
        matches = []
        for source_id in source_ids:
            matches.extend(re.findall(r"'([0-9a-fA-F-]+)'", source_id))
        matches = list(set(matches))
    else:
        matches = re.findall(r"'([0-9a-fA-F]+)'", source_ids)
    
    content = ""
    for id in matches:
        content += get_source(id)
    return content

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
            st.session_state.current_config = {}
            st.session_state.current_messages = []
            st.session_state.show_traceability = False
            st.rerun()
        
        st.markdown("---")
        
        # 历史对话列表
        st.subheader("历史对话")
        
        if st.session_state.chat_history:
            for chat in reversed(st.session_state.chat_history):
                is_active = chat['config'] == st.session_state.current_config
                
                # 创建列布局
                col1, col2 = st.columns([4, 1])
                thread_id = chat['config']['configurable']['thread_id']
                
                with col1:
                    # 显示对话标题和基本信息
                    display_title = f"{chat['title'][:20]}{'...' if len(chat['title']) > 20 else ''}"
                    
                    if st.button(
                        display_title,
                        key=thread_id,
                        use_container_width=True,
                        type="primary" if is_active else "secondary"
                    ):
                        load_chat(chat['config'])
                        st.session_state.show_traceability = False
                        st.rerun()
                
                with col2:
                    if st.button(
                        "×",
                        key="delete_"+thread_id,
                        help="删除对话"
                    ):
                        delete_chat(chat['config'])
                        st.session_state.show_traceability = False
                        st.rerun()
        else:
            st.info("暂无历史对话")
        
        # 使用说明
        st.subheader("使用说明")
        st.info(
            """
            **问答类型：**
            - 全局性查询：整体情况
            - 局部性查询：具体实体  
            - 普通咨询：系统介绍
            
            **溯源功能：**
            点击AI回复中的'查验引用'查看来源
            """
        )

def render_main_content():
    """渲染主内容"""
    # 检查是否需要显示溯源查验界面
    if st.session_state.show_traceability:
        render_traceability_tab()
    else:
        render_chat_interface()

def render_chat_interface():
    """渲染聊天界面"""
    # 显示对话历史
    if st.session_state.current_messages:
        # 创建聊天容器
        for i, message in enumerate(st.session_state.current_messages):
            if message["role"] == "user":
                st.markdown(f'<div class="message-user">{message["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="message-ai">{message["content"]}</div>', unsafe_allow_html=True)
                all_matches = deal_trace(message["content"])
                if all_matches:
                    # 溯源按钮
                    thread_id = st.session_state.current_config['configurable']['thread_id']
                    if st.button(
                        "查验引用",
                        key=f"trace_{thread_id}_{i}",  # 使用索引i来构造key
                        type="primary"
                    ):
                        match_type, matches = all_matches[0]
                        st.session_state.current_source_type = match_type
                        st.session_state.current_source_content = trace_source(matches)
                        st.session_state.show_traceability = True
                        st.rerun()
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
    user_input = st.chat_input("输入您的问题，按回车发送...")
    if user_input:
        # 立即显示用户消息
        with st.container():
            st.markdown(f'<div class="message-user">{user_input}</div>', unsafe_allow_html=True)
        
        # 处理用户输入
        deal_input(user_input)
        st.rerun()

def render_traceability_tab():
    """渲染溯源界面"""
    # 溯源查验界面
    st.title("溯源查验")
    st.markdown("查看知识图谱中的原始数据来源")
    
    # 返回按钮
    if st.button("返回对话", type="primary"):
        st.session_state.show_traceability = False
        st.rerun()
    
    st.markdown("---")
        
    # 溯源信息
    if st.session_state.current_source_type:
        # 溯源内容
        st.subheader("溯源内容")
        st.markdown(
            st.session_state.current_source_content
        )
    else:
        st.info("暂无溯源信息")

if __name__ == "__main__":
    # 初始化agent
    memory = InMemorySaver()
    agent = create_agent(memory)
    
    # 初始化配置
    setup_page_config()
    setup_custom_styles()
    initialize_session_state(agent, memory)

    # 渲染界面
    render_sidebar()
    render_main_content()
