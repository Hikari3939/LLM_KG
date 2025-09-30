import streamlit as st
import time
import re

# 配置与样式设置
def setup_page_config():
    """配置页面基本设置"""
    st.set_page_config(
        page_title="脑卒中智能问答系统",
        layout="wide",
        initial_sidebar_state="collapsed"
    )

def setup_custom_styles():
    """设置自定义CSS样式"""
    st.markdown("""
    <style>
        /* 用户消息气泡样式 */
        .user-msg {
            background-color: #DCF8C6;
            padding: 12px 16px;
            border-radius: 18px 18px 0 18px;
            margin: 6px 0;
            max-width: 80%;
            margin-left: auto;
            display: block;
            line-height: 1.5;
            box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        }
        
        /* 助手消息气泡样式 */
        .agent-msg {
            background-color: #FFFFFF;
            padding: 12px 16px;
            border-radius: 18px 18px 18px 0;
            margin: 6px 0;
            max-width: 80%;
            border: 1px solid #eee;
            display: block;
            line-height: 1.5;
            box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        }
        
        /* 欢迎框样式 */
        .welcome-box {
            text-align: center;
            color: #666;
            font-size: 16px;
            padding: 20px;
        }
        
        .welcome-box h3 {
            color: #333;
            margin-bottom: 10px;
            font-size: 18px;
        }
        
        /* 溯源按钮样式 */
        .stButton>button {
            background-color: transparent !important;
            color: #1a73e8 !important;
            border: none !important;
            padding: 0 !important;
            font-size: inherit !important;
            text-decoration: underline !important;
            cursor: pointer !important;
            margin-left: 5px;
            display: inline-block;
        }
        
        .stButton>button:hover {
            color: #0d47a1 !important;
        }

        /* 页脚样式 */
        .footer {
            font-size: 12px;
            color: #999;
            text-align: center;
            margin-top: 20px;
        }
    </style>
    """, unsafe_allow_html=True)

# 业务逻辑函数
def mock_ask_agent(prompt):
    """
    模拟Agent响应函数
    在实际应用中，这里会调用真实的GraphRAG系统
    """
    time.sleep(1)  # 模拟网络延迟
    
    # 脑卒中相关的响应映射
    responses = {
        "你好": "您好！我是基于脑卒中医学知识图谱的智能问答助手，请问您想了解脑卒中的哪些方面？",
        "脑卒中的主要症状": "脑卒中的主要症状包括：突发性面部歪斜、单侧肢体无力、言语不清、视力模糊等。[引用:脑卒中症状]",
        "脑卒中的危险因素": "脑卒中的主要危险因素包括：高血压、糖尿病、高血脂、吸烟、肥胖等。[引用:危险因素]",
        "脑卒中的急救措施": "发现脑卒中症状应立即拨打急救电话，保持患者平卧，解开衣领，保持呼吸道通畅。[引用:急救指南]",
        "脑卒中的康复治疗": "脑卒中康复包括物理治疗、作业治疗、言语治疗等，应在医生指导下进行系统康复训练。[引用:康复方案]",
    }
    
    return responses.get(
        prompt.strip(), 
        f"关于'{prompt}'的问题，我将在脑卒中知识图谱中为您查找相关信息。"
    )

def mock_get_source(source_id):
    """
    模拟溯源查询函数
    在实际应用中，这里会查询知识图谱数据库
    """
    time.sleep(0.5)  # 模拟查询延迟
    
    # 脑卒中相关的溯源内容
    sources = {
        "脑卒中症状": """脑卒中典型症状（FAST原则）：
- F（Face）：面部不对称，口角歪斜
- A（Arm）：手臂无力，抬起困难  
- S（Speech）：言语不清，表达困难
- T（Time）：立即就医，争取黄金治疗时间

其他症状可能包括：突发性头痛、眩晕、平衡障碍、视力模糊等。""",
        
        "危险因素": """脑卒中可干预的危险因素：
1. 高血压：最重要的危险因素
2. 心脏病：如房颤、冠心病
3. 糖尿病：增加卒中风险2-4倍
4. 血脂异常：低密度脂蛋白升高
5. 吸烟：使卒中风险增加2-4倍
6. 肥胖和缺乏运动""",
        
        "急救指南": """脑卒中急救黄金时间窗：
- 缺血性脑卒中：发病4.5小时内可进行静脉溶栓
- 出血性脑卒中：需立即手术干预

急救步骤：
1. 立即拨打120急救电话
2. 记录发病时间
3. 保持患者平卧位，头偏向一侧
4. 不要随意给药或进食
5. 准备就医所需的证件和资料""",
        
        "康复方案": """脑卒中康复分期：
1. 急性期康复（发病2周内）：床上体位摆放、被动活动
2. 恢复期康复（发病后2周-6个月）：功能训练、ADL训练
3. 后遗症期康复（发病6个月后）：社区康复、家庭改造

康复内容包括：
- 运动功能训练
- 言语吞咽训练  
- 认知功能训练
- 心理康复支持
- 日常生活能力训练"""
    }
    
    return sources.get(source_id, "在脑卒中知识图谱中没有检索到该语料。")

# 会话状态管理
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

def handle_source_click(source_id):
    """
    处理溯源点击事件
    切换到溯源查验标签页并加载对应内容
    """
    st.session_state.current_source_id = source_id
    st.session_state.current_source_content = mock_get_source(source_id)
    st.session_state.selected_tab = "溯源查验"
    st.rerun()

# 界面组件
def render_header():
    """渲染页面头部"""
    st.markdown("<h3 style='text-align: center;'>脑卒中智能问答系统</h3>", unsafe_allow_html=True)

def render_system_instructions():
    """渲染系统使用说明"""
    with st.expander("系统使用说明与示例 (点击展开/折叠)"):
        st.markdown("""
        **系统介绍**
        本GraphRAG系统基于专业的脑卒中医学知识图谱构建，能够准确回答关于脑卒中预防、症状、诊断、治疗和康复等方面的问题。
        
        **问答类型示例：**
        1. **全局性查询**（使用社区摘要回答）：
           - `脑卒中的主要危险因素有哪些？`
           - `脑卒中患者康复治疗的基本原则是什么？`
           
        2. **局部性查询**（检索相关材料回答）：
           - `脑卒中急性期的溶栓治疗有什么禁忌症？`
           - `脑卒中后言语障碍如何进行康复训练？`
           
        3. **普通咨询**（无需检索）：
           - `你好，请介绍一下这个系统`
           
        **溯源查验功能**
        回答中引用的知识图谱内容会标注来源，点击可以查看详细的医学依据和参考资料。
        
        **重要提示**
        本系统提供的医学信息仅供参考，不能替代专业医生的诊断和治疗建议。
        """)

def render_chat_history():
    """渲染聊天记录"""
    with st.container(height=400, border=True):
        if len(st.session_state.messages) == 0:
            render_welcome_message()
        else:
            render_message_history()

def render_welcome_message():
    """渲染欢迎消息"""
    st.markdown("""
    <div class="welcome-box">
        <h3>欢迎使用脑卒中智能问答系统</h3>
        本系统基于专业的脑卒中医学知识图谱，为您提供准确的医学知识解答。
        
            尝试提问:
            1. 脑卒中的主要症状有哪些？
            2. 脑卒中患者的康复过程是怎样的？
            3. 如何预防脑卒中？
    </div>
    """, unsafe_allow_html=True)

def render_message_history():
    """渲染消息历史记录"""
    for i, msg in enumerate(st.session_state.messages):
        if msg['role'] == 'user':
            st.markdown(f'<div class="user-msg"><b>您</b>: {msg["content"]}</div>', unsafe_allow_html=True)
        else:
            render_agent_message(msg, i)

def render_agent_message(msg, index):
    """渲染助手消息，处理溯源引用"""
    display_content = msg["content"]
    matches = list(re.finditer(r'\[引用:(.*?)]', display_content))
    
    # 分割文本并处理引用
    last_end = 0
    parts = []
    for match in matches:
        parts.append(display_content[last_end:match.start()])
        source_id = match.group(1)
        parts.append(f'<span style="display:inline-block;">')
        parts.append(f'__ST_BUTTON_PLACEHOLDER_{index}_{source_id}__')
        parts.append(f'</span>')
        last_end = match.end()
    parts.append(display_content[last_end:])
    
    # 渲染消息内容
    formatted_content = "".join(parts)
    st.markdown(f'<div class="agent-msg"><b>助手</b>: {formatted_content}</div>', unsafe_allow_html=True)
    
    # 渲染溯源按钮
    for match in matches:
        source_id = match.group(1)
        st.button(f"查验 '{source_id}'", key=f"source_btn_{index}_{source_id}",
                  on_click=handle_source_click, args=(source_id,))

def render_input_area():
    """渲染输入区域"""
    st.markdown('<div style="margin-top: 10px; display: flex; gap: 10px;">', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([6, 2, 2])
    
    with col1:
        user_input = st.text_input(
            "输入您的问题...",
            key=f"input_box_{st.session_state.input_key}",
            label_visibility="collapsed",
            placeholder="例如：脑卒中患者如何进行康复训练？"
        )
    
    with col2:
        submit = st.button("发送", type="primary", use_container_width=True)
    
    with col3:
        clear = st.button("清空对话", use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    return user_input, submit, clear

def render_traceability_tab():
    """渲染溯源查验标签页"""
    st.markdown("<h4><b>溯源查验</b></h4>", unsafe_allow_html=True)
    st.markdown("查看的知识图谱来源：")
    
    # 显示溯源ID
    st.text_input(
        "来源ID", 
        value=st.session_state.current_source_id, 
        key="source_id_display", 
        disabled=True, 
        label_visibility="collapsed"
    )
    
    # 显示溯源内容
    st.markdown(
        f'<div style="width:100%;height:300px;overflow-y:auto;border:1px solid #ddd;padding:10px;margin-top:10px;">'
        f'<p style="white-space: pre-wrap;">{st.session_state.current_source_content}</p>'
        f'</div>', 
        unsafe_allow_html=True
    )

def render_footer():
    """渲染页脚"""
    st.markdown(
        "<div class='footer'>脑卒中智能问答系统 · 基于专业医学知识图谱构建 · 医学信息仅供参考</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    # 初始化配置
    setup_page_config()
    setup_custom_styles()
    initialize_session_state()
    
    # 渲染界面组件
    render_header()
    
    # 创建标签页
    tab_dialogue, tab_traceability = st.tabs(["知识图谱对话", "溯源查验"])
    
    # 对话标签页
    with tab_dialogue:
        render_system_instructions()
        st.markdown("<h4><b>对话记录</b></h4>", unsafe_allow_html=True)
        render_chat_history()
        
        # 输入区域和处理逻辑
        user_input, submit, clear = render_input_area()
        
        # 处理用户输入
        if submit and user_input.strip():
            st.session_state.messages.append({"role": "user", "content": user_input})
            st.session_state.input_key += 1
            
            with st.spinner("助手正在查询知识图谱..."):
                response = mock_ask_agent(user_input)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()
        
        # 处理清空对话
        if clear:
            st.session_state.messages = []
            st.session_state.input_key += 1
            st.session_state.current_source_id = ""
            st.session_state.current_source_content = "暂无溯源内容。"
            st.rerun()
    
    # 溯源查验标签页
    with tab_traceability:
        render_traceability_tab()
    
    # 页脚
    render_footer()
