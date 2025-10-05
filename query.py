from my_packages.AgentAbout import create_agent, user_config, ask_agent, get_answer
from langgraph.checkpoint.memory import MemorySaver

if __name__ == "__main__":
    # 配置参数
    memory = MemorySaver()
    config = user_config()
    agent = create_agent(memory)
    print("输入 'quit' 退出程序")
    
    while True:
        # 获取用户输入
        query = input("\n请输入您的问题: ").strip()
        
        if query.lower() in ['quit', 'exit', '退出']:
            print("再见！")
            break
            
        if not query:
            print("请输入有效的问题")
            continue
        
        print(f"\n正在处理问题: {query}")
        print("-" * 50)
        
        # 向agent提问
        ask_agent(query, agent, config)
        
        # 获取最终答案
        answer = get_answer(memory, config)
        print(f"\n最终答案:\n{answer}")
        print("=" * 50)
