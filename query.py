from agent import agent, ask_agent, get_answer

def main():
    # 配置参数
    config = {"configurable": {"thread_id": "test_thread", "recursion_limit": 5}}
    print("输入 'quit' 退出程序")
    
    while True:
        try:
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
            answer = get_answer(config)
            print(f"\n最终答案:\n{answer}")
            print("=" * 50)
            
        except KeyboardInterrupt:
            print("\n\n程序被用户中断")
            break
        except Exception as e:
            print(f"\n处理问题时出错: {e}")
            continue

if __name__ == "__main__":
    main()
