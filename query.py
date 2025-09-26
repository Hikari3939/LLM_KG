from my_packages.QueryAbout import local_retriever, global_retriever

if __name__ == "__main__":
    while True:
        query = input("\n请输入问题 (输入 'exit' 退出): ").strip()
        
        if query.lower() == 'exit':
            break
            
        if not query:
            print("问题不能为空，请重新输入")
            continue
            
        print(f"\n查询问题: {query}")
        
        # 局部检索
        print("\n局部检索结果:")
        print("-" * 30)
        try:
            local_response = local_retriever(query)
            print("回答:", local_response)
        except Exception as e:
            print(f"局部检索失败: {str(e)}")
        
        # 全局检索
        print("\n全局检索结果:")
        print("-" * 30)
        try:
            global_response = global_retriever(query, level=0)
            print("回答:", global_response)
        except Exception as e:
            print(f"全局检索失败: {str(e)}")
