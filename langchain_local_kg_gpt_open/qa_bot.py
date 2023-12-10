from application.qa_bot import LangChainApplication

# 修改成自己的配置！！！
class LangChainCFG:
    llm_model_name = 'THUDM/chatglm-6b-int4-qe'  # 本地模型文件 or huggingface远程仓库
    embedding_model_name = 'GanymedeNil/text2vec-large-chinese'  # 检索模型文件 or huggingface远程仓库
    vector_store_path = './cache'
    docs_path = './docs'
    kg_vector_stores = {
        '中文维基百科': './cache/zh_wikipedia',
        '大规模金融研报': './cache/financial_research_reports',
        '初始化': './cache',
    }  # 可以替换成自己的知识库，如果没有需要设置为None
    # kg_vector_stores=None
    patterns = ['模型问答', '知识库问答']  #
    n_gpus=1


config = LangChainCFG()
application = LangChainApplication(config)

application.source_service.init_source_vector()


def predict(input,
            large_language_model,
            embedding_model,
            top_k,
            use_web,
            use_pattern,
            history=None):
    # print(large_language_model, embedding_model)
    print(input)
    if history == None:
        history = []

    if use_web == '使用':
        web_content = application.source_service.search_web(query=input)
    else:
        web_content = ''
    search_text = ''
    if use_pattern == '模型问答':
        result = application.get_llm_answer(query=input, web_content=web_content)
        history.append((input, result))
        search_text += web_content
        return '', history, history, search_text

    else:
        resp = application.get_knowledge_based_answer(
            query=input,
            history_len=1,
            temperature=0.1,
            top_p=0.9,
            top_k=top_k,
            web_content=web_content,
            chat_history=history
        )
        history.append((input, resp['result']))
        for idx, source in enumerate(resp['source_documents'][:4]):
            sep = f'----------【搜索结果{idx + 1}：】---------------\n'
            search_text += f'{sep}\n{source.page_content}\n\n'
        print(search_text)
        search_text += "----------【网络检索内容】-----------\n"
        search_text += web_content
        return '', history, history, search_text