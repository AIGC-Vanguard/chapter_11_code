from application import qa_bot


def test_bot():
    config = qa_bot.QABotConfig(embedding_model_name="../embed_model/shibing624-text2vec-base-chinese",
                                docs_path="../test_data/shoes_data.tsv",
                                vector_store_path = "../test_data",
                                history_len=5)
    bot = qa_bot.QABot(config)
    result = bot.get_knowledge_based_answer(llm_model="ChatGPT",
                                            query="价格是多少",
                                            goods_id="145",
                                            top_k=1)
    print(result)


if __name__ == '__main__':
    test_bot()