from dataclasses import dataclass
from kb_service import kb_service
from llm_tools import ChatGPTService,GLMService


@dataclass
class QABotConfig:
    embedding_model_name:str  # embedding模型名称
    docs_path:str  # 本地知识库路径
    vector_store_path:str  # 向量数据库本地存储路径
    history_len:int  # 保存的历史对话数量


class QABot(object):
    def __init__(self, config:QABotConfig):
        self.config = config

        self.prompt_template = """
                你目前是电商客服，你的职责是回答用户对于商品的问题。
                要求：
                1. 基于已知信息，简洁和专业的来回答用户的问题。
                2. 如果无法从中得到答案，请说 "对不起，我目前无法回答你的问题"，不允许在答案中添加编造成分。
                已知信息：
                {context}
                用户问题:                               
                {question}
                回答:
                """

        self.source_service = kb_service.KBSearchService(config)
        # self.llm_model_dict = {"ChatGPT":ChatGPTService}
        self.llm_model_dict = {"ChatGPT":ChatGPTService,"ChatGLM-6B":GLMService}


    def get_knowledge_based_answer(self,
                                   query,
                                   llm_model,
                                   goods_id,
                                   top_k,
                                   chat_history=[]):
        """
        根据query返回答案。
        :param query: 用户问题
        :param llm_model: 大模型服务
        :param goods_id: 所咨询的商品id
        :param top_k: 返回知识条数
        :param chat_history: 对话历史记录
        :return: 问题答案
        """
        # 数据库查询结果
        db_result = self.source_service.knowledge_search(goods_id,query,top_k)
        # 模型调用结果
        llm_service = self.llm_model_dict[llm_model]
        history = chat_history[-self.config.history_len:] if self.config.history_len > 0 else []
        prompt = self.prompt_template.format(context=db_result,question=query)
        result = llm_service.get_ans(prompt,history)

        return result,db_result
