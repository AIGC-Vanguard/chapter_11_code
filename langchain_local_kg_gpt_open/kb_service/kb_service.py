import os
from collections import defaultdict
import pandas as pd
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from chinese_text_splitter import SentencesTextSplitter
from chromadb.config import Settings


class KBSearchService(object):
    def __init__(self, config):
        self.vector_store = {}
        self.config = config
        self.embedding_model = HuggingFaceEmbeddings(model_name=self.config.embedding_model_name)
        self.docs_path = self.config.docs_path
        self.vector_store_path = self.config.vector_store_path
        self.init_source_vector()

    def corpus_preprocess(self):
        """
        数据预处理，切分并处理成方便导入向量数据库的格式
        :return: 处理好的数据，字典格式，key为商品ID，value是该商品下切分好的文本
        """
        text_splitter = SentencesTextSplitter(max_chunk_size=100)
        texts_list = defaultdict(list)

        data_corpus = pd.read_csv(self.docs_path, sep="\t")
        for index, row in data_corpus.iterrows():
            sentence = row['句子']
            goods_id = row['商品_ID']
            texts_split = text_splitter.split_text(sentence)
            for item in texts_split:
                texts_list[goods_id].append(item)

        return texts_list

    def init_source_vector(self):
        """
        将知识库数据导入向量数据库
        """
        data = self.corpus_preprocess()
        for goods_id, texts in data.items():
            goods_id = str(goods_id)
            print(goods_id)
            vs = Chroma.from_texts(
                texts=texts,
                embedding=self.embedding_model,
                collection_name=goods_id,
                client_settings=Settings(persist_directory=os.path.join(self.vector_store_path, goods_id))
            )
            self.vector_store[goods_id] = vs

    def knowledge_search(self,goods_id,question,top_k):
        db = self.vector_store[goods_id]
        search_result = db.similarity_search(question, k=top_k)
        doc = [r.page_content for r in search_result]
        return "\n".join(doc)



