import os
import pandas as pd
from langchain.vectorstores import Chroma
from chromadb.config import Settings
from chinese_text_splitter import SentencesTextSplitter
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from collections import defaultdict


def corpus_preprocess(data_path):
    """
    数据预处理，切分并处理成方便导入向量数据库的格式
    :param data_path: 知识库数据地址
    :return: 处理好的数据，字典格式，key为商品ID，value是该商品下切分好的文本
    """
    text_splitter = SentencesTextSplitter(max_chunk_size=100)
    texts_list = defaultdict(list)

    data_corpus = pd.read_csv(data_path,sep="\t")
    for index,row in data_corpus.iterrows():
        sentence = row['句子']
        goods_id = row['商品_ID']
        texts_split = text_splitter.split_text(sentence)
        for item in texts_split:
            texts_list[goods_id].append(item)

    return texts_list


def data_save_to_chroma(db_path, data, embedding_model):
    """
    将处理好的数据导入向量数据库
    :param db_path:数据吃就化存储路径
    :param data:要存储的数据，corpus_preprocess的输出
    :param embedding_model:选择的向量化模型
    :return:
    """
    for goods_id,texts in data.items():
        goods_id = str(goods_id)
        print(goods_id)
        Chroma.from_texts(
            texts=texts,
            embedding=embedding_model,
            collection_name=goods_id,
            client_settings=Settings(persist_directory=os.path.join(db_path, goods_id))
        )


if __name__ == '__main__':
    embedding_model_name = "./embed_model/shibing624-text2vec-base-chinese"
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
    data = corpus_preprocess("./test_data/shoes_data.tsv")
    data_save_to_chroma("./test_data",data,embeddings)

    # search
    deal_db_faq_qa = Chroma(
        collection_name = "145",
        embedding_function = embeddings,
        client_settings=Settings(
            persist_directory=os.path.join("./test_data", "145"),
        )
    )
    result = deal_db_faq_qa.similarity_search("价格")
    print(result)