from sentence_transformers import SentenceTransformer


class Embedding():
    def __init__(self):
        self.model = SentenceTransformer("shibing624/text2vec-base-chinese")

    def encoder(self, sentence):
        return self.model.encode(sentence)

    def get_sentence_embedding(self, sentence):
        """
        对sentence进行向量化
        :param sentence: 要进行向量化的文本
        :return: 向量结果
        """
        embeddings = self.encoder(sentence)
        return embeddings

if __name__ == '__main__':
    embed = Embedding()
    result= embed.get_sentence_embedding("中国的首都是北京安天门")
    print(result)
