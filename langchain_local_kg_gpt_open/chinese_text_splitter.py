from langchain.text_splitter import TextSplitter
import hanlp


class SentencesTextSplitter(TextSplitter):

    def __init__(self, max_chunk_size, **kwargs):
        """
        初始化文本分割器
        :param max_chunk_size:最大分割字符数量
        :param kwargs:
        """
        super().__init__(**kwargs)
        self.tokenizer = hanlp.utils.rules.split_sentence
        self.max_chunk_size = max_chunk_size

    def split_text(self, text):
        """
        当text字符数量超过max_chunk_size时，对text进行分割
        :param text:要分割的文本
        :return:分割之后的数组
        """
        if len(text) < self.max_chunk_size:
            return [text]
        splits_chunk = []
        texts = self.tokenizer(text)
        cur_text = []
        cur_len = 0
        for item in texts:
            item = str(item)
            if item.strip():
                if cur_len + len(item) >= self.max_chunk_size:
                    splits_chunk.append("".join(cur_text))
                    cur_text = [item]
                    cur_len = len(item)
                else:
                    cur_text.append(item)
                    cur_len += len(item)
        splits_chunk.append("".join(cur_text))
        return splits_chunk