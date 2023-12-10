import pandas as pd
from chinese_text_splitter import SentencesTextSplitter

data = pd.read_csv("../test_data/shoes_data.tsv",sep="\t")

splitter = SentencesTextSplitter(max_chunk_size=10)
for id,row in data.iterrows():
    result = splitter.split_text(row["句子名"])
    print(result)