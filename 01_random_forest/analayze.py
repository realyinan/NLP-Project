import pandas as pd
import numpy as np
from collections import Counter
import jieba


# 1. 读取数据
content = pd.read_csv(filepath_or_buffer="./data/train.txt", sep="\t", names=["sentence", "label"])
# print(content.head(10))

# 2.统计类别数量
counter = Counter(content["label"].values)
print(counter)
# print(len(counter))


# 数据样本分析
# 2.1 样本总量
total= 0
for key, value in counter.items():
    total += value
print(f"样本总量: {total}")

# 2.2 样本的类别比例
for key, value in counter.items():
    print(key, (value/total) *100 , "%")

# 2.3 文本长度分析
content["sentence_len"] = content["sentence"].apply(len)
content_mean = np.mean(content["sentence_len"])
content_std = np.std(content["sentence_len"])
print(content_mean)
print(content_std)


# 3. 分词处理
def cut_sentence(s):
    return jieba.lcut(s)


content["words"] = content["sentence"].apply(cut_sentence)
content["words"] = content["words"].apply(lambda s: " ".join(s))
print(content.head(10))

# 保存csv
content.to_csv("./data/train_new.csv")