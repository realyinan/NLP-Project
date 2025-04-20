from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import f1_score


content = pd.read_csv("./data/train_new.csv")

# 1. 构建语料库
corpus = content["words"].values
# print(corpus)

# 2. 获取停用词
stopwords = open("./data/stopwords.txt", encoding="utf-8").read().split()
# print(stopwords)

# 3. 计算tfidf特征
vector = TfidfVectorizer(stop_words=stopwords)  # TfidfVectorizer 是 Scikit-learn 中的一个工具，用于将 文本数据 转换为 TF-IDF 特征向量，从而使得文本可以输入到机器学习模型中进行训练。
text_vector = vector.fit_transform(corpus)
print(text_vector)

# 4. 模型的训练和预测
label = content["label"]
x_train, x_test, y_train, y_test = train_test_split(text_vector, label, test_size=0.2, random_state=1)
model = RandomForestClassifier(n_estimators=10)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
acc = model.score(x_test, y_test)
F1 = f1_score(y_pred, y_test)
print("acc: ", acc)
print(F1)
