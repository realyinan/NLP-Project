from flask import Flask
from flask import request
import fasttext
import jieba
import requests


# 实例化flask对象
app = Flask(__name__)

@app.route("/v1/fasttest", methods=["GET"])
def predict():
    text = "全球气候变化加剧，联合国呼吁各国加快减排步伐"
    # 加载自定义的停用词表
    jieba.load_userdict("./data/stopwords.txt")
    # 提供已经训练好的模型路径
    model_save_path= "./data/fasttest_model.bin"
    # 实例化fasttest对象
    model = fasttext.load_model(model_save_path)
    print("模型实例化完毕")

    # 1. 接收输入
    # text = request.form["text"]
    text_new = " ".join(list(text))

    # 2. 模型预测
    pred = model.predict(text_new)
    result = pred[0][0]

    return f"预测结果为: {result}"



if __name__ == "__main__":
    app.run()

