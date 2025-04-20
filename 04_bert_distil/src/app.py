import torch
from flask import Flask, request, render_template
from importlib import import_module
import numpy as np
import pickle


UNK, PAD, CLS = "[UNK]", "[PAD]", "[CLS]"
id_to_name = {0: 'finance', 1: 'realty', 2: 'stocks', 3: 'education', 4: 'science',
              5: 'society', 6: 'politics', 7: 'sports', 8: 'game', 9: 'entertainment'}

# 加载模型
model_name = "textCNN"
x = import_module("models." + model_name)
config = x.Config()
config.n_vocab = 4406

# 设置随机数种子, 保证实验的可重复性
np.random.seed(1)
torch.manual_seed(1)
torch.cuda.manual_seed_all(1)
torch.backends.cudnn.deterministic = True

# 创建并加载BERT模型
model = x.Model(config).to(config.device)
model.load_state_dict(torch.load(config.save_path, map_location=config.device))


def inference(model, config, input_text, pad_size=32):
    """
    :param model: 已加载的模型
    :param config: 模型配置信息
    :param input_text: 待分析的文本
    :param pad_size: 指定文本的填充长度
    """
    vocab = pickle.load(open("../data/data/vocab.pkl", "rb"))
    tokenizer = lambda x: [y for y in x]
    token = tokenizer(input_text)
    seq_len = len(token)

    if pad_size:
        if len(token) < pad_size:
            token.extend([PAD] * (pad_size - len(token)))
        else:
            token = token[:pad_size]
            seq_len = pad_size
    token_ids = [vocab.get(word, vocab.get(UNK)) for word in token]

    # 将处理好的文本转化为张量
    token_ids = torch.LongTensor(token_ids).to(config.device)
    seq_len = torch.LongTensor(seq_len).to(config.device)
    # 增加一维
    token_ids = token_ids.unsqueeze(0)
    seq_len = seq_len.unsqueeze(0)
    data = (token_ids, seq_len)
    # 模型推理
    output = model(data)
    # 获取模型预测结果
    predict_result = torch.max(output.data, dim=1)[1]
    predict_result = id_to_name[predict_result.detach().item()]
    return predict_result


# 创建flask应用
app = Flask(__name__)

# 定义路由
@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']
        sentiment = inference(model, config, text)
    return render_template('index.html', prediction=sentiment, text=text)

if __name__ == "__main__":
    app.run()

