import torch
from importlib import import_module
import numpy as np


CLS = "[CLS]"
id_to_name = {0: 'finance', 1: 'realty', 2: 'stocks', 3: 'education', 4: 'science',
              5: 'society', 6: 'politics', 7: 'sports', 8: 'game', 9: 'entertainment'}


def inference(model, config, input_text, pad_size=32):
    """
    :param model: 已加载的模型
    :param config: 模型配置信息
    :param input_text: 待分析的文本
    :param pad_size: 指定文本的填充长度
    """
    content = config.tokenizer.tokenize(input_text)
    content = [CLS] + content
    seq_len = len(content)
    token_ids = config.tokenizer.convert_tokens_to_ids(content)
    # 填充或截断文本至指定长度
    if seq_len < pad_size:
        mask = [1] * len(token_ids) + [0] * (pad_size - seq_len)
        token_ids += [0] * (pad_size - seq_len)
    else:
        mask = [1] * pad_size
        token_ids = token_ids[:pad_size]
        seq_len = pad_size
    # 将处理好的文本转化为张量
    token_ids = torch.LongTensor(token_ids).to(config.device)
    seq_len = torch.LongTensor(seq_len).to(config.device)
    mask = torch.LongTensor(mask).to(config.device)
    # 增加一维
    token_ids = token_ids.unsqueeze(0)
    seq_len = seq_len.unsqueeze(0)
    mask = mask.unsqueeze(0)
    data = (token_ids, seq_len, mask)
    # 模型推理
    output = model(data)
    # 获取模型预测结果
    predict_result = torch.max(output.data, dim=1)[1]
    return predict_result


if __name__ == "__main__":
    # 加载模型
    model_name = "bert"
    x = import_module("models." + model_name)
    config = x.Config()

    # 设置随机数种子, 保证实验的可重复性
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True

    # 创建并加载BERT模型
    model = torch.load(config.save_path2, map_location=config.device, weights_only=False)

    # 待分析的文本
    input_text = input("please input news title: ")

    # 进行模型推理
    res = inference(model, config, input_text)

    # 获取结果
    result = id_to_name[res.detach().item()]
    print(result)

