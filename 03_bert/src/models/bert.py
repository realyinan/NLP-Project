import torch
import torch.nn as nn
import os
from transformers import BertModel, BertConfig, BertTokenizer


class Config(object):
    def __init__(self):
        """
        配置类, 包含模型模型和训练的各种参数
        """
        self.model_name = "bert"  # 模型名称
        self.data_path = "../data/data/"  # 数据集的根路径
        self.train_path = self.data_path + "train.txt"
        self.test_path = self.data_path + "test.txt"
        self.class_list = [x.strip() for x in open(self.data_path + "class.txt").readlines()]  # 类别名单

        self.save_path = "./saved_dic"  # 模型训练的保存路径
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.save_path += "/" + self.model_name + ".pt"  # 模型训练结果

        self.save_path2 = "./saved_dic1"  # 量化模型存储路径
        if not os.path.exists(self.save_path2):
            os.makedirs(self.save_path2)
        self.save_path2 += "/" + self.model_name + "_quantized.pt"  # 模型训练结果

        # self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = "cpu"

        self.num_classes = len(self.class_list)  # 类别数
        self.num_epoches = 1
        self.batch_size = 128
        self.pad_size = 32
        self.learning_rate = 5e-5

        self.bert_path = "../data/bert_pretrain"  # 预训练模型的路径
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.bert_config = BertConfig.from_pretrained(self.bert_path + "/bert_config.json")
        self.hidden_size = 768


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert = BertModel.from_pretrained(config.bert_path, config=config.bert_config)
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, x):
        # x: 模型输入, 包含句子, 句子长度和填充掩码
        context = x[0]
        mask = x[2]

        _, pooled = self.bert(context, attention_mask=mask, return_dict=False)  # `return_dict` 参数决定了模型输出是以元组形式还是字典形式返回。
        out = self.fc(pooled)
        return out