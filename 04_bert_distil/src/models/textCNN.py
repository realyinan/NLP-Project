import torch
import torch.nn as nn
import torch.nn.functional as F
import os


class Config(object):
    def __init__(self):
        """
        配置类, 包含模型模型和训练的各种参数
        """
        self.model_name = "textCNN"  # 模型名称
        self.data_path = "../data/data/"  # 数据集的根路径
        self.train_path = self.data_path + "train.txt"
        self.test_path = self.data_path + "test.txt"
        self.class_list = [x.strip() for x in open(self.data_path + "class.txt").readlines()]  # 类别名单

        self.vocab_path = self.data_path + "vocab.pkl"
        self.save_path = "./saved_dic"  # 模型训练的保存路径
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        self.save_path += "/" + self.model_name + ".pt"  # 模型训练结果
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.dropout = 0.5  # 随机失活
        self.require_improvement = 1000 # 若超过1000batch效果还没有提升, 结束训练
        self.num_classes = len(self.class_list)  # 类别数
        self.n_vocab = 0  # 词表大小, 在运行时赋值
        self.num_epoches = 1
        self.batch_size = 128
        self.pad_size = 32
        self.learning_rate = 1e-3
        self.embed = 300  # 词向量维度
        self.filter_size = (2, 3, 4)
        self.num_filters = 64


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)  # padding_idx 的作用是在嵌入层中屏蔽填充符（如 <PAD>）的影响，避免模型学习无意义的内容
        self.convs = nn.ModuleList([nn.Conv2d(1, config.num_filters, (k, config.embed)) for k in config.filter_size])  # 卷积层列表, 包含不同卷积核大小的卷积层
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters * len(config.filter_size), config.num_classes)

    def conv_and_pool(self, x, conv):
        # 卷积和池化操作
        x = F.relu(conv(x).squeeze(3))
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        # 前向传播
        out = self.embedding(x[0])
        out = out.unsqueeze(1)
        # 对每个卷积层进行卷积和池化操作, 然后拼接在一起
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], dim=1)
        out = self.dropout(out)
        out = self.fc(out)
        return out

