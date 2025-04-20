import torch
import numpy as np
from train_eval import train_kd, train
from utils import build_dataset, build_iterator, build_dataset_cnn
from importlib import import_module
import argparse


# 命令行参数解析
parser = argparse.ArgumentParser(description="Chinese Text Classification")
parser.add_argument("--task", type=str, default="train_kd", help="choose a task: trainbert, or train_kd")
args = parser.parse_args()

if __name__ == "__main__":
    if args.task == "train_bert":
        # 导入的对应的模型配置和模型定义
        model_name = "bert"
        x = import_module("models." + model_name)
        config = x.Config()

        # 设置随机数种子, 保证实验的可重复性
        np.random.seed(1)
        torch.manual_seed(1)
        torch.cuda.manual_seed_all(1)
        torch.backends.cudnn.deterministic = True  # 保证每次结果一样
        print("Loading data for Bert Model...")

        # 构建训练, 验证, 测试数据集和数据迭代器
        train_data, test_data = build_dataset(config)
        train_iter = build_iterator(train_data, config)
        test_iter = build_iterator(test_data, config)

        # 训练模型
        model = x.Model(config).to(config.device)
        train(config, model, train_iter, test_iter)

    if args.task == "train_kd":
        # 加载bert模型
        model_name = "bert"
        bert_module = import_module("models." + model_name)
        bert_config = bert_module.Config()

        # 加载cnn模型
        model_name = "textCNN"
        cnn_module = import_module("models." + model_name)
        cnn_config = cnn_module.Config()

        # 初始化
        np.random.seed(1)
        torch.manual_seed(1)
        torch.cuda.manual_seed_all(1)
        torch.backends.cudnn.deterministic = True  # 保证每次结果一样
        print("Loading data for Bert Model...")

        # 构建bert数据集
        bert_train_data, _ = build_dataset(bert_config)
        bert_train_iter = build_iterator(bert_train_data, bert_config)

        # 构建cnn数据集
        vocab, cnn_train_data, cnn_test_data = build_dataset_cnn(cnn_config)
        cnn_train_iter = build_iterator(cnn_train_data, cnn_config)
        cnn_test_iter = build_iterator(cnn_test_data, cnn_config)
        cnn_config.n_vocab = len(vocab)

        # 加载训练好的teacher模型
        bert_model = bert_module.Model(bert_config).to(bert_config.device)
        bert_model.load_state_dict(torch.load(bert_config.save_path, map_location=bert_config.device))

        # 加载学生模型
        cnn_model = cnn_module.Model(cnn_config).to(cnn_config.device)
        print("Teacher and student models loaded, start training")
        train_kd(cnn_config, bert_model, cnn_model, bert_train_iter, cnn_train_iter, cnn_test_iter)






