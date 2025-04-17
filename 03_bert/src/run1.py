import torch
import numpy as np
from train_eval import train, test
from utils import build_dataset, build_iterator
from importlib import import_module
import argparse


# 命令行参数解析
parser = argparse.ArgumentParser(description="Chinese Text Classification")
parser.add_argument("--model", type=str, default='bert', help="choose a model: bert")
args = parser.parse_args()

if __name__ == "__main__":
    if args.model == "bert":
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

        # 实例化并加载参数, 注意不要加载到GPU上, 只能在CPU上实现模型量化
        model = x.Model(config)
        model.load_state_dict(torch.load("./saved_dic/bert.pt", map_location="cpu"))

        # 量化模型
        quantized_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        print(quantized_model)

        # 测试量化后的模型在测试集上的表现
        test(config, quantized_model, test_iter)

        # 保存模型
        torch.save(quantized_model, f=config.save_path2)
