import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from sklearn import metrics
import time
from utils import get_time_diff
from torch.optim import AdamW
from tqdm import tqdm


def loss_fn(outputs, labels):
    """
    定义损失函数, 使用交叉熵损失函数
    """
    return nn.CrossEntropyLoss()(outputs, labels)


def train(config, model, train_iter):
    """
    :param config: 配置信息对象
    :param model: 待训练的模型
    :param train_iter: 训练集的迭代器
    """
    # 记录开始训练的时间
    start_time = time.time()
    # 参数优化器设置
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": 0.01
         },
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0
         }]
    # 设置优化器
    optimizer = AdamW(params=optimizer_grouped_parameters, lr=config.learning_rate)
    # 将模型设置为训练状态
    model.train()
    for epoch in range(config.num_epoches):
        print("Epoch [{}/{}]".format(epoch + 1, config.num_epoches))
        for i, (trains, labels) in enumerate(tqdm(train_iter)):
            # 前向传播
            outputs = model(trains)
            # 梯度清零
            optimizer.zero_grad()
            # 计算损失
            loss = loss_fn(outputs, labels)
            # 反向传播
            loss.backward()
            # 参数更新
            optimizer.step()
            # 每100个batch输出在训练集和验证集上的效果
            if i % 100 == 0 and i != 0:
                real = labels.data.cpu()
                predict = torch.max(outputs.data, dim=1)[1].cpu()
                train_acc = metrics.accuracy_score(real, predict)
                # 计算时间差
                time_diff = get_time_diff(start_time)
                # 输出训练和验证集上的效果
                msg = "Iter: {0:>6}, Train Loss: {1:>5.2}, Train Acc: {2:>6.2%}, Time: {3}"
                print(msg.format(i, loss.detach().item(), train_acc, time_diff))
    torch.save(obj=model.state_dict(), f=config.save_path)


def evaluate(config, model, data_iter):
    # 采用量化模型进行推理时需要关闭
    # model.eval()
    loss_total = 0
    # 预测结果
    predict_all = np.array([], dtype=int)
    # 真实结果
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in tqdm(data_iter):
            # 将数据送入网络
            outputs = model(texts)
            # 损失函数
            loss = F.cross_entropy(outputs, labels)
            # 损失和
            loss_total += loss
            # 获取label信息
            labels = labels.data.cpu().numpy()
            # 获取预测结果
            predict = torch.max(outputs.data, dim=1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predict)
        # 计算准确率
        acc = metrics.accuracy_score(labels_all, predict_all)
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(data_iter), report, confusion



def test(config, model, test_iter):
    # 采用量化模型进行推理时需要关闭
    # model.eval()
    start_time = time.time()
    # 调用验证函数计算评估指标
    test_acc, test_loss, test_report, test_confusion = evaluate(config, model, test_iter)
    # 打印测试结果信息
    msg = "Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}"
    print(msg.format(test_loss, test_acc))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    time_dif = get_time_diff(start_time)
    print("Time usage:", time_dif)



















