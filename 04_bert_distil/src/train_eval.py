import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from sklearn import metrics
import time
from utils import get_time_diff
from torch.optim import AdamW
from tqdm import tqdm


def fetch_teacher_outputs(teacher_model, train_iter):
    teacher_model.eval()
    # 存储教师模型的输出
    teacher_outputs = []
    # 禁用给梯度计算
    with torch.no_grad():
        for i, (data_batch, labels_batch) in enumerate(tqdm(train_iter)):
            # 获取教师模型的输出
            outputs = teacher_model(data_batch)
            # 将输出添加到列表中
            teacher_outputs.append(outputs.detach())
    # 返回教师模型对训练集的所有输出
    return teacher_outputs


def loss_fn(outputs, labels):
    """
    定义损失函数, 使用交叉熵损失函数
    """
    return nn.CrossEntropyLoss()(outputs, labels)


# KL散度损失
criterion = nn.KLDivLoss(reduction='batchmean')
# 定义蒸馏损失函数
def loss_fn_kd(outputs, labels, teacher_outputs):
    # 设置两个重要的超参数
    alpha = 0.8
    T = 2

    # 学生网络的带T参数的log_softmax输出分布
    output_student = F.log_softmax(outputs/T, dim=1)
    # 教师网络的带T参数的softmax输出分布
    output_teacher = F.softmax(teacher_outputs/T, dim=1)

    # 计算软目标损失, 第一个参数为student网络输出, 第二个参数为teacher网络输出
    soft_loss = criterion(output_student, output_teacher)
    # 硬目标损失, 学生网络的输出概率和真实标签之间的损失, 因为真实标签是one_hot编码, 因此直接使用交叉熵损失即可
    hard_loss = F.cross_entropy(outputs, labels)
    # 计算总损失
    # 原始论文中已经证明, 引入T会导致软目标产生的梯度和真实目标产生的梯度相比只有1/(T*T)
    # 因此计算完软目标的loss值后要乘以T^2.
    KD_loss = soft_loss * alpha * T * T + hard_loss * (1.0-alpha)
    return KD_loss


def train(config, model, train_iter, test_iter):
    """
    :param config: 配置信息对象
    :param model: 待训练的模型
    :param train_iter: 训练集的迭代器
    :param test_iter: 测试集的迭代器
    """
    # 记录开始训练的时间
    start_time = time.time()
    # 将模型设置为训练模式
    model.train()
    # 参数优化器设置
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    # 分组参数并设置优化的权重衰减
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": 0.01
         },
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0
         }]
    # 设置优化器
    optimizer = AdamW(params=optimizer_grouped_parameters, lr=config.learning_rate)

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
    # 在测试集上测试最终的模型
    test(config, model, test_iter)


def train_kd(cnn_config, bert_model, cnn_model, bert_train_iter, cnn_train_iter, cnn_test_iter):
    """
    :param cnn_config: 包含CNN模型超参数和设置的配置对象
    :param bert_model: BERT模型
    :param cnn_model: CNN模型
    :param bert_train_iter: 用于BERT模型训练的迭代器
    :param cnn_train_iter: 用于CNN模型训练的迭代器
    :param cnn_test_iter: 用于CNN模型测试的迭代器
    :return:
    """
    # 记录训练开始时间
    start_time = time.time()

    # 获取CNN模型参数
    param_optimizer = list(cnn_model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01
        },
        {
            "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0
        }]

    # 使用AdamW优化器，设置学习率
    optimizer = AdamW(optimizer_grouped_parameters, lr=cnn_config.learning_rate)

    # 将CNN模型设置为训练模式
    cnn_model.train()

    # 将BERT模型设置为评估模式
    bert_model.eval()

    # 获取BERT模型的输出作为教师模型的预测结果
    teacher_outputs = fetch_teacher_outputs(bert_model, bert_train_iter)

    # 遍历每个epoch
    for epoch in range(cnn_config.num_epoches):
        print("Epoch [{}/{}]".format(epoch + 1, cnn_config.num_epoches))
        # 遍历CNN模型训练数据集的每个batch
        for i, (trains, labels) in enumerate(tqdm(cnn_train_iter)):
            # 前向传播
            outputs = cnn_model(trains)
            # 梯度清零
            optimizer.zero_grad()
            # 计算蒸馏损失
            loss = loss_fn_kd(outputs, labels, teacher_outputs[i])
            # 反向传播和优化
            loss.backward()
            # 梯度更新
            optimizer.step()
            # 每400个batch打印一次训练信息
            if i % 10 == 0 and i != 0:
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predic)
                time_diff = get_time_diff(start_time)
                msg = "Iter: {0:>6}, Train Loss: {1:>5.2}, Train Acc: {2:>6.2%}, Time: {3}"
                print(msg.format(i, loss.detach().item(), train_acc, time_diff))
    torch.save(obj=cnn_model.state_dict(), f=cnn_config.save_path)
    # 在CNN测试集上测试最终的CNN模型
    test(cnn_config, cnn_model, cnn_test_iter)


def evaluate(config, model, data_iter):
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
    model.load_state_dict(torch.load(config.save_path, map_location=config.device))
    model.eval()
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



















