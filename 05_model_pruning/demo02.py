import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 1: 图像的输入通道(1是黑白图像), 6: 输出通道, 3x3: 卷积核的尺寸
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5x5 是经历卷积操作后的图片尺寸
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, int(x.nelement() / x.shape[0]))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = LeNet().to(device)

# 实现模型的全局剪枝
print(model.state_dict().keys())
print("-"*150)

# 构建剪枝参数集合, 决定哪些层, 哪些参数会参与剪枝
parameters_to_prune = (
(model.conv1, "weight"),
(model.conv2, "weight"),
(model.fc1, "weight"),
(model.fc2, "weight"),
(model.fc3, "weight")
)

prune.global_unstructured(parameters=parameters_to_prune, pruning_method=prune.L1Unstructured, amount=0.2)

# 打印剪枝后的状态字典
print(model.state_dict().keys())

print(
    "Sparsity in conv1.weight: {:.2f}%".format(
        100. * float(torch.sum(model.conv1.weight == 0))
        / float(model.conv1.weight.nelement())
    ))

print(
    "Sparsity in conv2.weight: {:.2f}%".format(
        100. * float(torch.sum(model.conv2.weight == 0))
        / float(model.conv2.weight.nelement())
    ))

print(
    "Sparsity in fc1.weight: {:.2f}%".format(
        100. * float(torch.sum(model.fc1.weight == 0))
        / float(model.fc1.weight.nelement())
    ))

print(
    "Sparsity in fc2.weight: {:.2f}%".format(
        100. * float(torch.sum(model.fc2.weight == 0))
        / float(model.fc2.weight.nelement())
    ))

print(
    "Sparsity in fc3.weight: {:.2f}%".format(
        100. * float(torch.sum(model.fc3.weight == 0))
        / float(model.fc3.weight.nelement())
    ))


print(
    "Global sparsity: {:.2f}%".format(
        100. * float(torch.sum(model.conv1.weight == 0)
                     + torch.sum(model.conv2.weight == 0)
                     + torch.sum(model.fc1.weight == 0)
                     + torch.sum(model.fc2.weight == 0)
                     + torch.sum(model.fc3.weight == 0))
        / float(model.conv1.weight.nelement()
                + model.conv2.weight.nelement()
                + model.fc1.weight.nelement()
                + model.fc2.weight.nelement()
                + model.fc3.weight.nelement())
    ))
