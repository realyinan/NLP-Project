import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F


device = "cuda" if torch.cuda.is_available() else "cpu"

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16*6*6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, int(x.nelement()/ x.shape[0]))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def demo01():
    # 执行剪枝操作
    # 第一个参数: module, 代表要进行剪枝的特定模块, 之前我们已经指定了module=model.conv1, 说明要对第一个卷积层执行剪枝操作
    # 第二个参数: name, 指定要对选中的模型快中的哪些参数执行剪枝, 如果指定了name="weight", 意味着对连接网络中的weight剪枝, 不对bias进行剪枝
    # 第三个参数: amount, 指定要对模型中多大比例的参数执行剪枝, 介于0.0-1.0之间的float数值, 代表百分比
    prune.random_unstructured(module, name="weight", amount=0.3)
    print(list(module.named_parameters()))
    print("*"*80)
    print(list(module.named_buffers()))
    print("*"*80)
    print(module.weight)
    print('_'*100)


def demo02():
    # 注意: 第三个参数amount, 当为一个float值代表的是剪枝的百分比, 如果为整数值, 代表剪裁掉的绝对数量
    prune.l1_unstructured(module, name="bias", amount=2)  # 计算所有权重的 绝对值（L1范数）剪掉最小的那一部分
    print(list(module.named_parameters()))
    print("*"*80)
    print(list(module.named_buffers()))
    print("*"*80)
    print(module.bias)
    print("_"*100)


def demo03():
    # 首先将原始模型的状态字典打印出来
    print(module.state_dict().keys())
    print("*"*80)

    # 直接执行剪枝操作
    prune.random_unstructured(module, name="weight", amount=0.3)
    prune.l1_unstructured(module, name="bias", amount=3)

    # 然后将剪枝后的模型的状态字典打印出来
    print(module.state_dict().keys())

    # 打印剪枝后的模型参数
    print(list(module.named_parameters()))
    print("*"*80)

    # 打印剪枝后的模型mask buffers参数
    print(list(module.named_buffers()))
    print("*"*80)

    # 剪枝后的模型weight属性值
    print(module.weight)

    # 执行模型剪枝的永久化操作
    prune.remove(module, name="weight")

    # 打印执行remove之后的模型参数
    print(list(module.named_parameters()))

    # 打印执行remove之后的mask buffers参数
    print(list(module.named_buffers()))


def demo04():
    # 打印初始模型的mask buffers张量字典名称
    print(dict(model.named_buffers()).keys())
    print('------------------------------------------------------------------------')

    # 打印初始模型的所有状态字典
    print(model.state_dict().keys())
    print('------------------------------------------------------------------------')

    for name, module in model.named_modules():
        # 对模型中所有的卷积层执行l1_unstructured剪枝操作, 选取20%的参数进行剪枝
        if isinstance(module, torch.nn.Conv2d):
            prune.l1_unstructured(module, name="weight", amount=0.2)
        if isinstance(module, torch.nn.Linear):
            prune.ln_structured(module, name="weight", amount=0.4, n=2, dim=0)

    # 打印多参数模块剪枝后的mask buffers张量字典名称
    print(dict(model.named_buffers()).keys())
    print('------------------------------------------------------------------------')

    # 打印多参数模块剪枝后模型的所有状态字典名称
    print(model.state_dict().keys())


def demo05():
    prune.ln_structured(module, name="weight", n=2, amount=3, dim=0)
    print(module.weight)

if __name__ == "__main__":
    model = LeNet().to(device=device)
    module = model.conv1
    print(list(module.named_parameters()))
    print("*"*80)
    print(list(module.named_buffers()))  # 缓冲区张量
    print('_'*100)

    # demo01()
    # demo02()
    # demo03()
    # demo04()
    demo05()