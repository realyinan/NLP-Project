import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import time

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


model = LeNet().to(device=device)


# 自定义剪枝方法的类, 一定要继承prune.BasePruningMethod
class myself_pruning_method(prune.BasePruningMethod):
    PRUNING_TYPE = "unstructured"

    # 内部实现compute_mask函数, 完成程序员自己定义的剪枝规则, 本质上就是如何去mask掉权重参数
    def compute_mask(self, t, default_mask):
        mask = default_mask.clone()
        # 定义的规则是每隔一个参数就遮掩掉一个, 最终所有参与剪枝的参数量中的50%被mask掉
        mask.view(-1)[::3] = 0
        return mask


# 自定义剪枝方法的函数, 内部直接调用剪枝类的方法apply
def myself_unstructured_pruning(module, name):
    myself_pruning_method.apply(module, name)
    return module


start = time.time()

# 调用自定义的剪枝方法的函数, 对model中的第三个全连接层fc3中的偏置bias执行自定义剪枝
myself_unstructured_pruning(model.fc3, name='bias')

# 剪枝成功的最大标志, 就是拥有了bias_mask参数
print(model.fc3.bias_mask)

# 打印一下自定义剪枝的耗时
duration = time.time() - start
print(duration * 1000, 'ms')
