from pathlib import Path
import requests
import pickle
import gzip
import torch
from matplotlib import pyplot
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch import optim

# ============
# 下载手写数据集
# ============
DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"

PATH.mkdir(parents=True, exist_ok=True)

URL = "http://deeplearning.net/data/mnist/"
FILENAME = "mnist.pkl.gz"

if not (PATH / FILENAME).exists():
    content = requests.get(URL + FILENAME).content
    (PATH / FILENAME).open("wb").write(content)

# ============
# 加载
# ============
with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")

# ============
# 显示数据集大小
# ============
# 50000张图片和784个特征(28×28)
# print(x_train.shape)
# print(y_train.shape)

# =============
# 数据转换tensor
# =============
x_train, y_train, x_valid, y_valid = map(
    torch.tensor, (x_train, y_train, x_valid, y_valid)
)

n, c = x_train.shape

# print(x_train.shape)
# print(y_train.min(), y_train.max())

# ==============
# nn.functional
# ==============
loss_func = F.cross_entropy


# def model(xb):
#     return xb.mm(weights) + bias
#
bs = 64
# xb = x_train[0:bs]  # a mini-batch from x
# yb = y_train[0:bs]
# weights = torch.randn([784, 10], dtype = torch.float,  requires_grad = True)
# bs = 64
# bias = torch.zeros(10, requires_grad=True)
#
# print(loss_func(model(xb), yb))

# ==============
# nn.Module
# ==============
class Mnist_NN(nn.Module):
    def __init__(self):
        super().__init__()
        # 隐藏层1
        self.hidden1 = nn.Linear(784, 128)
        # 隐藏层2
        self.hidden2 = nn.Linear(128, 256)
        # 输出层(全连接层)
        self.out = nn.Linear(256, 10)
        # 丢弃
        # self.dropout = nn.Dropout(0.5)

    # 前向传播（自动反向传播）
    def forward(self, x):
        x = F.relu(self.hidden1(x))
        # x = x.self.dropout()
        x = F.relu(self.hidden2(x))
        # x = x.self.dropout()
        x = self.out(x)
        return x


# 神经网络结构
net = Mnist_NN()
print("神经网络结构:")
print(net)

# ==========================
# TensorDataset和DataLoader
# ==========================
# 打包训练集和测试集
# 训练集
train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)

# 测试集
valid_ds = TensorDataset(x_valid, y_valid)
valid_dl = DataLoader(valid_ds, batch_size=bs * 2)


# 获取数据
def get_data(train_ds, valid_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs),
    )


# 训练函数
def fit(steps, model, loss_func, opt, train_dl, valid_dl):
    for step in range(steps):
        model.train()
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)

        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl]
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        print('当前step:' + str(step), '验证集损失：' + str(val_loss))


# 获取模型
def get_model():
    model = Mnist_NN()
    return model, optim.Adam(model.parameters(), lr=0.001)


# ===================
# 计算损失并更新权重参数
# ===================
def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)


# 开始训练
train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
model, opt = get_model()
fit(6, model, loss_func, opt, train_dl, valid_dl)


# 验证结果
correct = 0
total = 0
for xb, yb in valid_dl:
    outputs = model(xb)
    _, predicted = torch.max(outputs.data, 1)
    total += yb.size(0)
    correct += (predicted == yb).sum().item()
print('准确率：' + str(correct / total))