import pandas as pd
import numpy as np
import torch as t
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.utils.data.dataset import Dataset
import tensorboard
from torch.utils.tensorboard import SummaryWriter


# 构建网络模型
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.model_c = nn.Sequential(
            nn.Linear(in_features=2, out_features=50, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=50, out_features=200, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=200, out_features=400, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=400, out_features=600, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=600, out_features=1000, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=1000, out_features=1000, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=1000, out_features=2000, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=2000, out_features=2000, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=2000, out_features=2000, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=2000, out_features=1500, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=1500, out_features=1500, bias=True),
            nn.ReLU(),
            nn.Linear(in_features=1500, out_features=4004, bias=True)
        )

    def forward(self, x):
        x = self.model_c(x)
        return x

    def forward(self, x):
        x = self.model_c(x)
        return x


if __name__ == '__main__':
    network = Network()
    device = t.device("cuda:1" if t.cuda.is_available() else "cpu")
    network.to(device)

    # 1.准备数据集
    train_data = pd.read_csv('train_correct_10000_10_0.8.csv', skiprows=0, header=None).values
    train_data = np.array(train_data, dtype=np.complex64)
    Dx = train_data[:, 0:1].real
    Dy = train_data[:, 1:2].real
    # rotation_angle = np.zeros((7401, 1))
    para = np.hstack((Dx, Dy))
    real_x = (train_data[:, 2:1003]).real
    imag_x = (train_data[:, 2:1003]).imag
    real_y = (train_data[:, 1003:2004]).real
    imag_y = (train_data[:, 1003:2004]).imag

    train_set = np.hstack((para, real_x, imag_x, real_y, imag_y))
    train_set = t.from_numpy(train_set).to(t.float32)
    train_Loader = t.utils.data.DataLoader(train_set, batch_size=64, shuffle=True, num_workers=2)
    # 测试集
    test_data = pd.read_csv('test_correct_10000_10.csv', skiprows=0).values
    test_data = np.array(test_data, dtype=np.complex64)
    Dx_test = test_data[:, 0:1].real
    Dy_test = test_data[:, 1:2].real
    # rotation_angle_test = np.zeros((410, 1))
    para_test = np.hstack((Dx_test, Dy_test))
    real_x_test = (test_data[:, 2:1003]).real
    imag_x_test = (test_data[:, 2:1003]).imag
    real_y_test = (test_data[:, 1003:2004]).real
    imag_y_test = (test_data[:, 1003:2004]).imag

    test_set = np.hstack((para_test, real_x_test, imag_x_test, real_y_test, imag_y_test))
    test_set = t.from_numpy(test_set).to(t.float32)
    test_Loader = t.utils.data.DataLoader(test_set, batch_size=64, shuffle=True, num_workers=2)
    # 4.损失函数
    loss_fn = nn.MSELoss().to(device)
    # 5.优化器
    optimizer = optim.Adam(network.parameters(), lr=1e-4)
    scheduler = t.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)
    # 6.训练网络
    total_train_step = 0
    total_test_step = 0

    iterations = 80000
    writer = SummaryWriter('./correct_10000_dataset10')

    for epoch in range(iterations):
        print("------第{}轮训练开始------".format(epoch + 1))
        traindata = next(iter(train_Loader))
        inputs, points = traindata[:, 0:2].to(device), traindata[:, 2:4006].to(device)
        outputs = network(inputs)
        train_loss = loss_fn(outputs, points)
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        writer.add_scalar('train_loss', train_loss, epoch)
        print("训练次数{}，loss：{}".format(epoch, train_loss.item()))
        if epoch % 100 == 0:
            # writer.add_text('configure', 'visible_forward.Learning rate:1e-3, step1000, gamma0.9, 8550组数据', epoch)
            t.save(network, "correct_10000_dataset10.pth")

        # 测试网络
        network.eval()
        testdata = next(iter(test_Loader))
        test_inputs, test_points = testdata[:, 0:2].to(device), testdata[:, 2:4006].to(device)
        test_outputs = network(test_inputs)
        test_loss = loss_fn(test_outputs, test_points)
        writer.add_scalars('MSE_LOSS', {'test_loss': test_loss, 'train_loss': train_loss}, epoch)
        scheduler.step()
    print('----------finished Training----------')


