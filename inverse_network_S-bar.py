import cmath
import time
import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy import *
from torch import nn, optim
import torch as t
from torch.utils.tensorboard import SummaryWriter
import os


class Network3d(nn.Module):
    def __init__(self):
        super(Network3d, self).__init__()
        self.model_3D = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=(2, 2, 3), stride=(2, 2, 1), padding=(0, 0, 1)),
            nn.BatchNorm3d(32),
            nn.ReLU(),  # 长宽厚50*50*5

            nn.Conv3d(32, 64, kernel_size=(2, 2, 2), stride=(2, 2, 1), padding=0),
            nn.BatchNorm3d(64),
            nn.ReLU(),  # 长宽厚25*25*4

            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(2, 2, 1), padding=(0, 0, 1)),  # 长宽厚12*12*4
            nn.Conv3d(128, 128, kernel_size=(3, 3, 2), stride=1, padding=(0, 0, 0)),  # 长宽厚10*10*3
            nn.Conv3d(128, 128, kernel_size=(3, 3, 3), stride=1, padding=(0, 0, 1)),  # 长宽厚8*8*3
            nn.BatchNorm3d(128),
            nn.ReLU(),

            nn.Conv3d(128, 256, kernel_size=(2, 2, 2), stride=(2, 2, 1), padding=0),  # 长宽厚4*4*2
            nn.BatchNorm3d(256),
            nn.ReLU(),  # 长宽厚4*4*2

            nn.Conv3d(256, 512, kernel_size=(2, 2, 1), stride=(2, 2, 1), padding=0),  # 长宽厚2*2*2
            nn.Conv3d(512, 512, kernel_size=(2, 2, 2), stride=1, padding=0),  # 长宽厚1*1*1
            nn.ReLU(),
        )

        self.model_Dx = nn.Sequential(
            nn.ConvTranspose3d(512, 512, kernel_size=(2, 2, 1), stride=(1, 1, 1), padding=(0, 0, 0)),  # 长宽厚2*2*1
            nn.ConvTranspose3d(512, 256, kernel_size=(2, 2, 1), stride=(2, 2, 1), padding=(0, 0, 0)),  # 长宽厚4*4*1
            nn.BatchNorm3d(256),
            nn.ReLU(),

            nn.ConvTranspose3d(256, 128, kernel_size=(3, 3, 1), stride=(1, 1, 1), padding=(1, 1, 0)),  # 长宽厚4*4*1
            nn.BatchNorm3d(128),
            nn.ReLU(),

            nn.ConvTranspose3d(128, 128, kernel_size=(2, 2, 1), stride=(2, 2, 1), padding=(0, 0, 0)),  # 长宽厚8*8*1
            nn.ConvTranspose3d(128, 128, kernel_size=(3, 3, 1), stride=(1, 1, 1), padding=(1, 1, 0)),  # 长宽厚8*8*1
            nn.ConvTranspose3d(128, 64, kernel_size=(2, 2, 1), stride=(2, 2, 1), padding=(0, 0, 0)),  # 长宽厚16*16*1
            nn.BatchNorm3d(64),
            nn.ReLU(),

            nn.ConvTranspose3d(64, 32, kernel_size=(2, 2, 1), stride=(2, 2, 1), padding=(0, 0, 0)),
            nn.BatchNorm3d(32),
            nn.ReLU(),

            nn.ConvTranspose3d(32, 1, kernel_size=(2, 2, 1), stride=(2, 2, 1), padding=(0, 0, 0)),  # 长宽厚64*64*1
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        self.model_Dy = nn.Sequential(
            nn.ConvTranspose3d(512, 512, kernel_size=(2, 2, 1), stride=(1, 1, 1), padding=(0, 0, 0)),  # 长宽厚2*2*1
            nn.ConvTranspose3d(512, 256, kernel_size=(2, 2, 1), stride=(2, 2, 1), padding=(0, 0, 0)),  # 长宽厚4*4*1
            nn.BatchNorm3d(256),
            nn.ReLU(),

            nn.ConvTranspose3d(256, 128, kernel_size=(3, 3, 1), stride=(1, 1, 1), padding=(1, 1, 0)),  # 长宽厚4*4*1
            nn.BatchNorm3d(128),
            nn.ReLU(),

            nn.ConvTranspose3d(128, 128, kernel_size=(2, 2, 1), stride=(2, 2, 1), padding=(0, 0, 0)),  # 长宽厚8*8*1
            nn.ConvTranspose3d(128, 128, kernel_size=(3, 3, 1), stride=(1, 1, 1), padding=(1, 1, 0)),  # 长宽厚8*8*1
            nn.ConvTranspose3d(128, 64, kernel_size=(2, 2, 1), stride=(2, 2, 1), padding=(0, 0, 0)),  # 长宽厚16*16*1
            nn.BatchNorm3d(64),
            nn.ReLU(),

            nn.ConvTranspose3d(64, 32, kernel_size=(2, 2, 1), stride=(2, 2, 1), padding=(0, 0, 0)),
            nn.BatchNorm3d(32),
            nn.ReLU(),

            nn.ConvTranspose3d(32, 1, kernel_size=(2, 2, 1), stride=(2, 2, 1), padding=(0, 0, 0)),  # 长宽厚64*64*1
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )
        self.model_theta = nn.Sequential(
            nn.ConvTranspose3d(512, 512, kernel_size=(2, 2, 1), stride=(1, 1, 1), padding=(0, 0, 0)),  # 长宽厚2*2*1
            nn.ConvTranspose3d(512, 256, kernel_size=(2, 2, 1), stride=(2, 2, 1), padding=(0, 0, 0)),  # 长宽厚4*4*1
            nn.BatchNorm3d(256),
            nn.ReLU(),

            nn.ConvTranspose3d(256, 128, kernel_size=(3, 3, 1), stride=(1, 1, 1), padding=(1, 1, 0)),  # 长宽厚4*4*1
            nn.BatchNorm3d(128),
            nn.ReLU(),

            nn.ConvTranspose3d(128, 128, kernel_size=(2, 2, 1), stride=(2, 2, 1), padding=(0, 0, 0)),  # 长宽厚8*8*1
            nn.ConvTranspose3d(128, 128, kernel_size=(3, 3, 1), stride=(1, 1, 1), padding=(1, 1, 0)),  # 长宽厚8*8*1
            nn.ConvTranspose3d(128, 64, kernel_size=(2, 2, 1), stride=(2, 2, 1), padding=(0, 0, 0)),  # 长宽厚16*16*1
            nn.BatchNorm3d(64),
            nn.ReLU(),

            nn.ConvTranspose3d(64, 32, kernel_size=(2, 2, 1), stride=(2, 2, 1), padding=(0, 0, 0)),
            nn.BatchNorm3d(32),
            nn.ReLU(),

            nn.ConvTranspose3d(32, 1, kernel_size=(2, 2, 1), stride=(2, 2, 1), padding=(0, 0, 0)),  # 长宽厚64*64*1
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )

    def forward(self, pattern_3d):
        intermedia = self.model_3D(pattern_3d)
        dx = self.model_Dx(intermedia)
        dy = self.model_Dy(intermedia)
        theta = self.model_theta(intermedia)
        return dx, dy, theta


# 已经训练好的网络结构
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.model_c = nn.Sequential(
            nn.Linear(2, 50, bias=True),
            nn.ReLU(),
            nn.Linear(50, 200, bias=True),
            nn.ReLU(),
            nn.Linear(200, 400, bias=True),
            nn.ReLU(),
            nn.Linear(400, 600, bias=True),
            nn.ReLU(),
            nn.Linear(600, 1000, bias=True),
            nn.ReLU(),
            nn.Linear(1000, 1000, bias=True),
            nn.ReLU(),
            nn.Linear(1000, 2000, bias=True),
            nn.ReLU(),
            nn.Linear(2000, 2000, bias=True),
            nn.ReLU(),
            nn.Linear(2000, 2000, bias=True),
            nn.ReLU(),
            nn.Linear(2000, 1500, bias=True),
            nn.ReLU(),
            nn.Linear(1500, 1500, bias=True),
            nn.ReLU(),
            nn.Linear(1500, 4004, bias=True)
        )

    def forward(self, x):
        x = self.model_c(x)
        return x


def get_dataset(origin_image):
    # 获取图片的高度和宽度
    h, w = origin_image.shape[:2]
    # 计算旋转中心点，这里设为图片中心
    center = (w / 2, h / 2)
    # 计算旋转矩阵
    angle_random = int(t.randint(180, (1,)))
    # angle_random = 0
    scale = 1  # 缩放比
    M1 = cv2.getRotationMatrix2D(center, angle_random, scale)
    M2 = cv2.getRotationMatrix2D(center, angle_random + 15, scale)
    M3 = cv2.getRotationMatrix2D(center, angle_random + 30, scale)
    M4 = cv2.getRotationMatrix2D(center, angle_random + 45, scale)
    M5 = cv2.getRotationMatrix2D(center, angle_random + 60, scale)
    # 平移矩阵M：[[1,0,x],[0,1,y]]
    M_tans1 = np.float32([[1, 0, 0], [0, 1, 0]])
    M_tans2 = np.float32([[1, 0, 0], [0, 1, 0]])
    # 应用旋转矩阵到图片上，得到输出图片
    rot_img1 = t.tensor(cv2.warpAffine(origin_image, M1, (w, h)))
    rot_img1[rot_img1 > 0] = 1
    rot_img2 = t.tensor(cv2.warpAffine(cv2.warpAffine(origin_image, M2, (w, h)), M_tans1, (w, h)))
    rot_img2[rot_img2 > 0] = 1
    rot_img3 = t.tensor(cv2.warpAffine(cv2.warpAffine(origin_image, M3, (w, h)), M_tans2, (w, h)))
    rot_img3[rot_img3 > 0] = 1
    rot_img4 = t.tensor(cv2.warpAffine(cv2.warpAffine(origin_image, M4, (w, h)), M_tans1, (w, h)))
    rot_img4[rot_img4 > 0] = 1
    rot_img5 = t.tensor(cv2.warpAffine(origin_image, M5, (w, h)))
    rot_img5[rot_img5 > 0] = 1
    rot_img = t.stack((rot_img1, rot_img2, rot_img3, rot_img4, rot_img5), dim=2)
    rot_image = t.reshape(rot_img, (-1, 1, 100, 100, 5))
    return rot_image


def compute_rotate(ex_x, ex_y, ey_x, ey_y, theta):
    e_xx = t.cos(theta) ** 2 * ex_x - t.cos(theta) * t.sin(theta) * ex_y - t.cos(theta) * t.sin(theta) * ey_x + t.sin(
        theta) ** 2 * ey_y
    e_xy = t.cos(theta) * t.sin(theta) * ex_x + t.cos(theta) ** 2 * ex_y - t.sin(theta) ** 2 * ey_x - t.cos(
        theta) * t.sin(theta) * ey_y
    e_yx = t.cos(theta) * t.sin(theta) * ex_x - t.sin(theta) ** 2 * ex_y + t.cos(theta) ** 2 * ey_x - t.cos(
        theta) * t.sin(theta) * ey_y
    e_yy = t.sin(theta) ** 2 * ex_x + t.cos(theta) * t.sin(theta) * ex_y + t.cos(theta) * t.sin(theta) * ey_x + t.cos(
        theta) ** 2 * ey_y
    return e_xx, e_xy, e_yx, e_yy


def spectrum_normalize(spectrum):
    amp_normalized_0 = t.abs(spectrum[:, :, 0]) / t.max(t.abs(spectrum[:, :, 0]))
    amp_normalized_1 = t.abs(spectrum[:, :, 1]) / t.max(t.abs(spectrum[:, :, 1]))
    amp_normalized_2 = t.abs(spectrum[:, :, 2]) / t.max(t.abs(spectrum[:, :, 2]))
    amp_normalized_3 = t.abs(spectrum[:, :, 3]) / t.max(t.abs(spectrum[:, :, 3]))
    amp_normalized_4 = t.abs(spectrum[:, :, 4]) / t.max(t.abs(spectrum[:, :, 4]))
    amp_normalized = t.stack((amp_normalized_0, amp_normalized_1, amp_normalized_2, amp_normalized_3, amp_normalized_4), dim=2)
    return amp_normalized


def metasurface_to_image_rayleigh_sommerfeld(freq, amp_metasurface, pha_metasurface, device):
    """
    :param freq: tensor  ->  超表面的频率 width x height
    :param amp_metasurface: tensor  ->  超表面的振幅分布 width x height
    :param pha_metasurface: tensor  ->  超表面的相位分布 width x height  >>> 必须是弧度制 <<<
    :param device: str  ->  所使用的硬件
    """
    c = 3e8
    # 计算角频所对应的波长 λ = c/f
    wavelength = c / (freq * 1e12)
    # 计算角频率所对应的波矢 k = 2π / λ
    k = 2 * t.pi / wavelength
    # 像的坐标网格,网格周期1e-4米，100*100个网格，像面10mm *10mm
    x = (np.arange(64) - 64 / 2 + 0.5) * 2e-4
    y = (np.arange(64) - 64 / 2 + 0.5) * 2e-4
    mesh_x, mesh_y = np.meshgrid(x, y)
    image_x = t.tensor(mesh_x, device=t.device(device))
    image_y = t.tensor(mesh_y, device=t.device(device))
    # 超表面的坐标网格,网格周期2e-4米，64*64个网格，超表面12.8mm *12.8mm
    mx = (np.arange(64) - 64 / 2 + 0.5) * 2e-4
    my = (np.arange(64) - 64 / 2 + 0.5) * 2e-4
    mesh_mx, mesh_my = np.meshgrid(mx, my)
    metasurface_x = t.tensor(mesh_mx, device=t.device(device))
    metasurface_y = t.tensor(mesh_my, device=t.device(device))
    # # 入射光振幅添加高斯
    # beam_waist_radius = 5e-3

    # r_gauss = t.sqrt(t.multiply(metasurface_x, metasurface_x) + t.multiply(metasurface_y, metasurface_y))
    # amp_metasurface_gauss = amp_metasurface * t.exp(- (r_gauss ** 2) / (beam_waist_radius ** 2))
    # 计算超表面的电磁场到像的 瑞利-索末菲 衍射场分布
    image_field = t.zeros((64, 64, 5), dtype=t.complex64)  # 像面上也是100*100个点
    for i in range(64):
        for j in range(64):
            for distance in range(0, 5):
                # 像上的第 i, j 像素点到超表面上每一个像素点的距离
                r_ij = t.sqrt((metasurface_x - image_x[i, j]) ** 2 + (metasurface_y - image_y[i, j]) ** 2
                              + (distance * 1e-3 + 3e-3) ** 2)
                # 计算离散场的 瑞利-索末菲 衍射积分 U(r) = Ad/(iλ)∬(exp(-ikr)/(r^2)dxdy
                image_field[i, j, distance] = t.sum(
                    amp_metasurface * t.exp(1j * pha_metasurface) * (distance * 1e-3 + 3e-3) * 4e-8 / (
                            1j * wavelength)
                    * t.exp(- 1j * k * r_ij) / (r_ij * r_ij))
    return image_field


def angular_spectrum_method(freq, amp_metasurface, pha_metasurface, device):
    c = 3e8
    # 计算角频所对应的波长 λ = c/f
    wavelength = c / (freq * 1e12)
    # 计算角频率所对应的波矢 k = 2π / λ
    k = 2 * t.pi / wavelength
    dx = 200 * 1e-6  # 单元结构周期
    n_meta = 64  # 超表面的单元结构个数
    comp_amp = amp_metasurface * t.exp(1j * pha_metasurface)
    # obtain the output field using Angular spectrum
    fx = t.arange(-1 / (2 * dx), 1 / (2 * dx), 1 / (dx * n_meta))  # freq coords
    FX, FY = t.meshgrid(fx, fx)
    aa = list(map(cmath.sqrt, ((1 - (wavelength * FX) ** 2 - (wavelength * FY) ** 2).flatten())))
    bb = t.reshape(t.tensor(aa, device=device), (n_meta, n_meta))
    u1 = t.fft.fft2(comp_amp)
    image_field_1 = t.fft.ifft2(u1 * t.conj(t.fft.fftshift(t.exp(1j * k * (3e-3 * bb)))))
    image_field_2 = t.fft.ifft2(u1 * t.conj(t.fft.fftshift(t.exp(1j * k * (4e-3 * bb)))))
    image_field_3 = t.fft.ifft2(u1 * t.conj(t.fft.fftshift(t.exp(1j * k * (5e-3 * bb)))))
    image_field_4 = t.fft.ifft2(u1 * t.conj(t.fft.fftshift(t.exp(1j * k * (6e-3 * bb)))))
    image_field_5 = t.fft.ifft2(u1 * t.conj(t.fft.fftshift(t.exp(1j * k * (7e-3 * bb)))))
    image_field = t.stack([image_field_1, image_field_2, image_field_3, image_field_4, image_field_5], dim=2)
    return image_field


def denorm_parameters(Dx_list_norm, Dy_list_norm):
    Dx_list_denorm = Dx_list_norm * (180 - 20) + 20
    Dy_list_denorm = Dy_list_norm * (180 - 20) + 20
    return Dx_list_denorm, Dy_list_denorm


def denorm_theta(theta_list_norm):
    theta_denorm = theta_list_norm * t.pi
    return theta_denorm


def replace_out_rot(Dx_denorm, Dy_denorm, theta_denorm, device):
    theta_filtered = theta_denorm
    index1 = (t.sqrt((Dx_denorm / 2) ** 2 + (Dy_denorm / 2) ** 2) * t.cos(
        -t.abs(theta_denorm) + t.arctan(Dy_denorm / Dx_denorm))) > 100
    index2 = (t.sqrt((Dx_denorm / 2) ** 2 + (Dy_denorm / 2) ** 2) * t.sin(
        t.abs(theta_denorm) + t.arctan(Dy_denorm / Dx_denorm))) > 100
    index = index1 + index2
    index_2 = t.sqrt((Dx_denorm / 2) ** 2 + (Dy_denorm / 2) ** 2) > 100

    theta_filtered[index_2 * index] = t.min(t.abs(t.arcsin(100 / t.sqrt((Dx_denorm / 2) ** 2 + (Dy_denorm / 2) ** 2)[index_2 * index])-t.arctan(Dy_denorm / Dx_denorm)[index_2 * index]),
                                            t.abs(t.arccos(100 / t.sqrt((Dx_denorm / 2) ** 2 + (Dy_denorm / 2) ** 2)[index_2 * index]) - t.arctan(Dy_denorm / Dx_denorm)[index_2 * index]))
    return theta_filtered


def main():
    device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
    network3d = Network3d()
    network3d.to(device)
    # 已经训练好的网络结构
    network = t.load('one_pillar_new_data.pth', map_location=device)
    network.to(device)
    for param in network.parameters():
        param.requires_grad = False

    fre = 0.75
    fre_index = int((fre - 0.3) / ((1.2 - 0.3) / 1000))
    # 损失函数
    loss_fn = nn.MSELoss().to(device)
    # 优化器
    optimizer = optim.Adam(network3d.parameters(), lr=1e-3)
    scheduler = t.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.9)
    iterations = 60000
    writer = SummaryWriter('./RS_ABCDEFGHIJ_pw_0.75T_15-30-45-60_yx')
    for epoch in range(iterations):
        print("------第{}轮训练开始------".format(epoch + 1))
        path_ = "/home/gyx/Documents/letters/letters_oneplane"
        path_list = os.listdir(path_)
        path_random = random.choice(path_list)
        image_path = path_ + '/' + path_random
        im = np.load(image_path)
        inputs = get_dataset(im).to(t.float32).to(device)
        struct_para_list = network3d(inputs)
        Dx_list = struct_para_list[0].flatten()
        Dy_list = struct_para_list[1].flatten()
        theta_list = struct_para_list[2].flatten()
        # 解归一化参数
        Dx_list_denorm, Dy_list_denorm = denorm_parameters(Dx_list_norm=Dx_list, Dy_list_norm=Dy_list)
        theta_denorm = denorm_theta(theta_list)
        # 筛选数据
        theta_filtered = replace_out_rot(Dx_denorm=Dx_list_denorm, Dy_denorm=Dy_list_denorm, theta_denorm=theta_denorm,
                                         device=device)
        struct_para = t.column_stack([Dx_list_denorm, Dy_list_denorm])  # 输入给已训练的正向预测网络
        # 选频点，后续试图将频点整合进channel数中；（现在仅一个频点一个channel）
        outputs = network(struct_para)[:, [fre_index, 1001 + fre_index, 2002 + fre_index, 3003 + fre_index]].to(device)
        # 添加柱子旋转因子 ，并转为64*64*4的二维频谱分布
        outputs_rot = compute_rotate(outputs[:, 0] + 1j * outputs[:, 1], 0, 0, outputs[:, 2] + 1j * outputs[:, 3],
                                     theta_filtered)
        # outputs_2d_xx = t.reshape(outputs_rot[0], (64, 64)).to(device)
        # outputs_2d_xy = t.reshape(outputs_rot[1], (64, 64)).to(device)
        outputs_2d_yx = t.reshape(outputs_rot[2], (64, 64)).to(device)
        # outputs_2d_yy = t.reshape(outputs_rot[3], (100, 100))
        amplitude_metasurface = t.abs(outputs_2d_yx.real + 1j * outputs_2d_yx.imag).to(device)
        phase_metasurface = t.angle(outputs_2d_yx.real + 1j * outputs_2d_yx.imag).to(device)
        # 瑞利-索莫非
        image_plane = t.abs(metasurface_to_image_rayleigh_sommerfeld(
            freq=fre,
            amp_metasurface=amplitude_metasurface,
            pha_metasurface=phase_metasurface,
            device=device
        )).to(device)
        # 角谱法
        # image_plane = t.abs(angular_spectrum_method(
        #     freq=fre,
        #     amp_metasurface=amplitude_metasurface,
        #     pha_metasurface=phase_metasurface,
        #     device=device
        # )).to(device)
        image_plane = t.reshape(image_plane, (-1, 1, 64, 64, 5)).to(device)
        # 成像振幅归一化
        # image_norm = t.reshape(spectrum_normalize(image_plane), (-1, 1, 64, 64, 5)).to(device)
        inputs_64 = t.nn.functional.interpolate(inputs, size=(64, 64, 5))

        train_loss = loss_fn(image_plane[0, 0, :, :, :], inputs_64[0, 0, :, :, :])
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        writer.add_scalar('train_loss', train_loss, epoch)
        # tensorboard 可视化振幅的分布
        writer.add_images(
            tag="predict_image_amplitude",
            img_tensor=image_plane[0, :, :, :, :].permute(3, 0, 1, 2),
            global_step=epoch,
            dataformats="NCHW"  # 第一维度, 第二维度, ...
        )
        writer.add_images(
            tag="origin_image_amplitude",
            img_tensor=inputs[0, :, :, :, :].permute(3, 0, 1, 2),
            global_step=epoch,
            dataformats="NCHW"  # 第一维度, 第二维度, ...
        )
        print("训练次数{}，loss：{}".format(epoch, train_loss.item()))
        if epoch % 100 == 0:
            t.save(network3d, "RS_ABCDEFGHIJ_pw_0.75T_15-30-45-60_yx.pth")
        scheduler.step()


if __name__ == '__main__':
    main()
