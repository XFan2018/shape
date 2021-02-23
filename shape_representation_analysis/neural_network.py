import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sparse_coding import im2poly, equal_arclength
import torch_interpolations


def conv_block(in_f, out_f, activation='relu', *args, **kwargs):
    activations = activation_func(activation)

    return nn.Sequential(
        nn.Conv2d(in_f, out_f, *args, **kwargs),
        nn.BatchNorm2d(out_f),
        activations[activation]
    )


def activation_func(activation):
    return nn.ModuleDict(
        {'leaky_relu': nn.LeakyReLU(negative_slope=0.2, inplace=True),
         'tanh': nn.Tanh()}
    )[activation]


class Net(nn.Module):
    def __init__(self, layer_list):
        super(Net, self).__init__()
        self.layer_number = len(layer_list)
        if self.layer_number == 3:
            self.fc1 = nn.Linear(layer_list[0], layer_list[1])
            self.fc2 = nn.Linear(layer_list[1], layer_list[2])
        elif self.layer_number == 4:
            self.fc1 = nn.Linear(layer_list[0], layer_list[1])
            self.fc2 = nn.Linear(layer_list[1], layer_list[2])
            self.fc3 = nn.Linear(layer_list[2], layer_list[3])
        elif self.layer_number == 5:
            self.fc1 = nn.Linear(layer_list[0], layer_list[1])
            self.fc2 = nn.Linear(layer_list[1], layer_list[2])
            self.fc3 = nn.Linear(layer_list[2], layer_list[3])
            self.fc4 = nn.Linear(layer_list[3], layer_list[4])
        else:
            raise Exception("layer number should be between 3 and 5")

    def forward(self, input_data):
        if self.layer_number == 3:
            input_data = F.relu(self.fc1(input_data))
            input_data = self.fc2(input_data)
            return input_data
        elif self.layer_number == 4:
            input_data = F.relu(self.fc1(input_data))
            input_data = F.relu(self.fc2(input_data))
            input_data = self.fc3(input_data)
            return input_data
        else:
            input_data = F.relu(self.fc1(input_data))
            input_data = F.relu(self.fc2(input_data))
            input_data = F.relu(self.fc3(input_data))
            input_data = self.fc4(input_data)
            return input_data


class TurningAngleNet(nn.Module):
    def __init__(self, input_layer, hidden_layer1, hidden_layer2, output_layer, kernel_size):
        """
        :param input_layer: number of nodes in the input layer
        :param hidden_layer: number of nodes in the hidden layer
        :param output_layer: number of nodes in the output layer
        :param kernel_size: kernel size
        """
        super(TurningAngleNet, self).__init__()
        if kernel_size % 2 != 1:
            raise Exception("kernel_size must be odd")
        self.input_layer = input_layer
        self.conv1d_1 = nn.Conv1d(1, 4, kernel_size=kernel_size, padding=1, padding_mode="circular")
        self.conv1d_2 = nn.Conv1d(4, 8, kernel_size=kernel_size, padding=1, padding_mode='zeros')
        self.conv1d_3 = nn.Conv1d(8, 16, kernel_size=kernel_size, padding=1, padding_mode='zeros')
        self.conv1d_4 = nn.Conv1d(16, 32, kernel_size=kernel_size, padding=1, padding_mode='zeros')
        self.fc1 = nn.Linear(input_layer, hidden_layer1)
        self.fc2 = nn.Linear(hidden_layer1, hidden_layer2)
        self.fc3 = nn.Linear(hidden_layer2, output_layer)
        self.kernel_size = kernel_size

    def forward(self, x):
        padding_number = (self.kernel_size - 1) // 2
        # x = torch.cat((x[:, -padding_number:], x, x[:, :padding_number]), dim=1)
        x.unsqueeze_(1)
        print(x.shape)
        x = F.relu(self.conv1d_1(x))
        print(x.shape)
        x = F.relu(self.conv1d_2(x))
        x = F.relu(self.conv1d_3(x))
        x = F.relu(self.conv1d_4(x))
        x = x.view((-1, self.input_layer))
        print(x.shape)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        print(x.shape)
        return x


class VGG11TurningAngle(nn.Module):
    def __init__(self, channel1, channel2, channel3, channel4, input1, input2, input3):
        super(VGG11TurningAngle, self).__init__()
        self.conv1d_1 = nn.Conv1d(1, channel1, kernel_size=3, padding=1, padding_mode="circular")  # 128   32
        self.pool1d_1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv1d_2 = nn.Conv1d(channel1, channel2, kernel_size=3, padding=1, padding_mode="circular")  # 64    16
        self.pool1d_2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv1d_3 = nn.Conv1d(channel2, channel3, kernel_size=3, padding=1, padding_mode="circular")  # 32    8
        self.conv1d_4 = nn.Conv1d(channel3, channel3, kernel_size=3, padding=1, padding_mode="circular")
        self.pool1d_3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv1d_5 = nn.Conv1d(channel3, channel4, kernel_size=3, padding=1, padding_mode="circular")  # 16   4
        self.conv1d_6 = nn.Conv1d(channel4, channel4, kernel_size=3, padding=1, padding_mode="circular")
        self.pool1d_4 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv1d_7 = nn.Conv1d(channel4, channel4, kernel_size=3, padding=1, padding_mode="circular")  # 8   2
        self.conv1d_8 = nn.Conv1d(channel4, channel4, kernel_size=3, padding=1, padding_mode="circular")
        self.pool1d_5 = nn.MaxPool1d(kernel_size=2, stride=2)  # 4
        self.input1 = input1
        self.fc1 = nn.Linear(input1, input2)
        self.fc2 = nn.Linear(input2, input3)
        self.fc3 = nn.Linear(input3, 17)

    def forward(self, x):
        x.unsqueeze_(1)
        print(x.shape)
        x = torch.tanh(self.conv1d_1(x))
        x = self.pool1d_1(x)
        x = torch.tanh(self.conv1d_2(x))
        x = self.pool1d_2(x)
        x = torch.tanh(self.conv1d_3(x))
        x = torch.tanh(self.conv1d_4(x))
        x = self.pool1d_3(x)
        x = torch.tanh(self.conv1d_5(x))
        x = torch.tanh(self.conv1d_6(x))
        x = self.pool1d_4(x)
        x = torch.tanh(self.conv1d_7(x))
        x = torch.tanh(self.conv1d_8(x))
        x = self.pool1d_5(x)
        x = x.view((-1, self.input1))
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        print(x.shape)
        return


class VGG16TurningAngle(nn.Module):
    def __init__(self, channel1, channel2, channel3, channel4, input1, input2, input3):
        super(VGG16TurningAngle, self).__init__()
        self.conv1d_1 = nn.Conv1d(1, channel1, kernel_size=3, padding=1, padding_mode="circular")  # 128   32
        self.conv1d_2 = nn.Conv1d(channel1, channel1, kernel_size=3, padding=1, padding_mode="circular")  # 128   32
        self.pool1d_1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv1d_3 = nn.Conv1d(channel1, channel2, kernel_size=3, padding=1, padding_mode="circular")  # 64    16
        self.conv1d_4 = nn.Conv1d(channel2, channel2, kernel_size=3, padding=1, padding_mode="circular")  # 64    16
        self.pool1d_2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv1d_5 = nn.Conv1d(channel2, channel3, kernel_size=3, padding=1, padding_mode="circular")  # 32    8
        self.conv1d_6 = nn.Conv1d(channel3, channel3, kernel_size=3, padding=1, padding_mode="circular")
        self.conv1d_7 = nn.Conv1d(channel3, channel3, kernel_size=3, padding=1, padding_mode="circular")
        self.pool1d_3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv1d_8 = nn.Conv1d(channel3, channel4, kernel_size=3, padding=1, padding_mode="circular")  # 16   4
        self.conv1d_9 = nn.Conv1d(channel4, channel4, kernel_size=3, padding=1, padding_mode="circular")
        self.conv1d_10 = nn.Conv1d(channel4, channel4, kernel_size=3, padding=1, padding_mode="circular")
        self.pool1d_4 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv1d_11 = nn.Conv1d(channel4, channel4, kernel_size=3, padding=1, padding_mode="circular")  # 8   2
        self.conv1d_12 = nn.Conv1d(channel4, channel4, kernel_size=3, padding=1, padding_mode="circular")
        self.conv1d_13 = nn.Conv1d(channel4, channel4, kernel_size=3, padding=1, padding_mode="circular")
        self.pool1d_5 = nn.MaxPool1d(kernel_size=2, stride=2)  # 1
        self.input1 = input1
        self.fc1 = nn.Linear(input1, input2)
        self.fc2 = nn.Linear(input2, input3)
        self.fc3 = nn.Linear(input3, 17)

    def forward(self, x):
        x.unsqueeze_(1)
        print(x.shape)
        x = torch.tanh(self.conv1d_1(x))
        x = torch.tanh(self.conv1d_2(x))
        x = self.pool1d_1(x)
        x = torch.tanh(self.conv1d_3(x))
        x = torch.tanh(self.conv1d_4(x))
        x = self.pool1d_2(x)
        x = torch.tanh(self.conv1d_5(x))
        print(x.shape)
        x = torch.tanh(self.conv1d_6(x))
        print(x.shape)
        x = torch.tanh(self.conv1d_7(x))
        x = self.pool1d_3(x)
        x = torch.tanh(self.conv1d_8(x))
        x = torch.tanh(self.conv1d_9(x))
        x = torch.tanh(self.conv1d_10(x))
        x = self.pool1d_4(x)
        x = torch.tanh(self.conv1d_11(x))
        x = torch.tanh(self.conv1d_12(x))
        x = torch.tanh(self.conv1d_13(x))
        x = self.pool1d_5(x)
        x = x.view((-1, self.input1))
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        print(x.shape)
        return x


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, batch_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o1 = nn.Linear(input_size + hidden_size, 32)
        self.i2o2 = nn.Linear(32, output_size)
        torch.nn.init.xavier_uniform(self.i2o1.weight)
        torch.nn.init.xavier_uniform(self.i2o2.weight)
        torch.nn.init.xavier_uniform(self.i2h.weight)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = F.tanh(self.i2h(combined))
        output1 = F.tanh(self.i2o1(combined))
        output2 = self.i2o2(output1)
        output = self.softmax(output2)
        return output, hidden

    def initHidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size).cuda()


class CNN2(nn.Module):
    def __init__(self, channel0, channel1, input1, input2, output):
        super(CNN2, self).__init__()
        self.conv1d_1 = nn.Conv1d(channel0, channel1, kernel_size=3, padding=1, padding_mode="circular")
        self.pool1d_1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.input1 = input1
        self.fc1 = nn.Linear(input1, input2)
        self.fc2 = nn.Linear(input2, output)

    def forward(self, x):
        print(x.shape)
        x = torch.relu(self.conv1d_1(x))
        x = self.pool1d_1(x)
        x = x.view((-1, self.input1))
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        print(x.shape)
        return x


class VGG11PolygonCoordinates(nn.Module):
    def __init__(self, channel1, channel2, channel3, channel4, input1, input2, input3):
        super(VGG11PolygonCoordinates, self).__init__()
        self.conv1d_1 = nn.Conv1d(2, channel1, kernel_size=3, padding=1, padding_mode="circular")  # 128   32
        self.pool1d_1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv1d_2 = nn.Conv1d(channel1, channel2, kernel_size=3, padding=1, padding_mode="circular")  # 64    16
        self.pool1d_2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv1d_3 = nn.Conv1d(channel2, channel3, kernel_size=3, padding=1, padding_mode="circular")  # 32    8
        self.conv1d_4 = nn.Conv1d(channel3, channel3, kernel_size=3, padding=1, padding_mode="circular")
        self.pool1d_3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv1d_5 = nn.Conv1d(channel3, channel4, kernel_size=3, padding=1, padding_mode="circular")  # 16   4
        self.conv1d_6 = nn.Conv1d(channel4, channel4, kernel_size=3, padding=1, padding_mode="circular")
        self.pool1d_4 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv1d_7 = nn.Conv1d(channel4, channel4, kernel_size=3, padding=1, padding_mode="circular")  # 8   2
        self.conv1d_8 = nn.Conv1d(channel4, channel4, kernel_size=3, padding=1, padding_mode="circular")
        self.pool1d_5 = nn.MaxPool1d(kernel_size=2, stride=2)  # 4
        self.input1 = input1
        self.fc1 = nn.Linear(input1, input2)
        self.fc2 = nn.Linear(input2, input3)
        self.fc3 = nn.Linear(input3, 17)

    def forward(self, x):
        # x.unsqueeze_(1)
        print(x.shape)
        x = torch.tanh(self.conv1d_1(x))
        x = self.pool1d_1(x)
        x = torch.tanh(self.conv1d_2(x))
        x = self.pool1d_2(x)
        x = torch.tanh(self.conv1d_3(x))
        x = torch.tanh(self.conv1d_4(x))
        x = self.pool1d_3(x)
        x = torch.tanh(self.conv1d_5(x))
        x = torch.tanh(self.conv1d_6(x))
        x = self.pool1d_4(x)
        x = torch.tanh(self.conv1d_7(x))
        x = torch.tanh(self.conv1d_8(x))
        x = self.pool1d_5(x)
        x = x.view((-1, self.input1))
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        print(x.shape)
        return x


class VGG9PolygonCoordinates(nn.Module):
    def __init__(self, channel1, channel2, channel3, channel4, input1, input2, input3):
        super(VGG9PolygonCoordinates, self).__init__()
        self.conv1d_1 = nn.Conv1d(2, channel1, kernel_size=3, padding=1, padding_mode="circular")  # 128   32
        self.pool1d_1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv1d_2 = nn.Conv1d(channel1, channel2, kernel_size=3, padding=1, padding_mode="circular")  # 64    16
        self.pool1d_2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv1d_3 = nn.Conv1d(channel2, channel3, kernel_size=3, padding=1, padding_mode="circular")  # 32    8
        self.conv1d_4 = nn.Conv1d(channel3, channel3, kernel_size=3, padding=1, padding_mode="circular")
        self.pool1d_3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv1d_5 = nn.Conv1d(channel3, channel4, kernel_size=3, padding=1, padding_mode="circular")  # 16   4
        self.conv1d_6 = nn.Conv1d(channel4, channel4, kernel_size=3, padding=1, padding_mode="circular")
        self.pool1d_4 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.input1 = input1
        self.fc1 = nn.Linear(input1, input2)
        self.fc2 = nn.Linear(input2, input3)
        self.fc3 = nn.Linear(input3, 17)

    def forward(self, x):
        # x.unsqueeze_(1)
        print(x.shape)
        x = torch.tanh(self.conv1d_1(x))
        x = self.pool1d_1(x)
        x = torch.tanh(self.conv1d_2(x))
        x = self.pool1d_2(x)
        x = torch.tanh(self.conv1d_3(x))
        x = torch.tanh(self.conv1d_4(x))
        x = self.pool1d_3(x)
        x = torch.tanh(self.conv1d_5(x))
        x = torch.tanh(self.conv1d_6(x))
        x = self.pool1d_4(x)
        x = x.view((-1, self.input1))
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        print(x.shape)
        return x


class VGG7PolygonCoordinates(nn.Module):
    def __init__(self, channel1, channel2, channel3, input1, input2, input3):
        super(VGG7PolygonCoordinates, self).__init__()
        self.conv1d_1 = nn.Conv1d(2, channel1, kernel_size=3, padding=1, padding_mode="circular")  # 128   32
        self.pool1d_1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv1d_2 = nn.Conv1d(channel1, channel2, kernel_size=3, padding=1, padding_mode="circular")  # 64    16
        self.pool1d_2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv1d_3 = nn.Conv1d(channel2, channel3, kernel_size=3, padding=1, padding_mode="circular")  # 32    8
        self.conv1d_4 = nn.Conv1d(channel3, channel3, kernel_size=3, padding=1, padding_mode="circular")
        self.pool1d_3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.input1 = input1
        self.fc1 = nn.Linear(input1, input2)
        self.fc2 = nn.Linear(input2, input3)
        self.fc3 = nn.Linear(input3, 17)

    def forward(self, x):
        # x.unsqueeze_(1)
        print(x.shape)
        x = torch.tanh(self.conv1d_1(x))
        x = self.pool1d_1(x)
        x = torch.tanh(self.conv1d_2(x))
        x = self.pool1d_2(x)
        x = torch.tanh(self.conv1d_3(x))
        x = torch.tanh(self.conv1d_4(x))
        x = self.pool1d_3(x)
        x = x.view((-1, self.input1))
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        print(x.shape)
        return x


class VGG6PolygonCoordinates(nn.Module):
    def __init__(self, channel1, channel2, channel3, input1, input2, input3):
        super(VGG6PolygonCoordinates, self).__init__()
        self.conv1d_1 = nn.Conv1d(2, channel1, kernel_size=3, padding=1, padding_mode="circular")  # 128   32
        self.pool1d_1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv1d_2 = nn.Conv1d(channel1, channel2, kernel_size=3, padding=1, padding_mode="circular")  # 64    16
        self.pool1d_2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv1d_3 = nn.Conv1d(channel2, channel3, kernel_size=3, padding=1, padding_mode="circular")  # 32    8
        self.pool1d_3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.input1 = input1
        self.fc1 = nn.Linear(input1, input2)
        self.fc2 = nn.Linear(input2, input3)
        self.fc3 = nn.Linear(input3, 17)

    def forward(self, x):
        # x.unsqueeze_(1)
        print(x.shape)
        x = torch.tanh(self.conv1d_1(x))
        x = self.pool1d_1(x)
        x = torch.tanh(self.conv1d_2(x))
        x = self.pool1d_2(x)
        x = torch.tanh(self.conv1d_3(x))
        x = self.pool1d_3(x)
        x = x.view((-1, self.input1))
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        print(x.shape)
        return x


class VGG6PolygonCoordinates_dropout(nn.Module):
    def __init__(self, channel1, channel2, channel3, input1, input2, input3):
        super(VGG6PolygonCoordinates_dropout, self).__init__()
        self.conv1d_1 = nn.Conv1d(2, channel1, kernel_size=3, padding=1, padding_mode="circular")  # 128   32
        self.conv1d_1_bn = nn.BatchNorm1d(channel1)
        self.pool1d_1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv1d_2 = nn.Conv1d(channel1, channel2, kernel_size=3, padding=1, padding_mode="circular")  # 64    16
        self.conv1d_2_bn = nn.BatchNorm1d(channel2)
        self.pool1d_2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv1d_3 = nn.Conv1d(channel2, channel3, kernel_size=3, padding=1, padding_mode="circular")  # 32    8
        self.conv1d_3_bn = nn.BatchNorm1d(channel3)
        self.pool1d_3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.input1 = input1
        self.fc1 = nn.Linear(input1, input2)
        self.fc1_bn = nn.BatchNorm1d(input2)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(input2, input3)
        self.fc2_bn = nn.BatchNorm1d(input3)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(input3, 17)
        # self.leaky_relu = activation_func("leaky_relu")

    def forward(self, x):
        # x.unsqueeze_(1)
        print(x.shape)
        x = torch.tanh(self.conv1d_1_bn(self.conv1d_1(x)))
        # x = self.leaky_relu(self.conv1d_1_bn(self.conv1d_1(x)))
        x = self.pool1d_1(x)
        x = torch.tanh(self.conv1d_2_bn(self.conv1d_2(x)))
        # x = self.leaky_relu(self.conv1d_2_bn(self.conv1d_2(x)))
        x = self.pool1d_2(x)
        x = torch.tanh(self.conv1d_3_bn(self.conv1d_3(x)))
        # x = self.leaky_relu(self.conv1d_3_bn(self.conv1d_3(x)))
        x = self.pool1d_3(x)
        x = x.view((-1, self.input1))
        x = torch.tanh(self.fc1_bn(self.fc1(self.dropout1(x))))
        # x = self.leaky_relu(self.fc1_bn(self.fc1(self.dropout1(x))))
        x = torch.tanh(self.fc2_bn(self.fc2(self.dropout2(x))))
        # x = self.leaky_relu(self.fc2_bn(self.fc2(self.dropout2(x))))
        x = self.fc3(x)
        print(x.shape)
        return x


class VGG5PolygonCoordinates_dropout_selfAttention(nn.Module):
    def __init__(self, channel1, channel2, channel3, input1, input2):
        super(VGG5PolygonCoordinates_dropout_selfAttention, self).__init__()
        self.conv1d_1 = nn.Conv1d(2, channel1, kernel_size=3, padding=1, padding_mode="circular")  # 128   32
        self.conv1d_1_bn = nn.BatchNorm1d(channel1)
        self.pool1d_1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv1d_2 = nn.Conv1d(channel1, channel2, kernel_size=3, padding=1, padding_mode="circular")  # 64    16
        self.conv1d_2_bn = nn.BatchNorm1d(channel2)
        self.pool1d_2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv1d_3 = nn.Conv1d(channel2, channel3, kernel_size=3, padding=1, padding_mode="circular")  # 32    8
        self.conv1d_3_bn = nn.BatchNorm1d(channel3)
        self.pool1d_3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.input1 = input1
        self.fc1 = nn.Linear(input1, input2)
        self.fc1_bn = nn.BatchNorm1d(input2)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(input2, 17)
        self.multi_head_attention = MultiHeadAttention(1, 2)  # 1 head and 2 dim (x, y)
        # self.leaky_relu = activation_func("leaky_relu")

    def forward(self, x):
        # x.unsqueeze_(1)
        print(x.shape)
        x = self.multi_head_attention(x, x, x)
        print(f"self attention output: {x.shape}")
        x = torch.tanh(self.conv1d_1_bn(self.conv1d_1(x)))
        # x = self.leaky_relu(self.conv1d_1_bn(self.conv1d_1(x)))
        x = self.pool1d_1(x)
        x = torch.tanh(self.conv1d_2_bn(self.conv1d_2(x)))
        # x = self.leaky_relu(self.conv1d_2_bn(self.conv1d_2(x)))
        x = self.pool1d_2(x)
        x = torch.tanh(self.conv1d_3_bn(self.conv1d_3(x)))
        # x = self.leaky_relu(self.conv1d_3_bn(self.conv1d_3(x)))
        x = self.pool1d_3(x)
        x = x.view((-1, self.input1))
        x = torch.tanh(self.fc1_bn(self.fc1(self.dropout1(x))))
        # x = self.leaky_relu(self.fc1_bn(self.fc1(self.dropout1(x))))
        x = self.fc2(x)
        print(x.shape)
        return x


class PolygonCoordinates_dropout_selfAttention(nn.Module):
    def __init__(self, head, input1, input2):
        super(PolygonCoordinates_dropout_selfAttention, self).__init__()
        self.input1 = input1
        self.input2 = input2
        self.fc1 = nn.Linear(input1, input2)
        self.fc1_bn = nn.BatchNorm1d(input2)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(input2, 17)
        self.multi_head_attention = MultiHeadAttention(head, 2)  # 1 head and 2 dim (x, y)

    def forward(self, x):
        print(x.shape)
        x = self.multi_head_attention(x, x, x)
        print(f"self attention output: {x.shape}")
        x = x.reshape((-1, self.input1))
        x = torch.tanh(self.fc1_bn(self.fc1(self.dropout1(x))))
        x = self.fc2(x)
        print(x.shape)
        return x


class VGG4PolygonCoordinates_dropout(nn.Module):
    def __init__(self, channel1, channel2, input1, input2):
        super(VGG4PolygonCoordinates_dropout, self).__init__()
        self.conv1d_1 = nn.Conv1d(2, channel1, kernel_size=3, padding=1, padding_mode="circular")  # 128   32

        self.pool1d_1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv1d_2 = nn.Conv1d(channel1, channel2, kernel_size=3, padding=1, padding_mode="circular")  # 64    16
        self.conv1d_2_bn = nn.BatchNorm1d(channel2)
        self.pool1d_2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.input1 = input1
        self.fc1 = nn.Linear(input1, input2)
        self.fc1_bn = nn.BatchNorm1d(input2)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(input2, 17)

    def forward(self, x):
        # x.unsqueeze_(1)
        print(x.shape)
        x = torch.tanh(self.conv1d_1(x))
        x = self.pool1d_1(x)
        x = torch.tanh(self.conv1d_2(x))
        x = self.pool1d_2(x)
        x = x.view((-1, self.input1))
        x = torch.tanh(self.fc1_bn(self.fc1(self.dropout1(x))))
        x = self.fc2(x)
        print(x.shape)
        return x


class VGG5PolygonCoordinatesSelfAttention(nn.Module):
    def __init__(self, channel1, channel2, channel3, input1, input2, head, d_model):
        super(VGG5PolygonCoordinatesSelfAttention, self).__init__()
        self.conv1d_1 = nn.Conv1d(2, channel1, kernel_size=3, padding=1, padding_mode="circular")  # 128   32
        self.conv1d_1_bn = nn.BatchNorm1d(channel1)
        self.pool1d_1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv1d_2 = nn.Conv1d(channel1, channel2, kernel_size=3, padding=1, padding_mode="circular")  # 64    16
        self.conv1d_2_bn = nn.BatchNorm1d(channel2)
        self.pool1d_2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv1d_3 = nn.Conv1d(channel2, channel3, kernel_size=3, padding=1, padding_mode="circular")  # 32    8
        self.conv1d_3_bn = nn.BatchNorm1d(channel3)
        self.pool1d_3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.multi_head_attention = MultiHeadAttention(head, d_model)
        self.input1 = input1
        self.fc1 = nn.Linear(input1, input2)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(input2, 17)

    def forward(self, x):
        # x.unsqueeze_(1)
        print(x.shape)
        x = torch.tanh(self.conv1d_1_bn(self.conv1d_1(x)))
        x = self.pool1d_1(x)
        x = torch.tanh(self.conv1d_2_bn(self.conv1d_2(x)))
        x = self.pool1d_2(x)
        x = torch.tanh(self.conv1d_3_bn(self.conv1d_3(x)))
        x = self.pool1d_3(x)
        x = self.multi_head_attention(x, x, x)
        print("attention output: ", x.shape)
        x = x.reshape((-1, self.input1))
        x = torch.tanh(self.fc1(self.dropout1(x)))
        x = self.fc2(x)
        print(x.shape)
        return x


class VGG16PolygonCoordinates(nn.Module):
    def __init__(self, channel1, channel2, channel3, channel4, input1, input2, input3):
        super(VGG16PolygonCoordinates, self).__init__()
        self.conv1d_1 = nn.Conv1d(2, channel1, kernel_size=3, padding=1, padding_mode="circular")  # 128   32
        self.conv1d_2 = nn.Conv1d(channel1, channel1, kernel_size=3, padding=1, padding_mode="circular")  # 128   32
        self.pool1d_1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv1d_3 = nn.Conv1d(channel1, channel2, kernel_size=3, padding=1, padding_mode="circular")  # 64    16
        self.conv1d_4 = nn.Conv1d(channel2, channel2, kernel_size=3, padding=1, padding_mode="circular")  # 64    16
        self.pool1d_2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv1d_5 = nn.Conv1d(channel2, channel3, kernel_size=3, padding=1, padding_mode="circular")  # 32    8
        self.conv1d_6 = nn.Conv1d(channel3, channel3, kernel_size=3, padding=1, padding_mode="circular")
        self.conv1d_7 = nn.Conv1d(channel3, channel3, kernel_size=3, padding=1, padding_mode="circular")
        self.pool1d_3 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv1d_8 = nn.Conv1d(channel3, channel4, kernel_size=3, padding=1, padding_mode="circular")  # 16   4
        self.conv1d_9 = nn.Conv1d(channel4, channel4, kernel_size=3, padding=1, padding_mode="circular")
        self.conv1d_10 = nn.Conv1d(channel4, channel4, kernel_size=3, padding=1, padding_mode="circular")
        self.pool1d_4 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv1d_11 = nn.Conv1d(channel4, channel4, kernel_size=3, padding=1, padding_mode="circular")  # 8   2
        self.conv1d_12 = nn.Conv1d(channel4, channel4, kernel_size=3, padding=1, padding_mode="circular")
        self.conv1d_13 = nn.Conv1d(channel4, channel4, kernel_size=3, padding=1, padding_mode="circular")
        self.pool1d_5 = nn.MaxPool1d(kernel_size=2, stride=2)  # 1
        self.input1 = input1
        self.fc1 = nn.Linear(input1, input2)
        self.fc2 = nn.Linear(input2, input3)
        self.fc3 = nn.Linear(input3, 17)

    def forward(self, x):
        # x.unsqueeze_(1)
        print(x.shape)
        x = torch.tanh(self.conv1d_1(x))
        x = torch.tanh(self.conv1d_2(x))
        x = self.pool1d_1(x)
        x = torch.tanh(self.conv1d_3(x))
        x = torch.tanh(self.conv1d_4(x))
        x = self.pool1d_2(x)
        x = torch.tanh(self.conv1d_5(x))
        print(x.shape)
        x = torch.tanh(self.conv1d_6(x))
        print(x.shape)
        x = torch.tanh(self.conv1d_7(x))
        x = self.pool1d_3(x)
        x = torch.tanh(self.conv1d_8(x))
        x = torch.tanh(self.conv1d_9(x))
        x = torch.tanh(self.conv1d_10(x))
        x = self.pool1d_4(x)
        x = torch.tanh(self.conv1d_11(x))
        x = torch.tanh(self.conv1d_12(x))
        x = torch.tanh(self.conv1d_13(x))
        x = self.pool1d_5(x)
        x = x.view((-1, self.input1))
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        print(x.shape)
        return x


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_size):
        super(LSTM, self).__init__()

        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers
        # define lstm layer
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers)
        self.linear = nn.Linear(self.hidden_size, self.output_size)
        self.hidden = self.initHidden(self.batch_size)

    def forward(self, input, batch_size):
        lstm_out, hidden = self.lstm(input, self.hidden)
        predictions = self.linear(lstm_out[-1].view(batch_size, -1))
        return predictions

    def initHidden(self, batch_size):
        # (hidden , cell)
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size).cuda(),
                torch.zeros(self.num_layers, batch_size, self.hidden_size).cuda())


def polygon_sets_transform(polygon, set_size):
    """
    :param polygon: polygon coordinates (length of sequence, batch size, coordinates x, y)
    :return: polygon coordinates sets (length of sequence, batch size, set of coordinates x, y)
    """
    len_seq = polygon.shape[0]
    polygon_sets = []
    for i in range(len_seq):
        polygon_set = []
        for j in range(set_size):
            polygon_set.append(polygon[(i + j) % len_seq])
        polygon_set = tuple(polygon_set)
        polygon_set = torch.cat(polygon_set, dim=1)
        polygon_sets.append(polygon_set)
    polygon_sets = torch.stack(tuple(polygon_sets))
    return polygon_sets


class AE(nn.Module):
    def __init__(self, input_shape, hidden_shape, output_shape):
        super(AE, self).__init__()
        self.encoder_hidden_layer = nn.Linear(
            in_features=input_shape, out_features=hidden_shape
        )
        self.encoder_output_layer = nn.Linear(
            in_features=hidden_shape, out_features=output_shape
        )
        self.decoder_hidden_layer = nn.Linear(
            in_features=output_shape, out_features=hidden_shape
        )
        self.decoder_output_layer = nn.Linear(
            in_features=hidden_shape, out_features=input_shape
        )

    def forward(self, features):
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)
        code = self.encoder_output_layer(activation)
        code = torch.relu(code)
        activation = self.decoder_hidden_layer(code)
        activation = torch.relu(activation)
        activation = self.decoder_output_layer(activation)
        # reconstructed = torch.relu(activation)
        return activation


class AE2(nn.Module):
    def __init__(self, input_shape, hidden_shape1, hidden_shape2, output_shape):
        super(AE2, self).__init__()
        self.encoder_hidden_layer = nn.Linear(
            in_features=input_shape, out_features=hidden_shape1
        )
        self.encoder_hidden_layer2 = nn.Linear(
            in_features=hidden_shape1, out_features=hidden_shape2
        )
        self.encoder_output_layer = nn.Linear(
            in_features=hidden_shape2, out_features=output_shape
        )
        self.decoder_hidden_layer = nn.Linear(
            in_features=output_shape, out_features=hidden_shape2
        )
        self.decoder_hidden_layer2 = nn.Linear(
            in_features=hidden_shape2, out_features=hidden_shape1
        )
        self.decoder_output_layer = nn.Linear(
            in_features=hidden_shape1, out_features=input_shape
        )

    def forward(self, features):
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)
        activation = self.encoder_hidden_layer2(activation)
        activation = torch.relu(activation)
        code = self.encoder_output_layer(activation)
        code = torch.relu(code)
        activation = self.decoder_hidden_layer(code)
        activation = torch.relu(activation)
        activation = self.decoder_hidden_layer2(activation)
        activation = torch.relu(activation)
        activation = self.decoder_output_layer(activation)
        # reconstructed = torch.relu(activation)
        return activation


class ConvAE(nn.Module):
    def __init__(self, channel1, channel2, channel3):
        super(ConvAE, self).__init__()
        self.conv1d_1 = nn.Conv1d(2, channel1, kernel_size=3, padding=1, padding_mode="circular")  # 128   32
        self.pool1d_1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv1d_2 = nn.Conv1d(channel1, channel2, kernel_size=3, padding=1, padding_mode="circular")  # 64    16
        self.pool1d_2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv1d_3 = nn.Conv1d(channel2, channel3, kernel_size=3, padding=1, padding_mode="circular")  # 32    8
        self.conv1d_4 = nn.Conv1d(channel3, channel3, kernel_size=3, padding=1, padding_mode="circular")
        self.pool1d_3 = nn.MaxPool1d(kernel_size=2, stride=2)
        # (channel:32， 4) -> (channel:32, 8)
        self.transpose_conv1d_1 = nn.ConvTranspose1d(in_channels=channel3,
                                                     out_channels=channel3,
                                                     stride=1,
                                                     kernel_size=3,
                                                     padding=0,
                                                     output_padding=0,
                                                     dilation=2,
                                                     padding_mode="zeros")
        # (channel:32， 8) -> (channel:16, 16)
        self.transpose_conv1d_2 = nn.ConvTranspose1d(in_channels=channel3,
                                                     out_channels=channel2,
                                                     stride=1,
                                                     kernel_size=3,
                                                     padding=0,
                                                     output_padding=0,
                                                     dilation=4,
                                                     padding_mode="zeros")
        # (channel:16, 16) -> (channel: 2, 32)
        self.transpose_conv1d_3 = nn.ConvTranspose1d(in_channels=channel2,
                                                     out_channels=2,
                                                     stride=1,
                                                     kernel_size=3,
                                                     padding=0,
                                                     output_padding=0,
                                                     dilation=8,
                                                     padding_mode="zeros")

    def forward(self, features):
        activation = torch.relu(self.conv1d_1(features))
        activation = self.pool1d_1(activation)
        activation = torch.relu(self.conv1d_2(activation))
        activation = self.pool1d_2(activation)
        activation = torch.relu(self.conv1d_3(activation))
        activation = torch.relu(self.conv1d_4(activation))
        activation = self.pool1d_3(activation)
        activation = torch.relu(self.transpose_conv1d_1(activation))
        activation = torch.relu(self.transpose_conv1d_2(activation))
        activation = self.transpose_conv1d_3(activation)

        return activation


class ConvAE2(nn.Module):
    def __init__(self, channel1, channel2, channel3):
        super(ConvAE2, self).__init__()
        self.conv1d_1 = nn.Conv1d(2, channel1, kernel_size=3, padding=1, padding_mode="circular")  # 128   32
        self.pool1d_1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv1d_2 = nn.Conv1d(channel1, channel2, kernel_size=3, padding=1, padding_mode="circular")  # 64    16
        self.pool1d_2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv1d_3 = nn.Conv1d(channel2, channel3, kernel_size=3, padding=1, padding_mode="circular")  # 32    8
        self.conv1d_4 = nn.Conv1d(channel3, channel3, kernel_size=3, padding=1, padding_mode="circular")
        self.pool1d_3 = nn.MaxPool1d(kernel_size=2, stride=2)
        # (channel:32， 4) -> (channel:32, 8)
        self.transpose_conv1d_1 = nn.ConvTranspose1d(in_channels=channel3,
                                                     out_channels=channel3,
                                                     stride=1,
                                                     kernel_size=5,
                                                     padding=0,
                                                     output_padding=0,
                                                     dilation=1,
                                                     padding_mode="zeros")
        # (channel:32， 8) -> (channel:16, 16)
        self.transpose_conv1d_2 = nn.ConvTranspose1d(in_channels=channel3,
                                                     out_channels=channel2,
                                                     stride=1,
                                                     kernel_size=9,
                                                     padding=0,
                                                     output_padding=0,
                                                     dilation=1,
                                                     padding_mode="zeros")
        # (channel:16, 16) -> (channel: 2, 32)
        self.transpose_conv1d_3 = nn.ConvTranspose1d(in_channels=channel2,
                                                     out_channels=2,
                                                     stride=1,
                                                     kernel_size=17,
                                                     padding=0,
                                                     output_padding=0,
                                                     dilation=1,
                                                     padding_mode="zeros")

    def forward(self, features):
        activation = torch.relu(self.conv1d_1(features))
        activation = self.pool1d_1(activation)
        activation = torch.relu(self.conv1d_2(activation))
        activation = self.pool1d_2(activation)
        activation = torch.relu(self.conv1d_3(activation))
        activation = torch.relu(self.conv1d_4(activation))
        activation = self.pool1d_3(activation)
        activation = torch.relu(self.transpose_conv1d_1(activation))
        activation = torch.relu(self.transpose_conv1d_2(activation))
        activation = self.transpose_conv1d_3(activation)

        return activation


# circular padding & upsampling by increasing number of layers
class ConvAE3(nn.Module):
    def __init__(self, channel1, channel2, channel3, circular):
        super(ConvAE3, self).__init__()
        self.circular = circular
        self.conv1d_1 = nn.Conv1d(2, channel1, kernel_size=3, padding=1, padding_mode="circular")  # 128   32
        self.pool1d_1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv1d_2 = nn.Conv1d(channel1, channel2, kernel_size=3, padding=1, padding_mode="circular")  # 64    16
        self.pool1d_2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv1d_3 = nn.Conv1d(channel2, channel3, kernel_size=3, padding=1, padding_mode="circular")  # 32    8
        self.conv1d_4 = nn.Conv1d(channel3, channel3, kernel_size=3, padding=1, padding_mode="circular")
        self.pool1d_3 = nn.MaxPool1d(kernel_size=2, stride=2)
        # (channel:32， 4) -> (channel:32, 8)
        if circular:
            padding = 2
        else:
            padding = 0
        self.transpose_conv1d_1 = nn.ConvTranspose1d(in_channels=channel3,
                                                     out_channels=channel3,
                                                     stride=1,
                                                     kernel_size=5,
                                                     padding=padding,
                                                     output_padding=0,
                                                     dilation=1,
                                                     padding_mode="zeros")
        # (channel:32， 8) -> (channel:28, 12)
        self.transpose_conv1d_2 = nn.ConvTranspose1d(in_channels=channel3,
                                                     out_channels=28,
                                                     stride=1,
                                                     kernel_size=5,
                                                     padding=padding,
                                                     output_padding=0,
                                                     dilation=1,
                                                     padding_mode="zeros")
        # (channel:28, 12) -> (channel: 24, 16)
        self.transpose_conv1d_3 = nn.ConvTranspose1d(in_channels=28,
                                                     out_channels=24,
                                                     stride=1,
                                                     kernel_size=5,
                                                     padding=padding,
                                                     output_padding=0,
                                                     dilation=1,
                                                     padding_mode="zeros")

        # (channel:24, 16) -> (channel: 20, 20)
        self.transpose_conv1d_4 = nn.ConvTranspose1d(in_channels=24,
                                                     out_channels=20,
                                                     stride=1,
                                                     kernel_size=5,
                                                     padding=padding,
                                                     output_padding=0,
                                                     dilation=1,
                                                     padding_mode="zeros")

        # (channel:20, 20) -> (channel: 16, 24)
        self.transpose_conv1d_5 = nn.ConvTranspose1d(in_channels=20,
                                                     out_channels=16,
                                                     stride=1,
                                                     kernel_size=5,
                                                     padding=padding,
                                                     output_padding=0,
                                                     dilation=1,
                                                     padding_mode="zeros")

        # (channel:16, 24) -> (channel: 12, 28)
        self.transpose_conv1d_6 = nn.ConvTranspose1d(in_channels=16,
                                                     out_channels=12,
                                                     stride=1,
                                                     kernel_size=5,
                                                     padding=padding,
                                                     output_padding=0,
                                                     dilation=1,
                                                     padding_mode="zeros")

        # (channel:12, 28) -> (channel: 2, 32)
        self.transpose_conv1d_7 = nn.ConvTranspose1d(in_channels=12,
                                                     out_channels=2,
                                                     stride=1,
                                                     kernel_size=5,
                                                     padding=padding,
                                                     output_padding=0,
                                                     dilation=1,
                                                     padding_mode="zeros")

    def forward(self, features):
        activation = torch.relu(self.conv1d_1(features))
        activation = self.pool1d_1(activation)
        activation = torch.relu(self.conv1d_2(activation))
        activation = self.pool1d_2(activation)
        activation = torch.relu(self.conv1d_3(activation))
        activation = torch.relu(self.conv1d_4(activation))
        activation = self.pool1d_3(activation)
        if self.circular:
            activation = self.circular_padding(activation, 5)
        activation = torch.relu(self.transpose_conv1d_1(activation))
        print(activation.shape)
        if self.circular:
            activation = self.circular_padding(activation, 5)
        activation = torch.relu(self.transpose_conv1d_2(activation))
        print(activation.shape)
        if self.circular:
            activation = self.circular_padding(activation, 5)
        activation = torch.relu(self.transpose_conv1d_3(activation))
        print(activation.shape)
        if self.circular:
            activation = self.circular_padding(activation, 5)
        activation = torch.relu(self.transpose_conv1d_4(activation))
        print(activation.shape)
        if self.circular:
            activation = self.circular_padding(activation, 5)
        activation = torch.relu(self.transpose_conv1d_5(activation))
        print(activation.shape)
        if self.circular:
            activation = self.circular_padding(activation, 5)
        activation = torch.relu(self.transpose_conv1d_6(activation))
        print(activation.shape)
        if self.circular:
            activation = self.circular_padding(activation, 5)
        activation = self.transpose_conv1d_7(activation)
        print(activation.shape)
        return activation

    def circular_padding(self, features, kernel_size):
        padding_size = (kernel_size - 1) // 2
        points_num = features.shape[2]
        bottom_padding = features[:, :, 0:padding_size]
        # print("bottom_padding", bottom_padding)
        top_padding = features[:, :, points_num - padding_size:points_num]
        # print("top_padding", top_padding)
        result = features.clone()
        result = torch.cat((top_padding, result, bottom_padding), dim=2)
        # print("padding result", result)
        return result


# circular padding & upsampling by stride
class ConvAE4(nn.Module):
    def __init__(self, channel0, channel1, channel2, channel3, circular):
        super(ConvAE4, self).__init__()
        self.circular = circular
        self.conv1d_1 = nn.Conv1d(channel0, channel1, kernel_size=3, padding=1, padding_mode="circular")  # 128   32
        self.pool1d_1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv1d_2 = nn.Conv1d(channel1, channel2, kernel_size=3, padding=1, padding_mode="circular")  # 64    16
        self.pool1d_2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv1d_3 = nn.Conv1d(channel2, channel3, kernel_size=3, padding=1, padding_mode="circular")  # 32    8
        self.conv1d_4 = nn.Conv1d(channel3, channel3, kernel_size=3, padding=1, padding_mode="circular")
        self.pool1d_3 = nn.MaxPool1d(kernel_size=2, stride=2)
        # (channel:32， 4) -> (channel:16, 8)
        if circular:
            padding = 2
        else:
            padding = 0
        self.transpose_conv1d_1 = nn.ConvTranspose1d(in_channels=channel3,
                                                     out_channels=24,
                                                     stride=2,
                                                     kernel_size=5,
                                                     padding=4,
                                                     output_padding=1,
                                                     dilation=1,
                                                     padding_mode="zeros")
        # (channel:16， 8) -> (channel:8, 16)
        self.transpose_conv1d_2 = nn.ConvTranspose1d(in_channels=24,
                                                     out_channels=16,
                                                     stride=2,
                                                     kernel_size=5,
                                                     padding=4,
                                                     output_padding=1,
                                                     dilation=1,
                                                     padding_mode="zeros")
        # (channel:8， 16) -> (channel:2, 32)
        self.transpose_conv1d_3 = nn.ConvTranspose1d(in_channels=16,
                                                     out_channels=channel0,
                                                     stride=2,
                                                     kernel_size=5,
                                                     padding=4,
                                                     output_padding=1,
                                                     dilation=1,
                                                     padding_mode="zeros")

    def forward(self, features):
        print(features.shape)
        activation = torch.relu(self.conv1d_1(features))
        activation = self.pool1d_1(activation)
        print(activation.shape)
        activation = torch.relu(self.conv1d_2(activation))
        activation = self.pool1d_2(activation)
        print(activation.shape)
        activation = torch.relu(self.conv1d_3(activation))
        activation = torch.relu(self.conv1d_4(activation))
        activation = self.pool1d_3(activation)
        print(activation.shape)
        if self.circular:
            activation = self.circular_padding(activation, 3)
        activation = torch.relu(self.transpose_conv1d_1(activation))
        print(activation.shape)
        if self.circular:
            activation = self.circular_padding(activation, 3)
        activation = torch.relu(self.transpose_conv1d_2(activation))
        print(activation.shape)
        if self.circular:
            activation = self.circular_padding(activation, 3)
        activation = self.transpose_conv1d_3(activation)
        print(activation.shape)
        return activation

    def circular_padding(self, features, kernel_size):
        padding_size = (kernel_size - 1) // 2
        points_num = features.shape[2]
        bottom_padding = features[:, :, 0:padding_size]
        # print("bottom_padding", bottom_padding)
        top_padding = features[:, :, points_num - padding_size:points_num]
        # print("top_padding", top_padding)
        result = features.clone()
        result = torch.cat((top_padding, result, bottom_padding), dim=2)
        # print("padding result", result)
        return result


# circular padding & upsampling by stride, 1 conv, 1 pooling, 1 de_conv
class ConvAE1_1(nn.Module):
    def __init__(self, channel0, channel1, circular):
        super(ConvAE1_1, self).__init__()
        self.circular = circular
        self.conv1d_1 = nn.Conv1d(channel0, channel1, kernel_size=3, padding=1, padding_mode="circular")  # 128   32
        self.pool1d_1 = nn.MaxPool1d(kernel_size=2, stride=2)
        # (channel:8， 16) -> (channel:2, 32)
        if circular:
            padding = 2
        else:
            padding = 0
        self.transpose_conv1d_1 = nn.ConvTranspose1d(in_channels=channel1,
                                                     out_channels=channel0,
                                                     stride=2,
                                                     kernel_size=5,
                                                     padding=4,
                                                     output_padding=1,
                                                     dilation=1,
                                                     padding_mode="zeros")

    def forward(self, features):
        print(features.shape)
        activation = torch.relu(self.conv1d_1(features))
        activation = self.pool1d_1(activation)
        print(activation.shape)

        if self.circular:
            activation = self.circular_padding(activation, 3)
        activation = self.transpose_conv1d_1(activation)
        print(activation.shape)
        return activation

    def circular_padding(self, features, kernel_size):
        padding_size = (kernel_size - 1) // 2
        points_num = features.shape[2]
        bottom_padding = features[:, :, 0:padding_size]
        # print("bottom_padding", bottom_padding)
        top_padding = features[:, :, points_num - padding_size:points_num]
        # print("top_padding", top_padding)
        result = features.clone()
        result = torch.cat((top_padding, result, bottom_padding), dim=2)
        # print("padding result", result)
        return result


# circular padding & upsampling by stride, 2 conv, 2 pooling, 2 de_conv
class ConvAE2_2(nn.Module):
    def __init__(self, channel0, channel1, channel2, channel3, circular):
        super(ConvAE2_2, self).__init__()
        self.circular = circular
        self.conv1d_1 = nn.Conv1d(channel0, channel1, kernel_size=3, padding=1, padding_mode="circular")  # 128   32
        self.pool1d_1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv1d_2 = nn.Conv1d(channel1, channel2, kernel_size=3, padding=1, padding_mode="circular")  # 64    16
        self.pool1d_2 = nn.MaxPool1d(kernel_size=2, stride=2)
        # (channel:16， 8) -> (channel:8, 16)
        if circular:
            padding = 2
        else:
            padding = 0
        self.transpose_conv1d_1 = nn.ConvTranspose1d(in_channels=channel2,  # 16
                                                     out_channels=channel1,  # 8
                                                     stride=2,
                                                     kernel_size=5,
                                                     padding=4,
                                                     output_padding=1,
                                                     dilation=1,
                                                     padding_mode="zeros")
        # (channel:8, 16) -> (channel:2, 32)
        self.transpose_conv1d_2 = nn.ConvTranspose1d(in_channels=channel1,  # 8
                                                     out_channels=channel0,  # 2
                                                     stride=2,
                                                     kernel_size=5,
                                                     padding=4,
                                                     output_padding=1,
                                                     dilation=1,
                                                     padding_mode="zeros")

    def forward(self, features):
        print(features.shape)
        activation = torch.relu(self.conv1d_1(features))
        activation = self.pool1d_1(activation)
        activation = torch.relu(self.conv1d_2(activation))
        activation = self.pool1d_2(activation)
        print(activation.shape)

        if self.circular:
            activation = self.circular_padding(activation, 3)
        activation = torch.relu(self.transpose_conv1d_1(activation))
        print(activation.shape)
        if self.circular:
            activation = self.circular_padding(activation, 3)
        activation = self.transpose_conv1d_2(activation)
        print(activation.shape)
        return activation

    def circular_padding(self, features, kernel_size):
        padding_size = (kernel_size - 1) // 2
        points_num = features.shape[2]
        bottom_padding = features[:, :, 0:padding_size]
        # print("bottom_padding", bottom_padding)
        top_padding = features[:, :, points_num - padding_size:points_num]
        # print("top_padding", top_padding)
        result = features.clone()
        result = torch.cat((top_padding, result, bottom_padding), dim=2)
        # print("padding result", result)
        return result


class ConvAEEqualArcLength(nn.Module):
    def __init__(self, channel1, channel2, channel3, circular, points_num, device):
        super(ConvAEEqualArcLength, self).__init__()
        self.circular = circular
        self.points_num = points_num
        self.device = device
        self.conv1d_1 = nn.Conv1d(2, channel1, kernel_size=3, padding=1, padding_mode="circular")  # 128   32
        self.pool1d_1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv1d_2 = nn.Conv1d(channel1, channel2, kernel_size=3, padding=1, padding_mode="circular")  # 64    16
        self.pool1d_2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv1d_3 = nn.Conv1d(channel2, channel3, kernel_size=3, padding=1, padding_mode="circular")  # 32    8
        self.conv1d_4 = nn.Conv1d(channel3, channel3, kernel_size=3, padding=1, padding_mode="circular")
        self.pool1d_3 = nn.MaxPool1d(kernel_size=2, stride=2)

        if circular:
            padding = 2
        else:
            padding = 0
        # (channel:32， 4) -> (channel:32, 8)
        self.transpose_conv1d_1 = nn.ConvTranspose1d(in_channels=channel3,
                                                     out_channels=channel3,
                                                     stride=1,
                                                     kernel_size=5,
                                                     padding=padding,
                                                     output_padding=0,
                                                     dilation=1,
                                                     padding_mode="zeros")
        # (channel:32， 8) -> (channel:28, 12)
        self.transpose_conv1d_2 = nn.ConvTranspose1d(in_channels=channel3,
                                                     out_channels=28,
                                                     stride=1,
                                                     kernel_size=5,
                                                     padding=padding,
                                                     output_padding=0,
                                                     dilation=1,
                                                     padding_mode="zeros")
        # (channel:28, 12) -> (channel: 24, 16)
        self.transpose_conv1d_3 = nn.ConvTranspose1d(in_channels=28,
                                                     out_channels=24,
                                                     stride=1,
                                                     kernel_size=5,
                                                     padding=padding,
                                                     output_padding=0,
                                                     dilation=1,
                                                     padding_mode="zeros")

        # (channel:24, 16) -> (channel: 20, 20)
        self.transpose_conv1d_4 = nn.ConvTranspose1d(in_channels=24,
                                                     out_channels=20,
                                                     stride=1,
                                                     kernel_size=5,
                                                     padding=padding,
                                                     output_padding=0,
                                                     dilation=1,
                                                     padding_mode="zeros")

        # (channel:20, 20) -> (channel: 16, 24)
        self.transpose_conv1d_5 = nn.ConvTranspose1d(in_channels=20,
                                                     out_channels=16,
                                                     stride=1,
                                                     kernel_size=5,
                                                     padding=padding,
                                                     output_padding=0,
                                                     dilation=1,
                                                     padding_mode="zeros")

        # (channel:16, 24) -> (channel: 12, 28)
        self.transpose_conv1d_6 = nn.ConvTranspose1d(in_channels=16,
                                                     out_channels=12,
                                                     stride=1,
                                                     kernel_size=5,
                                                     padding=padding,
                                                     output_padding=0,
                                                     dilation=1,
                                                     padding_mode="zeros")

        # (channel:12, 28) -> (channel: 2, 32)
        self.transpose_conv1d_7 = nn.ConvTranspose1d(in_channels=12,
                                                     out_channels=2,
                                                     stride=1,
                                                     kernel_size=5,
                                                     padding=padding,
                                                     output_padding=0,
                                                     dilation=1,
                                                     padding_mode="zeros")

    def forward(self, features):
        activation = torch.relu(self.conv1d_1(features))
        activation = self.pool1d_1(activation)
        activation = torch.relu(self.conv1d_2(activation))
        activation = self.pool1d_2(activation)
        activation = torch.relu(self.conv1d_3(activation))
        activation = torch.relu(self.conv1d_4(activation))
        activation = self.pool1d_3(activation)
        if self.circular:
            activation = self.circular_padding(activation, 5)
        activation = torch.relu(self.transpose_conv1d_1(activation))
        if self.circular:
            activation = self.circular_padding(activation, 5)
        activation = torch.relu(self.transpose_conv1d_2(activation))
        if self.circular:
            activation = self.circular_padding(activation, 5)
        activation = torch.relu(self.transpose_conv1d_3(activation))
        if self.circular:
            activation = self.circular_padding(activation, 5)
        activation = torch.relu(self.transpose_conv1d_4(activation))
        if self.circular:
            activation = self.circular_padding(activation, 5)
        activation = torch.relu(self.transpose_conv1d_5(activation))
        if self.circular:
            activation = self.circular_padding(activation, 5)
        activation = torch.relu(self.transpose_conv1d_6(activation))
        if self.circular:
            activation = self.circular_padding(activation, 5)
        activation = self.transpose_conv1d_7(activation)
        return self.equal_arc_length(activation, self.points_num)

    def circular_padding(self, features, kernel_size):
        padding_size = (kernel_size - 1) // 2
        points_num = features.shape[2]
        bottom_padding = features[:, :, 0:padding_size]
        # print("bottom_padding", bottom_padding)
        top_padding = features[:, :, points_num - padding_size:points_num]
        # print("top_padding", top_padding)
        result = features.clone()
        result = torch.cat((top_padding, result, bottom_padding), dim=2)
        # print("padding result", result)
        return result

    def equal_arc_length(self, activation, points_num):
        result = []
        for polygon in activation:
            polygon = torch.transpose(polygon, 0, 1)
            polygon_x = torch.cat((polygon[:, 0], polygon[0, 0].view(1)))
            polygon_y = torch.cat((polygon[:, 1], polygon[0, 1].view(1)))
            arclength = torch.cat([torch.tensor([0]).to(self.device), torch.cumsum(
                torch.sqrt((polygon_x[1:] - polygon_x[:-1]) ** 2 + (polygon_y[1:] - polygon_y[:-1]) ** 2), dim=0)])
            target = arclength[-1] * torch.arange(points_num).to(self.device) / points_num

            gi_x = torch_interpolations.RegularGridInterpolator([arclength], polygon_x)
            gi_y = torch_interpolations.RegularGridInterpolator([arclength], polygon_y)
            fx = gi_x([target])
            fy = gi_y([target])
            new_polygon = torch.vstack((fx, fy))
            result.append(new_polygon)
        return torch.stack(result)


# input = Batch_size * seq_len * d_model.
class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def attention(self, q, k, v, d_k, mask=None, dropout=None):

        scores = torch.matmul(k.transpose(1, 2), q.transpose(1, 2).transpose(2, 3)) / math.sqrt(d_k)

        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)

        scores = F.softmax(scores, dim=-1)

        if dropout is not None:
            scores = dropout(scores)

        v = v.transpose(1, 2)
        output = torch.matmul(scores, v)
        output = output.transpose(1, 2)
        return output

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        # transpose to get dimensions bs * h * sl * d_model

        # k = k.transpose(1, 2)
        # q = q.transpose(1, 2)
        # v = v.transpose(1, 2)

        # perform linear operation and split into h heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # calculate attention using function we will define next
        scores = self.attention(q, k, v, self.d_k, mask, None)
        print(f"scores dim: {scores.shape}")

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous() \
            .view(bs, -1, self.d_model)

        output = self.out(concat)
        output = output.transpose(1, 2)
        return output


class PreActBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, activation="tanh"):
        super(PreActBlock, self).__init__()
        self.dropout1 = nn.Dropout(0.2)
        self.bn1 = nn.BatchNorm1d(in_planes)
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.dropout2 = nn.Dropout(0.2)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.activation_func = activation_func(activation)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = self.activation_func(self.bn1(x))
        # if number of planes changed, use kernel size = 1 convolution to change the shortcut planes number
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(self.dropout1(out))
        out = self.conv2(self.activation_func(self.bn2(self.dropout2(out))))
        out += shortcut
        return out


class PreActBottleneck(nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, activation="leaky_relu"):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm1d(in_planes)
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=1, bias=False)
        self.dropout2 = nn.Dropout(0.2)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.dropout3 = nn.Dropout(0.2)
        self.bn3 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.activation_func = activation_func(activation)
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = self.activation_func(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(self.activation_func(self.bn2(self.dropout2(out))))
        out = self.conv3(self.activation_func(self.bn3(self.dropout3(out))))
        out += shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=17):
        super(PreActResNet, self).__init__()
        self.in_planes = 8

        self.conv1 = nn.Conv1d(2, 8, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 8, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 16, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 32, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 64, num_blocks[3], stride=2)
        self.linear = nn.Linear(64 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool1d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def PreActResNet18():
    return PreActResNet(PreActBlock, [2, 2, 2, 2])


def PreActResNet34():
    return PreActResNet(PreActBlock, [3, 4, 6, 3])


def PreActResNet50():
    return PreActResNet(PreActBottleneck, [3, 4, 6, 3])


def PreActResNet101():
    return PreActResNet(PreActBottleneck, [3, 4, 23, 3])


def PreActResNet152():
    return PreActResNet(PreActBottleneck, [3, 8, 36, 3])


def test():
    net = PreActResNet18()
    y = net((torch.randn(1, 2, 32)))
    print(y.size())


if __name__ == "__main__":
    # input = torch.ones(1, 1, 4)
    # t_conv = nn.ConvTranspose1d(in_channels=1, out_channels=1, stride=1, kernel_size=3, padding=0, output_padding=0,
    #                             dilation=2, padding_mode="zeros", bias=False)
    # t_conv.weight.data = torch.tensor(data=[[[1, 2, 3]]], dtype=torch.float)
    # print(t_conv(input))

    # polygon = torch.rand(64).view(1, 2, 32)
    # ae = ConvAE3(8, 16, 32, True)
    # print(polygon)
    # pad_polygon = ae.circular_padding(polygon, 5)
    # print(pad_polygon)

    # im = Image.open("D:\\projects\\shape_dataset\\animal_dataset\\bird\\bird1.tif")
    # img = np.array(im)
    # polygon = im2poly(img, 32)
    # polygon = polygon.transpose()
    # polygon = polygon[np.newaxis, :]
    # print(polygon.shape)
    # polygon = torch.from_numpy(polygon)
    # ae = ConvAEEqualArcLength(8, 16, 32, True, 32)
    # result = ae.equal_arc_length(polygon, 32)
    # result = result.view(2, 32)
    # print(result.shape)
    #
    # x = result[0, :]
    # y = result[1, :]
    # plt.scatter(x.numpy(), y.numpy())
    # plt.show()

    test()
