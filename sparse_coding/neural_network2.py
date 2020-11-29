import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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


class VGG11SparseCoding(nn.Module):
    def __init__(self, layer_list):
        channel1 = layer_list[0]
        channel2 = layer_list[1]
        channel3 = layer_list[2]
        channel4 = layer_list[3]
        input1 = layer_list[4]
        input2 = layer_list[5]
        input3 = layer_list[6]
        super(VGG11SparseCoding, self).__init__()
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
        return x


