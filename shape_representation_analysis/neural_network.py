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


