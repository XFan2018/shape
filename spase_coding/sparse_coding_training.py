import argparse
from Animal_dataset2 import AnimalDataset
from sklearn.model_selection import train_test_split
import torch.nn as nn
from torchvision.datasets.folder import make_dataset, pil_loader
import torchvision
from sparse_coding2 import *
import matplotlib.pyplot as plt
from torchvision import transforms
from neural_network2 import Net, VGG11SparseCoding
import scipy.io as sio
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

parser = argparse.ArgumentParser(description="Train a classifier with polygon coordinates")
parser.add_argument("-dts", "--dataset", help="Path to training dataset.")
parser.add_argument("-nn", "--node_number", help="Input, hidden, and output node number")
parser.add_argument("-ext", "--extension", help="Image file extension")
parser.add_argument("-t", "--test", help="For test mode, it is True, otherwise it is False")
parser.add_argument("-pn", "--polygon_number", help="Number of coordinates in polygon")
parser.add_argument("-epn", "--epoch_number", help="Number of epochs")
parser.add_argument("-m", "--model", help="path to the model")
parser.add_argument("-ts", "--testset", help="path to the testset")
parser.add_argument("-ltrp", "--log_training_path", help="path to the log of training")
parser.add_argument("-ltsp", "--log_testing_path", help="path to the log of testing")
args = parser.parse_args()

with open("coefficients.mat", 'rb') as file:
    dictionary = sio.loadmat(file)
dataset = dictionary["coefficients"]
targets = dictionary["targets"]
X_train, X_val, y_train, y_val = train_test_split(dataset, targets, test_size=0.2,
                                                  stratify=targets)  # use stratify to keep equal proportions to each class
# print(np.unique(y_train, return_counts=True))
# print(np.unique(y_val, return_counts=True))
X_train = torch.tensor(X_train)
X_val = torch.tensor(X_val)
y_train = torch.tensor(y_train, dtype=torch.long)
y_val = torch.tensor(y_val, dtype=torch.long)
y_train.squeeze_(dim=1)
y_val.squeeze_(dim=1)
trainset = TensorDataset(X_train, y_train)
testset = TensorDataset(X_val, y_val)


def training(model, dataloader, criterion, optimizer, num_epochs, device, log_training_path, architecture=""):
    model_path = args.model + architecture

    # write to the log_finetune_silhouette every epoch
    if os.path.exists(log_training_path):
        epoch_log = open(log_training_path + "/train_epoch_loss.txt", "w+")
    else:
        os.mkdir(log_training_path)
        epoch_log = open(log_training_path + "/train_epoch_loss.txt", "w+")

    epoch_log.write("epoch".ljust(30) +
                    "epoch loss".ljust(30) +
                    "epoch correct top1".ljust(30) +
                    "epoch correct top5\n")
    epoch_log.close()

    # training
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        if os.path.exists(log_training_path):
            epoch_log = open(log_training_path + "/train_epoch_loss.txt", "a+")

        model.train()  # Set model to training mode

        epoch_loss = 0.0
        epoch_corrects_top1 = 0
        epoch_corrects_top5 = 0

        # Iterate over data.
        for data_index, (inputs, labels) in enumerate(dataloader):

            inputs = inputs.to(device)
            labels = labels.to(device)

            print("\n" + "data_index: " + str(data_index), "\n" + "-" * 10)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history
            with torch.set_grad_enabled(True):
                # Get model outputs and calculate loss
                outputs = model(inputs.float())
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                top5, top5_preds = torch.topk(outputs, 5, 1)

                print(  # f"\ninputs: {inputs.detach()}"
                    f"\nloss1: {loss.detach():.4f} ",
                    f"\nmax probability1: {_.detach()} "
                    f"\npreds1: ", preds.detach(),
                    f"\nlabels1: {labels.detach()}")

                loss.backward()

                optimizer.step()

            corrects_top1 = torch.sum(preds == labels.detach())
            corrects_top5 = 0

            for i in range(len(labels)):
                if labels.detach()[i] in top5_preds[i]:
                    corrects_top5 += 1

            # statistics of epoch loss
            epoch_loss += loss
            epoch_corrects_top1 += corrects_top1
            epoch_corrects_top5 += corrects_top5

            ########## to delete ###########
            # if data_index == 20:
            #    break

        epoch_result = (f'{epoch:<30} '
                        f'{epoch_loss / (len(dataloader.dataset)) :<30.6f} '
                        f'{float(epoch_corrects_top1) / (len(dataloader.dataset)):<30.6f} '
                        f'{float(epoch_corrects_top5) / (len(dataloader.dataset)):.6f}\n')

        epoch_log.write(epoch_result)
        epoch_log.close()

        # save the model
        if (epoch + 1) % 20 == 0:
            if os.path.exists(model_path):
                torch.save(model, model_path + "/model.pkl" + str(epoch))
            else:
                os.mkdir(model_path)
                torch.save(model, model_path + "/model.pkl" + str(epoch))


def testing(model, test_loader, device, model_id, log_testing_path):
    if os.path.exists(log_testing_path):
        f_log = open(log_testing_path + "/test_after_train.txt", "a+")
    else:
        os.mkdir(log_testing_path)
        f_log = open(log_testing_path + "/test_after_train.txt", "a+")

    # set evaluate mode
    model.eval()

    running_corrects = 0
    running_corrects_top5 = 0

    # test model
    for index, (inputs, labels) in enumerate(test_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        print("\nval", str(index), "\n" + "-" * 10)

        with torch.set_grad_enabled(False):
            outputs = model(inputs.float())
            _, preds = torch.max(outputs, 1)
            top5_, top5_preds = torch.topk(outputs, 5, 1)

            print(f"\ninputs: {inputs.detach()}"
                  f"\nmax outputs: {_.detach()}",
                  f"\npreds: ", preds.detach(),
                  f"\nlabels: {labels.detach()}")

        running_corrects += torch.sum(preds == labels.detach())
        for i in range(len(labels)):
            if labels.detach()[i] in top5_preds[i]:
                running_corrects_top5 += 1

        print(f"running corrects: {running_corrects}\t{running_corrects.double() / ((index + 1) * len(labels)):.4f}")
        print(f"running corrects top5: {running_corrects_top5}\t"
              f"{float(running_corrects_top5) / ((index + 1) * len(labels)):.4f}")
        # if index == 5:
        #    break

    total_acc = running_corrects.double() / len(test_loader.dataset)
    total_acc_top5 = float(running_corrects_top5) / len(test_loader.dataset)

    result = f"Model_id: {model_id:<30} {total_acc:<30}" + 'Acc: {:.4f} Acc_top5: {:.4f}\n'.format(
        total_acc,
        total_acc_top5)
    f_log.write(result)
    f_log.close()




def sparse_coding_training():
    layer_list = [int(i) for i in args.node_number.split()]
    dataloader = torch.utils.data.DataLoader(trainset,
                                             batch_size=16,
                                             shuffle=True)
    model = Net(layer_list)
    # model = VGG11SparseCoding(layer_list)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=1e-2,
                                momentum=0.9,
                                weight_decay=1e-4)
    if torch.cuda.is_available():
        model.cuda()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    training(model=model,
             dataloader=dataloader,
             criterion=criterion,
             optimizer=optimizer,
             num_epochs=int(args.epoch_number),
             device=device,
             log_training_path=args.log_training_path)


def sparse_coding_testing():
    test_loader = torch.utils.data.DataLoader(testset,
                                              batch_size=16,
                                              shuffle=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    epoch_number = int(args.epoch_number)
    for i in range(19, epoch_number, 20):
        model = torch.load(args.model + "/model.pkl" + str(i))
        testing(model=model,
                test_loader=test_loader,
                device=device,
                model_id=i,
                log_testing_path=args.log_testing_path)


if __name__ == "__main__":
    sparse_coding_training()
    sparse_coding_testing()
