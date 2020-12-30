import sys
import os

print(sys.path)
sys.path.append(os.path.split(sys.path[0])[0])
import argparse
from PIL import Image
from shape_representation_analysis.sparse_coding import im2poly
import matplotlib.pyplot as plt
from shape_representation_analysis.Animal_dataset import AnimalDataset
from shape_representation_analysis.Hemera_dataset import HemeraDataset
from shape_representation_analysis.PolygonAEHemeraDataset import PolygonAEHemeraDataset, TurningAngleAEHemeraDataset, \
    FourierDescriptorAEHemeraDataset
from shape_representation_analysis.PolygonAEAnimalDataset import PolygonAEAnimalDataset, TurningAngleAEAnimalDataset
from torchvision.datasets.folder import make_dataset, pil_loader
import torchvision
from sparse_coding import *
from shape_representation_analysis.shape_transforms import TurningAngleTransform, PolygonTransform, Angle2VecTransform, \
    RandomRotatePoints, \
    FourierDescriptorTransform, InterpolationTransform, InterpolationTransform2, EqualArclengthTransform, \
    RandomRotatePoints, RandomFlipPoints, RandomTranslation, IndexRotate
import matplotlib.pyplot as plt
from torchvision import transforms
from shape_representation_analysis.neural_network import TurningAngleNet, Net, VGG11TurningAngle, VGG16TurningAngle, \
    RNN, VGG11PolygonCoordinates, \
    VGG9PolygonCoordinates, VGG7PolygonCoordinates, VGG16PolygonCoordinates, LSTM, polygon_sets_transform, AE, AE2, \
    ConvAE4, ConvAE2, ConvAE3, ConvAEEqualArcLength, ConvAE1_1, ConvAE2_2, CNN2, VGG6PolygonCoordinates, \
    VGG4PolygonCoordinates, VGG6PolygonCoordinates_dropout
from shape_representation_analysis.pytorchtools import EarlyStopping
import numpy as np
import torch.nn as nn
import torch

parser = argparse.ArgumentParser(description="Train a classifier with polygon coordinates")
parser.add_argument("-dts", "--dataset", help="Path to training dataset.")
parser.add_argument("-vs", "--validset", help="Path to validation dataset.")
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

batch_size = 16
cuda = 0


# for data, label in dataset:
#     print(data.shape)

def set_bn_eval(module):
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.eval()


def training_no_es(model, dataloader, criterion, optimizer, num_epochs, device, log_training_path, architecture=""):
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


def train_autoencoder(model,
                      dataloader,
                      validloader,
                      criterion,
                      optimizer,
                      num_epochs,
                      device,
                      log_training_path,
                      architecture="",
                      patience=3):
    lambda1 = lambda epoch: 0.99 ** epoch if 0.99 ** epoch > 0.01 else 0.01
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    model_path = args.model + architecture
    stop_point = 0
    # training loss
    train_losses = []
    # validation loss
    valid_losses = []
    # average training loss per epoch
    avg_train_losses = []
    # average validation loss per epoch
    avg_valid_losses = []
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    # write to the log_finetune_silhouette every epoch
    if os.path.exists(log_training_path):
        epoch_log = open(log_training_path + "/train_epoch_loss.txt", "w+")
    else:
        os.mkdir(log_training_path)
        epoch_log = open(log_training_path + "/train_epoch_loss.txt", "w+")

    epoch_log.write("epoch".ljust(30) +
                    "epoch loss".ljust(30))
    epoch_log.close()

    # training
    for epoch in range(num_epochs):

        print('-' * 10)

        if os.path.exists(log_training_path):
            epoch_log = open(log_training_path + "/train_epoch_loss.txt", "a+")

        model.train()  # Set model to training mode

        epoch_loss = 0.0

        # Iterate over data.
        for data_index, inputs in enumerate(dataloader):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            for param_group in optimizer.param_groups:
                print("learning rate: ", param_group["lr"])
            inputs = inputs.to(device)

            print("\n" + "data_index: " + str(data_index), "\n" + "-" * 10)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            with torch.set_grad_enabled(True):
                # Get model outputs and calculate loss
                print(inputs.shape)
                outputs = model(inputs.float())
                loss = criterion(outputs, inputs.float())

                print(f"\nloss1: {loss.detach():.4f} ")

                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            epoch_loss += loss.item()
            ########## to delete ###########
            # if data_index == 2:
            #     break

        ######################
        # validate the model #
        ######################
        model.eval()  # prep model for evaluation
        for data_index, inputs in enumerate(validloader):
            inputs = inputs.to(device)
            # forward pass: compute predicted outputs by passing inputs to the model
            outputs = model(inputs.float())
            # calculate the loss
            loss = criterion(outputs, inputs.float())
            # append validation loss
            valid_losses.append(loss.item())
            ########## to delete ###########
            # if data_index == 2:
            #     break
        # print training/validation statistics
        # calculate average loss per an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        epoch_result = (f'{epoch:<30} '
                        f'{train_loss :<30.6f} \n')

        epoch_log.write(epoch_result)
        epoch_log.close()
        # clear lists to track next epoch
        train_losses = []
        valid_losses = []
        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            stop_point = epoch
            break
        scheduler.step()
    if stop_point == 0:
        stop_point = num_epochs
    model.load_state_dict(torch.load('checkpoint.pt'))
    if os.path.exists(model_path):
        torch.save(model, model_path + "/model.pkl" + str(stop_point - patience))
    else:
        os.mkdir(model_path)
        torch.save(model, model_path + "/model.pkl" + str(stop_point - patience))

    return model, avg_train_losses, avg_valid_losses, stop_point


def autoencoder_training_no_es(model, dataloader, criterion, optimizer, num_epochs, device, log_training_path,
                               architecture=""):
    model_path = args.model + architecture

    # write to the log_finetune_silhouette every epoch
    if os.path.exists(log_training_path):
        epoch_log = open(log_training_path + "/train_epoch_loss.txt", "w+")
    else:
        os.mkdir(log_training_path)
        epoch_log = open(log_training_path + "/train_epoch_loss.txt", "w+")

    epoch_log.write("epoch".ljust(30) +
                    "epoch loss".ljust(30))
    epoch_log.close()

    # training
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        if os.path.exists(log_training_path):
            epoch_log = open(log_training_path + "/train_epoch_loss.txt", "a+")

        model.train()  # Set model to training mode

        epoch_loss = 0.0

        # Iterate over data.
        for data_index, (inputs, labels) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            # inputs = inputs.view(-1, 1, 64).squeeze()
            print("\n" + "data_index: " + str(data_index), "\n" + "-" * 10)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history
            with torch.set_grad_enabled(True):
                # Get model outputs and calculate loss
                print(inputs.shape)
                outputs = model(inputs.float())
                loss = criterion(outputs, inputs.float())

                print(  # f"\ninputs: {inputs.detach()}"
                    f"\nloss1: {loss.detach():.4f} ")

                loss.backward()

                optimizer.step()

            # statistics of epoch loss
            epoch_loss += loss

            ########## to delete ###########
            # if data_index == 2:
            #     break
        img = Image.open(r"D:\projects\shape_dataset\animal_dataset\bird\bird13.tif")
        np_img = np.array(img)  # PIL image to numpy (row, col, channel)
        polygon_coordinates_img = im2poly(np_img, 32)
        # counter clockwise
        polygon_coordinates_img = np.flip(polygon_coordinates_img, axis=0)
        # index start from the left most point
        idx = np.argmin(polygon_coordinates_img[:, 0])
        polygon_coordinates_img = np.vstack((polygon_coordinates_img[idx:], polygon_coordinates_img[0:idx]))
        original_input = polygon_coordinates_img.reshape(-1, 32)[np.newaxis, ...]
        input = torch.tensor(original_input)
        input = input.to(device)
        print(input.shape)
        reconstruction = model(input.float())
        reconstruction = np.array(reconstruction.cpu().detach())

        # original_input = original_input.squeeze((0, 1))
        # reconstruction = reconstruction.squeeze((0, 1))
        original_input = original_input.squeeze(0)
        reconstruction = reconstruction.squeeze(0)

        np.savetxt("ConvAE2_training\\original_input" + str(epoch), original_input)
        np.savetxt("ConvAE2_training\\reconstruction" + str(epoch), reconstruction)

        epoch_result = (f'{epoch:<30} '
                        f'{epoch_loss / (len(dataloader.dataset)) :<30}\n')

        epoch_log.write(epoch_result)
        epoch_log.close()

    # save the model
    if os.path.exists(model_path):
        torch.save(model, model_path + "/model.pkl")
    else:
        os.mkdir(model_path)
        torch.save(model, model_path + "/model.pkl")


def testing_no_es(model, test_loader, device, model_id, log_testing_path):
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

            print(  # f"\ninputs: {inputs.detach()}"
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

    total_acc = running_corrects.double() / len(test_loader.dataset)
    total_acc_top5 = float(running_corrects_top5) / len(test_loader.dataset)

    result = f"Model_id: {model_id:<30} {total_acc:<30}" + 'Acc: {:.4f} Acc_top5: {:.4f}\n'.format(
        total_acc,
        total_acc_top5)
    f_log.write(result)
    f_log.close()


def training(model, dataloader, validloader, criterion, optimizer, num_epochs, device, log_training_path,
             architecture="", patience=50):
    lambda1 = lambda epoch: 0.99 ** epoch if 0.99 ** epoch > 0.01 else 0.01
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)
    model_path = args.model + architecture
    stop_point = 0
    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []
    early_stopping = EarlyStopping(patience=patience, verbose=True)
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
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            for param_group in optimizer.param_groups:
                print("learning rate: ", param_group["lr"])

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
                train_losses.append(loss.item())

            corrects_top1 = torch.sum(preds == labels.detach())
            corrects_top5 = 0

            for i in range(len(labels)):
                if labels.detach()[i] in top5_preds[i]:
                    corrects_top5 += 1

            # statistics of epoch loss
            epoch_loss += loss.item()
            epoch_corrects_top1 += corrects_top1
            epoch_corrects_top5 += corrects_top5

        ######################
        # validate the model #
        ######################
        model.eval()  # prep model for evaluation
        for inputs, labels in validloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(inputs.float())
            # calculate the loss
            loss = criterion(output, labels)
            # record validation loss
            valid_losses.append(loss.item())
        # print training/validation statistics
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        epoch_result = (f'{epoch:<30} '
                        f'{train_loss :<30.6f} '
                        f'{float(epoch_corrects_top1) / (len(dataloader.dataset)):<30.6f} '
                        f'{float(epoch_corrects_top5) / (len(dataloader.dataset)):.6f}\n')

        epoch_log.write(epoch_result)
        epoch_log.close()
        # clear lists to track next epoch
        train_losses = []
        valid_losses = []
        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            stop_point = epoch
            break
        scheduler.step()
    if stop_point == 0:
        stop_point = num_epochs
    model.load_state_dict(torch.load('checkpoint.pt'))
    if os.path.exists(model_path):
        torch.save(model, model_path + "/model.pkl" + str(stop_point - patience))
    else:
        os.mkdir(model_path)
        torch.save(model, model_path + "/model.pkl" + str(stop_point - patience))

    return model, avg_train_losses, avg_valid_losses, stop_point


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

            print(  # f"\ninputs: {inputs.detach()}"
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


def dfs_freeze(model):
    for name, child in model.named_children():
        if name not in ["fc1", "fc2", "fc3"]:
            print(name)
            for param in child.parameters():
                param.requires_grad = False
        dfs_freeze(child)


def polygon_training():
    # input_nodes, hidden1_nodes, hidden2_nodes, output_nodes = args.node_number.split()
    rrt = RandomRotatePoints(20)
    hft = RandomFlipPoints(0.5)
    # vft = RandomFlipPoints(0.05, True)
    irt = IndexRotate()
    rtt = RandomTranslation(0.01)

    transform = torchvision.transforms.Compose([hft, rrt, irt, rtt])
    # transform_valid = torchvision.transforms.Compose([PolygonTransform(int(args.polygon_number), False)])
    # dataset = AnimalDataset(args.dataset, args.extension, transforms=transform_train)
    dataset = PolygonAEAnimalDataset(
        r"polygon_animal_dataset.csv",
        r"polygon_animal_dataset_label.csv",
        transform,
        twoDim=True)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=True)
    validset = PolygonAEAnimalDataset(
        r"polygon_animal_dataset_validation.csv",
        r"polygon_animal_dataset_label_validation.csv",
        twoDim=True)
    validloader = torch.utils.data.DataLoader(validset,
                                              batch_size=batch_size,
                                              shuffle=False)
    # model = Net([int(input_nodes), int(hidden1_nodes), int(hidden2_nodes), int(output_nodes)])
    # model = VGG4PolygonCoordinates(8, 16, 128, 64)
    model = VGG6PolygonCoordinates_dropout(8, 16, 32, 128, 128, 64)
    # model = torch.load(
    #     r"D:\projects\shape\shape_representation_analysis\log_model_ConvAE1_1_es_8_bs=64\pretrained_CNN2.pkl")
    device = torch.device("cuda:" + str(cuda) if torch.cuda.is_available() else "cpu")
    # model = torch.load(
    #    r"D:\projects\shape\shape_representation_analysis\log_model_ConvAE4_es_8_16_32_turning_angle\no_pretrain_conv_ae_turning_angle")
    # model.apply(dfs_freeze)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=1e-2,
                                 betas=(0.9, 0.999),
                                 eps=1e-8,
                                 weight_decay=1e-4)

    if torch.cuda.is_available():
        model.cuda(torch.device('cuda:' + str(cuda)))

    # old_state_dict = {}
    # for key in model.state_dict():
    #     old_state_dict[key] = model.state_dict()[key].clone()

    model, train_loss, valid_loss, stop_point = training(model=model,
                                                         dataloader=dataloader,
                                                         validloader=validloader,
                                                         criterion=criterion,
                                                         optimizer=optimizer,
                                                         num_epochs=int(args.epoch_number),
                                                         device=device,
                                                         log_training_path=args.log_training_path,
                                                         patience=5000)

    # new_state_dict = {}
    # for key in model.state_dict():
    #     new_state_dict[key] = model.state_dict()[key].clone()
    #
    # # Compare params
    # count = 0
    # for key in old_state_dict:
    #     if not (old_state_dict[key] == new_state_dict[key]).all():
    #         print('Diff in {}'.format(key))
    #         count += 1
    # print(count)

    return model, train_loss, valid_loss, stop_point


def polygon_testing(model_trained, stop_point):
    # transform = torchvision.transforms.Compose([PolygonTransform(int(args.polygon_number), False)])
    validset = PolygonAEAnimalDataset(
        r"polygon_animal_dataset_validation.csv",
        r"polygon_animal_dataset_label_validation.csv",
        twoDim=True)
    valid_loader = torch.utils.data.DataLoader(validset,
                                               batch_size=batch_size,
                                               shuffle=False)
    device = torch.device("cuda:" + str(cuda) if torch.cuda.is_available() else "cpu")
    model = model_trained
    testing(model=model,
            test_loader=valid_loader,
            device=device,
            model_id=stop_point,
            log_testing_path=args.log_testing_path)


# Fourier transform
def fd_training():
    input_nodes, hidden1_nodes, output_nodes = args.node_number.split()
    transform_train = torchvision.transforms.Compose([FourierDescriptorTransform(int(args.polygon_number))])
    transform_valid = torchvision.transforms.Compose([FourierDescriptorTransform(int(args.polygon_number))])
    dataset = AnimalDataset(args.dataset, args.extension, transforms=transform_train)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=16,
                                             shuffle=True)
    validset = AnimalDataset(args.validset, args.extension, transforms=transform_valid)
    validloader = torch.utils.data.DataLoader(validset,
                                              batch_size=16,
                                              shuffle=False)
    model = Net([int(input_nodes), int(hidden1_nodes), int(output_nodes)])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=1e-2,
                                momentum=0.9,
                                weight_decay=1e-4)
    if torch.cuda.is_available():
        model.cuda()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, train_loss, valid_loss, stop_point = training(model=model,
                                                         dataloader=dataloader,
                                                         validloader=validloader,
                                                         criterion=criterion,
                                                         optimizer=optimizer,
                                                         num_epochs=int(args.epoch_number),
                                                         device=device,
                                                         log_training_path=args.log_training_path,
                                                         patience=50)
    return model, train_loss, valid_loss, stop_point


def fd_testing(model_trained, stop_point):
    transform = torchvision.transforms.Compose([FourierDescriptorTransform(int(args.polygon_number))])
    validset = AnimalDataset(args.validset, args.extension, transforms=transform)
    valid_loader = torch.utils.data.DataLoader(validset,
                                               batch_size=16,
                                               shuffle=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model_trained
    testing(model=model,
            test_loader=valid_loader,
            device=device,
            model_id=stop_point,
            log_testing_path=args.log_testing_path)


def polygon_training_no_es():
    # input_nodes, hidden1_nodes, hidden2_nodes, output_nodes = args.node_number.split()
    transform = torchvision.transforms.Compose([transforms.RandomHorizontalFlip(0.5),
                                                transforms.RandomVerticalFlip(0.05),
                                                PolygonTransform(int(args.polygon_number)),
                                                RandomRotatePoints(20)
                                                ])
    dataset = AnimalDataset(args.dataset, args.extension, transforms=transform)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=16,
                                             shuffle=True)
    # model = Net([int(input_nodes), int(hidden1_nodes), int(hidden2_nodes), int(output_nodes)])
    model = VGG7PolygonCoordinates(8, 16, 32, 128, 128, 64)
    # model = VGG16PolygonCoordinates(32, 64, 128, 256, 256, 64, 17)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=1e-2,
                                momentum=0.9,
                                weight_decay=1e-4)
    if torch.cuda.is_available():
        model.cuda()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    training_no_es(model=model,
                   dataloader=dataloader,
                   criterion=criterion,
                   optimizer=optimizer,
                   num_epochs=int(args.epoch_number),
                   device=device,
                   log_training_path=args.log_training_path)


def polygon_testing_no_es():
    transform = torchvision.transforms.Compose([PolygonTransform(int(args.polygon_number))])
    testset = AnimalDataset(args.testset, args.extension, transforms=transform)
    test_loader = torch.utils.data.DataLoader(testset,
                                              batch_size=16,
                                              shuffle=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    epoch_number = int(args.epoch_number)
    for i in range(19, epoch_number, 20):
        model = torch.load(args.model + "/model.pkl" + str(i))
        testing_no_es(model=model,
                      test_loader=test_loader,
                      device=device,
                      model_id=i,
                      log_testing_path=args.log_testing_path)


def rnn_training(model, dataloader, criterion, optimizer, num_epochs, device, log_training_path, architecture=""):
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
                inputs = inputs.view(-1, int(args.polygon_number), 360)
                hidden = model.initHidden(inputs.size()[0])
                print("hidden", hidden.shape)
                for i in range(inputs.size()[1]):
                    input = torch.squeeze(inputs[:, i, :])
                    outputs, hidden = model(input.float(), hidden)
                # outputs = model(inputs.float())
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


def rnn_testing(model, test_loader, device, model_id, log_testing_path):
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
            inputs = inputs.view(-1, int(args.polygon_number), 360)
            hidden = model.initHidden(inputs.size()[0])
            print("hidden", hidden.shape)
            for i in range(inputs.size()[1]):
                input = torch.squeeze(inputs[:, i, :])
                outputs, hidden = model(input.float(), hidden)
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


# set_size is number of polygon coordinates used as input
def lstm_training(model, dataloader, criterion, optimizer, num_epochs, device, log_training_path, set_size=None,
                  architecture=""):
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
                inputs = torch.transpose(inputs, 2, 1)
                inputs = torch.transpose(inputs, 1, 0)
                if set_size is not None:
                    inputs = polygon_sets_transform(inputs, set_size)
                batch = inputs.shape[1]
                print(inputs.shape)
                print("batch", batch)
                model.hidden = model.initHidden(batch)
                outputs = model(inputs.float(), batch)
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
            #     break

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


def lstm_testing(model, test_loader, device, model_id, log_testing_path, set_size=None):
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
            inputs = torch.transpose(inputs, 2, 1)
            inputs = torch.transpose(inputs, 1, 0)
            if set_size is not None:
                inputs = polygon_sets_transform(inputs, set_size)
            batch = inputs.shape[1]
            model.hidden = model.initHidden(batch)
            outputs = model(inputs.float(), batch)
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
        #     break

    total_acc = running_corrects.double() / len(test_loader.dataset)
    total_acc_top5 = float(running_corrects_top5) / len(test_loader.dataset)

    result = f"Model_id: {model_id:<30} {total_acc:<30}" + 'Acc: {:.4f} Acc_top5: {:.4f}\n'.format(
        total_acc,
        total_acc_top5)
    f_log.write(result)
    f_log.close()


def turning_angle_training():
    # input_nodes, hidden1_nodes, hidden2_nodes, output_nodes, kernel_size = args.node_number.split()
    # print(input_nodes, hidden1_nodes, hidden2_nodes, output_nodes, kernel_size)
    transform = torchvision.transforms.Compose([TurningAngleTransform(int(args.polygon_number))])
    dataset = AnimalDataset(args.dataset, args.extension, transforms=transform)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=16,
                                             shuffle=True)
    # model = TurningAngleNet(int(input_nodes), int(hidden1_nodes), int(hidden2_nodes), int(output_nodes), int(kernel_size))
    # model = VGG11TurningAngle(32, 64, 128, 256, 1028, 128, 64)
    model = VGG16TurningAngle(32, 64, 128, 256, 1024, 512, 128)
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


def turning_angle_testing():
    transform = torchvision.transforms.Compose([TurningAngleTransform(int(args.polygon_number))])
    testset = AnimalDataset(args.testset, args.extension, transforms=transform)
    test_loader = torch.utils.data.DataLoader(testset,
                                              batch_size=16,
                                              shuffle=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    for i in range(19, int(args.epoch_number), 20):
        model = torch.load(args.model + "/model.pkl" + str(i))
        testing(model=model,
                test_loader=test_loader,
                device=device,
                model_id=i,
                log_testing_path=args.log_testing_path)


def train_multi_polygon():
    """
    train and test models with different architecture
    """
    input_nodes = 64
    output_nodes = 17
    architectures = []  # a list of strings containing the architecture e.g. "62_32_17"
    transform = torchvision.transforms.Compose([PolygonTransform(int(args.polygon_number))])
    dataset = AnimalDataset(args.dataset, args.extension, transforms=transform)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=16,
                                             shuffle=True)
    criterion = nn.CrossEntropyLoss()
    # 1 hidden layer
    for hidden_nodes in range(32, 104, 8):
        layer_list = [input_nodes, hidden_nodes, output_nodes]
        architectures.append("_".join([str(i) for i in layer_list]))
        model = Net(layer_list)
        optimizer = torch.optim.SGD(model.parameters(),
                                    lr=1e-2,
                                    momentum=0.9,
                                    weight_decay=1e-4)
        if torch.cuda.is_available():
            print("################model is available####################")
            model.cuda()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        training(model=model,
                 dataloader=dataloader,
                 criterion=criterion,
                 optimizer=optimizer,
                 num_epochs=int(args.epoch_number),
                 device=device,
                 log_training_path="polygon_training_" + str(input_nodes) +
                                   "_" + str(hidden_nodes) +
                                   "_" + str(output_nodes),
                 architecture="_" + str(input_nodes) +
                              "_" + str(hidden_nodes) +
                              "_" + str(output_nodes))

    # 2 hidden layers
    for hidden1_nodes in range(32, 104, 8):
        for hidden2_nodes in range(32, 72, 8):
            layer_list = [input_nodes, hidden1_nodes, hidden2_nodes, output_nodes]
            architectures.append("_".join([str(i) for i in layer_list]))
            model = Net(layer_list)
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
                     log_training_path="polygon_training_" + str(input_nodes) +
                                       "_" + str(hidden1_nodes) +
                                       "_" + str(hidden2_nodes) +
                                       "_" + str(output_nodes),
                     architecture="_" + str(input_nodes) +
                                  "_" + str(hidden1_nodes) +
                                  "_" + str(hidden2_nodes) +
                                  "_" + str(output_nodes))

    # 3 hidden layers
    for hidden1_nodes in range(32, 104, 8):
        for hidden2_nodes in range(32, 104, 8):
            for hidden3_nodes in range(32, 104, 8):
                layer_list = [input_nodes, hidden1_nodes, hidden2_nodes, hidden3_nodes, output_nodes]
                architectures.append("_".join([str(i) for i in layer_list]))
                model = Net(layer_list)
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
                         log_training_path="polygon_training_" + str(input_nodes) +
                                           "_" + str(hidden1_nodes) +
                                           "_" + str(hidden2_nodes) +
                                           "_" + str(hidden3_nodes) +
                                           "_" + str(output_nodes),
                         architecture="_" + str(input_nodes) +
                                      "_" + str(hidden1_nodes) +
                                      "_" + str(hidden2_nodes) +
                                      "_" + str(hidden3_nodes) +
                                      "_" + str(output_nodes))
    return architectures


def test_multi_polygon(architectures):
    transform = torchvision.transforms.Compose([PolygonTransform(int(args.polygon_number))])
    testset = AnimalDataset(args.testset, args.extension, transforms=transform)
    test_loader = torch.utils.data.DataLoader(testset,
                                              batch_size=16,
                                              shuffle=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    epoch_number = int(args.epoch_number)
    for architecture in architectures:
        for i in range(19, epoch_number, 20):
            model = torch.load(args.model + "_" + architecture + "/model.pkl" + str(i))
            if torch.cuda.is_available():
                model.cuda()
            testing(model=model,
                    test_loader=test_loader,
                    device=device,
                    model_id=i,
                    log_testing_path="polygon_testing_" + architecture)


def rnn_training_turning_angles():
    input_size, hidden_size, output_size = args.node_number.split()
    transform = torchvision.transforms.Compose([TurningAngleTransform(int(args.polygon_number)),
                                                Angle2VecTransform()])
    dataset = AnimalDataset(args.dataset, args.extension, transforms=transform)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=True)
    model = RNN(int(input_size), int(hidden_size), int(output_size), batch_size)
    model = model.float()
    criterion = nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=5e-3)
    if torch.cuda.is_available():
        model.cuda()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    rnn_training(model=model,
                 dataloader=dataloader,
                 criterion=criterion,
                 optimizer=optimizer,
                 num_epochs=int(args.epoch_number),
                 device=device,
                 log_training_path=args.log_training_path)


def rnn_testing_turning_angles():
    transform = torchvision.transforms.Compose([TurningAngleTransform(int(args.polygon_number)),
                                                Angle2VecTransform()])
    testset = AnimalDataset(args.testset, args.extension, transforms=transform)
    test_loader = torch.utils.data.DataLoader(testset,
                                              batch_size=batch_size,
                                              shuffle=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    epoch_number = int(args.epoch_number)
    for i in range(19, epoch_number, 20):
        model = torch.load(args.model + "/model.pkl" + str(i))
        rnn_testing(model=model,
                    test_loader=test_loader,
                    device=device,
                    model_id=i,
                    log_testing_path=args.log_testing_path)


def lstm_polygon_coordinates_training():
    num_layers, hidden_size, output_size = args.node_number.split()
    transform = torchvision.transforms.Compose([transforms.RandomHorizontalFlip(0.5),
                                                transforms.RandomVerticalFlip(0.05),
                                                PolygonTransform(int(args.polygon_number)),
                                                RandomRotatePoints(20)
                                                ])
    dataset = AnimalDataset(args.dataset, args.extension, transforms=transform)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=True)
    # input_size = 2: two coordinates
    # model = LSTM(2, int(hidden_size), int(num_layers), int(output_size), batch_size)
    model = torch.load(
        "D:\\projects\\summerProject2020\\project3\\model_polygon_lstm_1_32_aug_20degree(lr1-3)\\model.pkl1999")
    model = model.float()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=1e-3)
    if torch.cuda.is_available():
        model.cuda()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    lstm_training(model=model,
                  dataloader=dataloader,
                  criterion=criterion,
                  optimizer=optimizer,
                  num_epochs=int(args.epoch_number),
                  device=device,
                  log_training_path=args.log_training_path)


def lstm_polygon_coordinates_testing():
    transform = torchvision.transforms.Compose([PolygonTransform(int(args.polygon_number))])
    testset = AnimalDataset(args.testset, args.extension, transforms=transform)
    test_loader = torch.utils.data.DataLoader(testset,
                                              batch_size=batch_size,
                                              shuffle=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    epoch_number = int(args.epoch_number)
    for i in range(19, epoch_number, 20):
        model = torch.load(args.model + "/model.pkl" + str(i))
        lstm_testing(model=model,
                     test_loader=test_loader,
                     device=device,
                     model_id=i,
                     log_testing_path=args.log_testing_path)


def lstm_polygon_coordinates_training_polygon_set():
    num_layers, hidden_size, output_size = args.node_number.split()
    transform = torchvision.transforms.Compose([PolygonTransform(int(args.polygon_number))])
    dataset = AnimalDataset(args.dataset, args.extension, transforms=transform)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=True)
    # input_size = 8: two coordinates * 4(set_size)
    model = LSTM(8, int(hidden_size), int(num_layers), int(output_size), batch_size)
    # model = torch.load(args.model + "/model.pkl1499")
    model = model.float()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=5e-3)
    if torch.cuda.is_available():
        model.cuda()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    lstm_training(model=model,
                  dataloader=dataloader,
                  criterion=criterion,
                  optimizer=optimizer,
                  num_epochs=int(args.epoch_number),
                  device=device,
                  log_training_path=args.log_training_path,
                  set_size=4)


def lstm_polygon_coordinates_testing_polygon_set():
    transform = torchvision.transforms.Compose([PolygonTransform(int(args.polygon_number))])
    testset = AnimalDataset(args.testset, args.extension, transforms=transform)
    test_loader = torch.utils.data.DataLoader(testset,
                                              batch_size=batch_size,
                                              shuffle=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    epoch_number = int(args.epoch_number)
    for i in range(19, epoch_number, 20):
        model = torch.load(args.model + "/model.pkl" + str(i))
        lstm_testing(model=model,
                     test_loader=test_loader,
                     device=device,
                     model_id=i,
                     log_testing_path=args.log_testing_path,
                     set_size=4)


def autoencoder_training():
    input_size, hidden_size1, hidden_size2, output_size = args.node_number.split()
    # transform = torchvision.transforms.Compose([PolygonTransform(int(args.polygon_number), oneDim=True)])
    training_set = FourierDescriptorAEHemeraDataset(args.dataset)
    valid_set = FourierDescriptorAEHemeraDataset(args.validset)

    dataloader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False)

    model = AE2(int(input_size), int(hidden_size1), int(hidden_size2), int(output_size))
    criterion = nn.MSELoss()
    device = torch.device("cuda:" + str(cuda) if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=1e-2,
                                 betas=(0.9, 0.999),
                                 eps=1e-8,
                                 weight_decay=1e-4)
    if torch.cuda.is_available():
        model.to(torch.device('cuda:' + str(cuda)))

    model, avg_train_losses, avg_valid_losses, stop_point = train_autoencoder(model=model,
                                                                              dataloader=dataloader,
                                                                              validloader=validloader,
                                                                              criterion=criterion,
                                                                              optimizer=optimizer,
                                                                              num_epochs=int(args.epoch_number),
                                                                              device=device,
                                                                              log_training_path=args.log_training_path,
                                                                              patience=100)
    return avg_train_losses, avg_valid_losses, stop_point


def conv_autoencoder_training():
    channel0, channel1, channel2, channel3 = args.node_number.split()
    # transform = torchvision.transforms.Compose(
    #     [PolygonTransform(int(args.polygon_number)), EqualArclengthTransform(int(args.polygon_number))])
    training_set = PolygonAEHemeraDataset(args.dataset, int(args.polygon_number), twoDim=True)
    valid_set = PolygonAEHemeraDataset(args.validset, int(args.polygon_number), twoDim=True)
    dataloader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_set, batch_size=batch_size, shuffle=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ConvAE2_2(int(channel0), int(channel1), int(channel2), int(channel3), circular=True)
    criterion = nn.MSELoss()
    # should put model in cuda before construct optimizer for it
    if torch.cuda.is_available():
        model.cuda()
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=1e-2,
                                 betas=(0.9, 0.999),
                                 eps=1e-8,
                                 weight_decay=1e-4)

    model, avg_train_losses, avg_valid_losses, stop_point = train_autoencoder(model=model,
                                                                              dataloader=dataloader,
                                                                              validloader=validloader,
                                                                              criterion=criterion,
                                                                              optimizer=optimizer,
                                                                              num_epochs=int(args.epoch_number),
                                                                              device=device,
                                                                              log_training_path=args.log_training_path,
                                                                              patience=100)
    return avg_train_losses, avg_valid_losses, stop_point


def plot(train_loss, valid_loss, stop_point):
    # visualize the loss as the network trained
    fig = plt.figure(figsize=(10, 8))
    plt.plot(range(1, len(train_loss) + 1), train_loss, label='Training Loss')
    plt.plot(range(1, len(valid_loss) + 1), valid_loss, label='Validation Loss')

    # find position of lowest validation loss
    minposs = valid_loss.index(min(valid_loss)) + 1
    plt.axvline(minposs, linestyle='--', color='r', label='Early Stopping Checkpoint')

    plt.xlabel('epochs')
    plt.ylabel('loss')
    # plt.ylim(0, 0.5)  # consistent scale
    plt.xlim(0, len(train_loss) + 1)  # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    fig.savefig(args.log_training_path + "_" + str(stop_point) + '_loss_plot.png', bbox_inches='tight')


def evaluate_ae_result():
    img = Image.open(r"D:\projects\shape_dataset\animal_dataset\spider\spider1.tif")
    img.show()
    sample = np.array(img)  # PIL image to numpy (row, col, channel)
    result = im2poly(sample, 32)
    # counter clockwise
    result = np.flip(result, axis=0)
    # index start from the left most point
    idx = np.argmin(result[:, 0])
    new_polygon = np.vstack((result[idx:], result[0:idx]))
    plt.scatter(new_polygon[:, 0], new_polygon[:, 1])
    plt.show()
    print(new_polygon)
    model = torch.load(r"D:\projects\shape\shape_representation_analysis\log_model_AE_es_64_64_32\model.pkl17")
    input = new_polygon.reshape(-1, 64)[np.newaxis, ...]
    input = torch.tensor(input)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input = input.to(device)
    result = model(input.float())
    result = np.array(result.cpu().detach())
    result = result.squeeze(0).reshape(32, 2)
    print(result)
    plt.scatter(result[:, 0], result[:, 1])
    plt.show()


def evaluate_fourier_descriptor_ae_result(model_dir, polygon_num):
    img = Image.open(r"D:\projects\shape_dataset\animal_dataset\spider\spider1.tif")
    np_img = np.array(img)  # PIL image to numpy (row, col, channel)
    polygon_coordinates_img = im2poly(np_img, polygon_num)

    # counter clockwise
    polygon_coordinates_img = np.flip(polygon_coordinates_img, axis=0)

    # index start from the left most point
    idx = np.argmin(polygon_coordinates_img[:, 0])
    polygon_coordinates_img = np.vstack((polygon_coordinates_img[idx:], polygon_coordinates_img[0:idx]))
    original_input = polygon_coordinates_img.transpose()
    print("original input", original_input.shape)

    # generate Fourier descriptor
    ft = FourierDescriptorTransform(polygon_num)
    original_fd = ft(img)
    print(original_fd)

    # reconstruct by autoencoder
    model = torch.load(model_dir)
    input = original_fd.reshape(-1, polygon_num * 2)[np.newaxis, ...]
    input = torch.tensor(input)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input = input.to(device)
    reconstruct_result = model(input.float())
    reconstruct_result = np.array(reconstruct_result.cpu().detach())
    reconstruct_result = reconstruct_result.squeeze(0).reshape(polygon_num * 2)

    # reconstruct output Fourier descriptor
    output_fd_complex = np.zeros(polygon_num, dtype=complex)
    output_fd_complex.real = reconstruct_result[0:polygon_num]
    output_fd_complex.imag = reconstruct_result[polygon_num:]
    fd_reconstruct = np.fft.ifft(output_fd_complex)
    fd_reconstruct = np.array([fd_reconstruct.real, fd_reconstruct.imag])

    # reconstruct input Fourier descriptor
    input_fd_complex = np.zeros(polygon_num, dtype=complex)
    input_fd_complex.real = original_fd[0:polygon_num]
    input_fd_complex.imag = original_fd[polygon_num:]
    input_fd_reconstruct = np.fft.ifft(input_fd_complex)
    input_fd_reconstruct = np.array([input_fd_reconstruct.real, input_fd_reconstruct.imag])

    plt.scatter(range(polygon_num * 2), reconstruct_result[:], c="b")
    plt.plot(range(polygon_num * 2), reconstruct_result[:], c="b", label="output Fourier descriptor")
    plt.plot(range(polygon_num * 2), original_fd[:], c="r", label="input Fourier descriptor")
    plt.legend()
    plt.show()

    plt.plot(fd_reconstruct[0, :], fd_reconstruct[1, :], c="g", label="autoencoder output fd reconstrution")
    plt.plot(original_input[0, :], original_input[1, :], c="r", label="polygon coordinates")
    plt.legend()
    plt.show()


def evaluate_conv_ae_result(conv, model_dir):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_dir, map_location=device)
    print(model.state_dict())
    img = Image.open(r"D:\projects\shape_dataset\animal_dataset\bird\bird1.tif")

    np_img = np.array(img)  # PIL image to numpy (row, col, channel)
    polygon_coordinates_img = im2poly(np_img, 32)

    # counter clockwise
    polygon_coordinates_img = np.flip(polygon_coordinates_img, axis=0)

    # index start from the left most point
    idx = np.argmin(polygon_coordinates_img[:, 0])
    polygon_coordinates_img = np.vstack((polygon_coordinates_img[idx:], polygon_coordinates_img[0:idx]))
    plt.plot(polygon_coordinates_img[:, 0], polygon_coordinates_img[:, 1])
    original_input = polygon_coordinates_img.transpose()[np.newaxis, ...]
    print(original_input)
    if not conv:
        input = original_input.squeeze(0).flatten()
    else:
        input = original_input

    print(input)
    input = torch.tensor(input)
    input = input.to(device)

    reconstruction = model(input.float())
    reconstruction = np.array(reconstruction.cpu().detach())
    print(reconstruction.shape)

    original_input = original_input.squeeze(0).transpose()
    if conv:
        reconstruction = reconstruction.squeeze(0).transpose()
    else:
        reconstruction = reconstruction.reshape(2, 32).transpose()
    print(reconstruction.shape)

    plt.scatter(original_input[:, 0], original_input[:, 1])
    plt.plot(reconstruction[:, 0], reconstruction[:, 1])
    plt.scatter(reconstruction[:, 0], reconstruction[:, 1])
    plt.show()


def save_pretrained_model(autoencoder_dir, model_save_path):
    # autoencoder_dir = r"D:\projects\summerProject2020\project3\model_polygon_ae_64_64_32_epoch20\model.pkl"
    # model_save_path = r"D:\projects\summerProject2020\project3\pre_trained_models\pretrained_model_64_64_32"

    model = torch.load(autoencoder_dir)
    model1 = Net([64, 64, 32, 17])

    model1.fc1.weight.data = model.state_dict()['encoder_hidden_layer.weight']
    model1.fc1.bias.data = model.state_dict()['encoder_hidden_layer.bias']
    model1.fc2.weight.data = model.state_dict()['encoder_output_layer.weight']
    model1.fc2.bias.data = model.state_dict()['encoder_output_layer.bias']

    print(model1.fc2.weight.data)
    print(model1.state_dict()['fc2.weight'])

    torch.save(model1, model_save_path)


def save_pretrained_conv_ae(autoencoder_dir, model_save_path, no_pretrain=False):
    # autoencoder_dir = r"D:\projects\shape\shape_representation_analysis\log_model_Conv_AE_8_16_32\model.pkl"
    # model_save_path = r'D:\projects\shape\shape_representation_analysis\log_model_Conv_AE_8_16_32\pretrained_conv_ae'

    if no_pretrain:
        model = ConvAE1_1(2, 8, True)
    else:
        model = torch.load(autoencoder_dir)
    model1 = CNN2(2, 8, 128, 64, 17)
    model1.conv1d_1.weight.data = model.state_dict()['conv1d_1.weight']
    model1.conv1d_1.bias.data = model.state_dict()['conv1d_1.bias']
    # model1.conv1d_2.weight.data = model.state_dict()['conv1d_2.weight']
    # model1.conv1d_2.bias.data = model.state_dict()['conv1d_2.bias']
    # model1.conv1d_3.weight.data = model.state_dict()['conv1d_3.weight']
    # model1.conv1d_3.bias.data = model.state_dict()['conv1d_3.bias']
    # model1.conv1d_4.weight.data = model.state_dict()['conv1d_4.weight']
    # model1.conv1d_4.bias.data = model.state_dict()['conv1d_4.bias']

    print(model1.conv1d_1.weight.data)
    print(model.state_dict()['conv1d_1.weight'])

    torch.save(model1, model_save_path)


if __name__ == "__main__":
    ################# train/test classifier #####################
    model, train_loss, valid_loss, stop_point = polygon_training()
    plot(train_loss, valid_loss, stop_point)
    polygon_testing(model, stop_point=stop_point)


    ################# evaluate convolutional auto-encoder #####################
    # evaluate_conv_ae_result(True, r"D:\projects\shape\shape_representation_analysis\log_model_AE_es_256_256_192_128_Fourier_descriptor_128_bs=64")

    ################# evaluate Fourier descriptor auto-encoder #####################
    # evaluate_fourier_descriptor_ae_result(
    #     r"D:\projects\shape\shape_representation_analysis\log_model_AE_es_256_256_192_128_Fourier_descriptor_128_bs=64\model.pkl916",
    #     128)

    ################# train fully connected autoencoder #####################
    # avg_train_losses, avg_valid_losses, stop_point = autoencoder_training()
    # plot(avg_train_losses, avg_valid_losses, stop_point)

    ################# train convolutional autoencoder #####################
    # avg_train_losses, avg_valid_losses, stop_point = conv_autoencoder_training()
    # plot(avg_train_losses, avg_valid_losses, stop_point)

    ################# save auto-encoder pretrained auto-encoder #####################
    # save_pretrained_conv_ae(r"D:\projects\shape\shape_representation_analysis\log_model_ConvAE1_1_es_8_bs=64\model.pkl52",
    #                         r'D:\projects\shape\shape_representation_analysis\log_model_ConvAE1_1_es_8_bs=64\pretrained_CNN2.pkl')
