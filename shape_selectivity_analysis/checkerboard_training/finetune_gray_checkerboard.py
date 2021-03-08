import os
import random
import sys
import time

import numpy as np
import torch

sys.path.append(os.path.split(os.getcwd())[0])
from shape_selectivity_analysis.checkerboard_training.scramble_checkerboard import checkerboard_scrambled_gray_batch, checkerboard_intact_gray_batch
from shape_selectivity_analysis.tools.pytorchtools import EarlyStopping


random.seed(os.getenv("SEED"))
"""
fine-tune on scrambled checker board
two dataloaders are used, one is for ground truth images, and the other is for adversarial attack images
"""

np.set_printoptions(precision=2)


def set_bn_eval(module):
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module.eval()


def train_model_gray(model, trainloaders1, validloaders1, criterion, optimizer, num_epochs, device, batch_size, model_path,
                log_path, block_size, fc_only, patience, intact):
    since = time.time()
    # block_size == 0 -> user random block sizes
    block_sizes = [7, 14, 28, 56]
    random_block_size = False
    if block_size == 0:
        random_block_size = True

    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True)

    log_frequency = 16  # batch size is 256, write to log1 every 10 batches

    batch_num = len(trainloaders1.dataset) // batch_size
    print("batch number: ", str(batch_num) + "\n")

    # running log1, write to the log1 every "log_frequency" number of batches
    if os.path.exists(log_path):
        running_log = open(log_path + "/finetune_running_loss.txt", "w+")
    else:
        os.mkdir(log_path)
        running_log = open(log_path + "/finetune_running_loss.txt", "w+")

    running_log.write("index".ljust(30) +
                      "every {} batches loss".format(log_frequency).ljust(30) +
                      "corrects top1".ljust(30) +
                      "corrects top5".ljust(30) +
                      "attack corrects top1".ljust(30) +
                      "attack corrects top5\n")

    # epoch log1, write to the log1 every epoch
    epoch_log = open(log_path + "/finetune_epoch_loss.txt", "w+")
    epoch_log.write("epoch".ljust(30) +
                    "epoch loss".ljust(30) +
                    "epoch correct top1".ljust(30) +
                    "epoch correct top5".ljust(30) +
                    "epoch attack correct top1".ljust(30) +
                    "epoch attack correct top5\n")

    epoch_log.close()
    running_log.close()

    print("number of data in {} batches".format(log_frequency),
          batch_size * log_frequency)

    stop_point = 0  # the number of epoch

    # training
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        if os.path.exists(log_path):
            epoch_log = open(log_path + "/finetune_epoch_loss.txt", "a+")

        model.train()  # Set model to training mode

        if fc_only:
            model.apply(set_bn_eval)

        running_loss = 0.0
        running_corrects_top1 = 0
        running_corrects_top5 = 0

        epoch_loss = 0.0
        epoch_corrects_top1 = 0
        epoch_corrects_top5 = 0

        confidence_coherent = 0

        ###################
        # train the model #
        ###################
        for data_index, (inputs1, labels1) in enumerate(trainloaders1):
            #############try to push data to device earlier#################
            labels1 = labels1.to(device)
            running_log = open(log_path + "/finetune_running_loss.txt", "a+")
            # inputs1 = np.array(inputs1)
            # inputs2 = np.array(inputs2)
            print(type(inputs1[0]))
            if random_block_size:
                block_size = random.choice(block_sizes)
            if intact:
                inputs = checkerboard_intact_gray_batch(inputs1, block_size)
            else:
                inputs = checkerboard_scrambled_gray_batch(inputs1, block_size)
            inputs = inputs.to(device)

            print("\n" + "data_index: " + str(data_index), "\n" + "-" * 10)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # track history
            with torch.set_grad_enabled(True):
                # Get model outputs and calculate loss
                output = model(inputs)
                loss = criterion(output, labels1)
                loss.backward()
                optimizer.step()

            train_losses.append(loss.item())

            confidence = torch.softmax(output, 1)
            for bat in range(output.shape[0]):
                confidence_coherent += confidence[bat][labels1[bat]].item()
            del confidence
            torch.cuda.empty_cache()
            _, pred = torch.max(output, 1)
            top5, top5_pred = torch.topk(output, 5, 1)

            print(f"\nloss1: {loss.detach():.4f} "
                  f"\nmax probability1: {_.detach()} "
                  f"\npreds1: ", pred.detach(),
                  f"\nlabels1: {labels1.detach()}")

            # loss = loss.item() * inputs.size(0)
            corrects_top1 = torch.sum(pred == labels1.detach())
            corrects_top5 = 0

            for i in range(len(labels1)):
                if labels1.detach()[i] in top5_pred[i]:
                    corrects_top5 += 1

            # statistics of running loss
            running_loss += loss.item()
            running_corrects_top1 += corrects_top1
            running_corrects_top5 += corrects_top5

            # statistics of epoch loss
            epoch_loss += loss.item()
            epoch_corrects_top1 += corrects_top1
            epoch_corrects_top5 += corrects_top5

            if (data_index + 1) % log_frequency == 0:  # write to log1 every 10 batches
                running_result = f"{epoch * batch_num + data_index:<30} " \
                                 f"{running_loss / (log_frequency * batch_size * 2):<30.6f}" \
                                 f"{float(running_corrects_top1) / (log_frequency * batch_size * 2):<30.6f}" \
                                 f"{float(running_corrects_top5) / (log_frequency * batch_size * 2):<30.6f}" + "\n"
                running_log.write(running_result)
                running_log.close()
                running_loss = 0.0
                running_corrects_top1 = 0
                running_corrects_top5 = 0

            ########## to delete ###########
            if data_index == 9:
                break

        ######################
        # validate the model #
        ######################
        model.eval()
        for data_index, (inputs1, labels1) in enumerate(validloaders1):
            labels1 = labels1.to(device)
            if random_block_size:
                block_size = random.choice(block_sizes)

            if intact:
                inputs = checkerboard_intact_gray_batch(inputs1, block_size)
            else:
                inputs = checkerboard_scrambled_gray_batch(inputs1, block_size)
            inputs = inputs.to(device)
            with torch.set_grad_enabled(False):
                output = model(inputs)
                loss = criterion(output, labels1)
                valid_losses.append(loss.item())

            ########## to delete ###########
            if data_index == 9:
                break

        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        train_losses = []
        valid_losses = []

        epoch_result = (f'{epoch:<30} '
                        f'{epoch_loss / (2 * len(trainloaders1.dataset)) :<30.6f} '
                        f'{float(epoch_corrects_top1) / (2 * len(trainloaders1.dataset)):<30.6f} '
                        f'{float(epoch_corrects_top5) / (2 * len(trainloaders1.dataset)):<30.6f}'
                        f" confidence coherent: {confidence_coherent}" + "\n")

        epoch_log.write(epoch_result)
        epoch_log.close()

        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            stop_point = epoch
            break
    if stop_point == 0:
        stop_point = num_epochs
    model.load_state_dict(torch.load('checkpoint.pt'))
    # save the model every epoch
    if os.path.exists(model_path):
        torch.save(model, model_path + "/model.pkl" + str(stop_point))
    else:
        os.mkdir(model_path)
        torch.save(model, model_path + "/model.pkl" + str(stop_point))
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    return model, avg_train_losses, avg_valid_losses, stop_point
