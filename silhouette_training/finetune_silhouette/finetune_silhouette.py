import time
import torch
import numpy as np
import os
from pytorchtools import EarlyStopping

"""
fine-tune on intact animal silhouettes
"""

np.set_printoptions(precision=2)


def finetune_model_silhouette_es(model, trainloader, validloader, criterion, optimizer, num_epochs, device, batch_size, patience):
    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    log_frequency = 16  # batch size is 256, write to log1 every 10 batches

    model_path = "finetune_model_silhouette_es_low_res"
    batch_num = len(trainloader.dataset) // batch_size
    print("batch number: ", str(batch_num) + "\n")
    # running log1, write to the log1 every "log_frequency" number of batches
    if os.path.exists("log_finetune_silhouette_shuffles_es_low_res"):
        running_log = open("log_finetune_silhouette_shuffles_es_low_res/train_finetune_running_loss.txt", "w+")
    else:
        os.mkdir("log_finetune_silhouette_shuffles_es_low_res")
        running_log = open("log_finetune_silhouette_shuffles_es_low_res/train_finetune_running_loss.txt", "w+")

    running_log.write("index".ljust(30) +
                      "every {} batches loss".format(log_frequency).ljust(30) +
                      "corrects top1".ljust(30) +
                      "\n")

    # write to the log_finetune_silhouette every epoch
    epoch_log = open("log_finetune_silhouette_shuffles_es_low_res/train_finetune_epoch_loss.txt", "w+")
    epoch_log.write("epoch".ljust(30) +
                    "epoch loss".ljust(30) +
                    "epoch correct top1".ljust(30) +
                    "\n")
    epoch_log.close()
    running_log.close()

    print("number of data in {} batches".format(log_frequency),
          batch_size * log_frequency)

    stop_point = 0  # the number of epoch

    # training
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        if os.path.exists("log_finetune_silhouette_shuffles_es_low_res"):
            epoch_log = open("log_finetune_silhouette_shuffles_es_low_res/train_finetune_epoch_loss.txt", "a+")

        model.train()  # Set model to training mode

        running_loss = 0.0
        running_corrects_top1 = 0

        epoch_loss = 0.0
        epoch_corrects_top1 = 0

        # Iterate over data.
        for data_index, (inputs, labels) in enumerate(trainloader):
            running_log = open("log_finetune_silhouette_shuffles_low_resolution/train_finetune_running_loss.txt", "a+")
            inputs = inputs.to(device)
            labels = labels.to(device)
            print("\n" + "data_index: " + str(data_index), "\n" + "-" * 10)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            # track history
            with torch.set_grad_enabled(True):
                # Get model outputs and calculate loss
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                print(f"\nloss1: {loss.detach():.4f} "
                      f"\nmax probability1: {_.detach()} "
                      f"\npreds1: ", preds.detach(),
                      f"\nlabels1: {labels.detach()}")

                loss.backward()

                optimizer.step()

                train_losses.append(loss.item())

            # loss = loss.item() * inputs.size(0)
            corrects_top1 = torch.sum(preds == labels.detach())

            # statistics of running loss
            running_loss += loss.item()
            running_corrects_top1 += corrects_top1

            # statistics of epoch loss
            epoch_loss += loss.item()
            epoch_corrects_top1 += corrects_top1

            if (data_index + 1) % log_frequency == 0:  # write to log1 every 10 batches
                running_result = f"{epoch * batch_num + data_index:<30} " \
                                 f"{running_loss / (log_frequency * batch_size * 2):<30.6f}" \
                                 f"{float(running_corrects_top1) / (log_frequency * batch_size * 2):<30.6f}" \
                                 "\n"
                running_log.write(running_result)
                running_log.close()
                running_loss = 0.0
                running_corrects_top1 = 0

        epoch_result = (f'{epoch:<30} '
                        f'{epoch_loss / (len(trainloader.dataset)) :<30.6f} '
                        f'{float(epoch_corrects_top1) / (len(trainloader.dataset)):<30.6f}'
                        '\n')

        epoch_log.write(epoch_result)
        epoch_log.close()
        ######################
        # validate the model #
        ######################
        model.eval()
        for data_index, (inputs, labels) in enumerate(validloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            print("\n" + "data_index: " + str(data_index), "\n" + "-" * 10)
            # track history
            with torch.set_grad_enabled(False):
                # Get model outputs and calculate loss
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                valid_losses.append(loss.item())
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        train_losses = []
        valid_losses = []
        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            stop_point = epoch
            break
    if stop_point == 0:
        stop_point = num_epochs
    # save the model
    model.load_state_dict(torch.load('checkpoint.pt'))
    if os.path.exists(model_path):
        torch.save(model, model_path + "/model.pkl" + str(stop_point-patience))
    else:
        os.mkdir(model_path)
        torch.save(model, model_path + "/model.pkl" + str(stop_point-patience))
    return model, avg_train_losses, avg_valid_losses, stop_point
