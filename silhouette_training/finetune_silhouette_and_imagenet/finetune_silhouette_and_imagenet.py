import time
import torch
import numpy as np
import os

"""
fine-tune on intact animal silhouettes
"""

np.set_printoptions(precision=2)


def finetune_model_silhouette_and_imagenet(model, trainloader, criterion, optimizer, num_epochs, device, batch_size):
    since = time.time()

    log_frequency = 16  # batch size is 256, write to log1 every 10 batches

    model_path = "finetune_model_silhouette"
    model_weight_path = "finetune_model_weight_silhouette"

    batch_num = len(trainloader.dataset) // batch_size
    print("batch number: ", str(batch_num) + "\n")

    # running log1, write to the log1 every "log_frequency" number of batches
    if os.path.exists("log_finetune_silhouette"):
        running_log = open("log_finetune_silhouette/train_finetune_running_loss.txt", "w+")
    else:
        os.mkdir("log_finetune_silhouette")
        running_log = open("log_finetune_silhouette/train_finetune_running_loss.txt", "w+")

    running_log.write("index".ljust(30) +
                      "every {} batches loss".format(log_frequency).ljust(30) +
                      "corrects top1".ljust(30) +
                      "corrects top5\n")

    # write to the log_finetune_silhouette every epoch
    if os.path.exists("log_finetune_silhouette"):
        epoch_log = open("log_finetune_silhouette/train_finetune_epoch_loss.txt", "w+")

    epoch_log.write("epoch".ljust(30) +
                    "epoch loss".ljust(30) +
                    "epoch correct top1".ljust(30) +
                    "epoch correct top5\n")
    epoch_log.close()
    running_log.close()

    print("number of data in {} batches".format(log_frequency),
          batch_size * log_frequency)

    # training
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        if os.path.exists("log_finetune_silhouette"):
            epoch_log = open("log_finetune_silhouette/train_finetune_epoch_loss.txt", "a+")

        model.train()  # Set model to training mode

        running_loss = 0.0
        running_corrects_top1 = 0
        running_corrects_top5 = 0

        epoch_loss = 0.0
        epoch_corrects_top1 = 0
        epoch_corrects_top5 = 0

        # Iterate over data.
        for data_index, (inputs, labels) in enumerate(trainloader):
            running_log = open("log_finetune_silhouette/train_finetune_running_loss.txt", "a+")

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
                top5, top5_preds = torch.topk(outputs, 5, 1)

                print(f"\nloss1: {loss.detach():.4f} "
                      f"\nmax probability1: {_.detach()} "
                      f"\npreds1: ", preds.detach(),
                      f"\nlabels1: {labels.detach()}")

                loss.backward()

                optimizer.step()

            # loss = loss.item() * inputs.size(0)
            corrects_top1 = torch.sum(preds == labels.detach())
            corrects_top5 = 0

            for i in range(len(labels)):
                if labels.detach()[i] in top5_preds[i]:
                    corrects_top5 += 1

            # statistics of running loss
            running_loss += loss
            running_corrects_top1 += corrects_top1
            running_corrects_top5 += corrects_top5

            # statistics of epoch loss
            epoch_loss += loss
            epoch_corrects_top1 += corrects_top1
            epoch_corrects_top5 += corrects_top5

            if (data_index + 1) % log_frequency == 0:  # write to log1 every 10 batches
                running_result = f"{epoch * batch_num + data_index:<30} " \
                                 f"{running_loss / (log_frequency * batch_size * 2):<30.6f}" \
                                 f"{float(running_corrects_top1) / (log_frequency * batch_size * 2):<30.6f}" \
                                 f"{float(running_corrects_top5) / (log_frequency * batch_size * 2):.6f}\n"
                running_log.write(running_result)
                running_log.close()
                running_loss = 0.0
                running_corrects_top1 = 0
                running_corrects_top5 = 0

            ########## to delete ###########
            if data_index == 2:
                break

        epoch_result = (f'{epoch:<30} '
                        f'{epoch_loss / (len(trainloader.dataset)) :<30.6f} '
                        f'{float(epoch_corrects_top1) / (len(trainloader.dataset)):<30.6f} '
                        f'{float(epoch_corrects_top5) / (len(trainloader.dataset)):.6f}\n')

        epoch_log.write(epoch_result)
        epoch_log.close()

        # save the model
        if os.path.exists(model_path):
            torch.save(model, model_path + "/model.pkl" + str(epoch))
        else:
            os.mkdir(model_path)
            torch.save(model, model_path + "/model.pkl" + str(epoch))

        if os.path.exists(model_weight_path):
            torch.save(model.state_dict(), model_weight_path + "/param.pkl" + str(epoch))
        else:
            os.mkdir(model_weight_path)
            torch.save(model.state_dict(), model_weight_path + "/param.pkl" + str(epoch))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
