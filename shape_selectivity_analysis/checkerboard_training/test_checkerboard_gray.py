import time
import torch
from os import path
from scramble_checker_board import checker_board_scrambled_gray_batch, checker_board_intact_gray_batch
import os


def test_model_gray(model, test_loader, log_path, device, block_size, model_path, model_number, intact, horizontal):
    since = time.time()
    if path.exists(log_path):
        f_log = open(log_path + "/test_gray_after_finetune.txt", "a+")
    else:
        os.mkdir(log_path)
        f_log = open(log_path + "/test_gray_after_finetune.txt", "a+")

    # set evaluate mode
    model.eval()

    running_corrects = 0
    running_corrects_top5 = 0
    confidence_score = 0

    # test model
    for index, (inputs, labels) in enumerate(test_loader):
        labels = labels.to(device)
        if intact:
            inputs = checker_board_intact_gray_batch(inputs, block_size)
        else:
            inputs = checker_board_scrambled_gray_batch(inputs, block_size, horizontal)
        inputs = inputs.to(device)

        print("\nval", str(index), "\n" + "-" * 10)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            confidence = torch.softmax(outputs, 1)
            for bat in range(outputs.shape[0]):
                confidence_score += confidence[bat][labels[bat]]
            _, preds = torch.max(outputs, 1)
            top5_, top5_preds = torch.topk(outputs, 5, 1)

            print(f"\nmax outputs: {_.detach()} "
                  f"\npreds: ", preds.detach(),
                  f"\nlabels: {labels.detach()}")

        running_corrects += torch.sum(preds == labels.detach())
        for i in range(len(labels)):
            if labels.detach()[i] in top5_preds[i]:
                running_corrects_top5 += 1

        print(f"running corrects: {running_corrects}\t{running_corrects.double()/((index+1) * len(labels)):.4f}")
        print(f"running corrects top5: {running_corrects_top5}\t"
              f"{float(running_corrects_top5)/((index+1) * len(labels)):.4f}")
        # if index == 10:
        #     break

    total_acc = running_corrects.double() / len(test_loader.dataset)
    total_acc_top5 = float(running_corrects_top5) / len(test_loader.dataset)
    time_elapsed = time.time() - since

    tag = model_path + model_number

    result = f"{tag:<30}" + 'time_elapsed: {:4f} Acc: {:.4f} Acc_top5: {:.4f} confidence: {:.4f} intact:{}\n'.format(time_elapsed, total_acc,
                                                                                        total_acc_top5, confidence_score, intact)
    f_log.write(result)
    f_log.close()
