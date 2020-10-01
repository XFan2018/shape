"""
log the accuracy of each classes
"""
import time
import torch
from os import path
import os


def test_model(model, test_loader, device):
    since = time.time()
    # class labels
    classes = [i for i in range(1000)] # number of classes
    # a dictionary contains accuracy of each classes
    classes_acc = {}
    for c in classes:
        classes_acc[c] = [0, 20]  # (correct number per class, data number per class)
    log_path = "log_test_after_finetune_silhouette_low_resolution"
    if path.exists(log_path):
        f_log = open(log_path + "/test_after_finetune_model.txt", "a+")
    else:
        os.mkdir(log_path)
        f_log = open(log_path + "/test_after_finetune_model.txt", "a+")

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
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            top5_, top5_preds = torch.topk(outputs, 5, 1)

            print(f"\nmax outputs: {_.detach()} "
                  f"\npreds: ", preds.detach(),
                  f"\nlabels: {labels.detach()}")

        running_corrects += torch.sum(preds == labels.detach())
        for i in range(len(labels)):
            if labels.detach()[i] in top5_preds[i]:
                running_corrects_top5 += 1

            if labels.detach()[i] == preds[i]:
                classes_acc[labels[i].item()][0] += 1

        print(f"running corrects: {running_corrects}\t{running_corrects.double()/((index+1) * len(labels)):.4f}")
        print(f"running corrects top5: {running_corrects_top5}\t"
              f"{float(running_corrects_top5)/((index+1) * len(labels)):.4f}")
        #if index == 5:
        #    break

    total_acc = running_corrects.double() / len(test_loader.dataset)
    total_acc_top5 = float(running_corrects_top5) / len(test_loader.dataset)
    time_elapsed = time.time() - since

    result = f"Model_id: {total_acc:<30}" + 'time_elapsed: {:4f} Acc: {:.4f} Acc_top5: {:.4f}\n'.format(time_elapsed, total_acc,
                                                                                              total_acc_top5)
    f_log.write(result)
    for cls in classes:
        result = f"class {cls:<30} corrects: {classes_acc[cls][0]} " \
                 f"total: {classes_acc[cls][1]} " \
                 f"Acc: {float(classes_acc[cls][0]) / classes_acc[cls][1]:.4f}\n"
        f_log.write(result)
    f_log.close()
