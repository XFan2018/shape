import time
import torch
from os import path
import os
torch.manual_seed(os.getenv("SEED"))
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def test_model(model, test_loader, log_path, device, scrambled, size):
    since = time.time()
    if path.exists(log_path):
        f_log = open(log_path + "/test_before_finetune.txt", "a+")
    else:
        os.mkdir(log_path)
        f_log = open(log_path + "/test_before_finetune.txt", "a+")

    # set evaluate mode
    model.eval()

    running_corrects = 0
    running_corrects_top5 = 0
    total_confidence = 0

    # test model
    for index, (inputs, labels) in enumerate(test_loader):
        print(labels)
        inputs = inputs.to(device)
        labels = labels.to(device)

        print("\nval", str(index), "\n" + "-" * 10)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            confidence = torch.softmax(outputs, 1)
            for bat in range(outputs.shape[0]):
                total_confidence += confidence[bat][labels[bat]]
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
        # if index == 1:
        #     break

    total_acc = running_corrects.double() / len(test_loader.dataset)
    total_acc_top5 = float(running_corrects_top5) / len(test_loader.dataset)
    time_elapsed = time.time() - since

    if scrambled:
        tag = "scrambled block_size=" + str(size) + ":"
    else:
        tag = "original: "

    result = f"{tag:<30}" + 'time_elapsed: {:4f} Acc: {:.4f} Acc_top5: {:.4f} confidence: {: .4f}\n'.format(time_elapsed, total_acc,
                                                                                        total_acc_top5, total_confidence)
    f_log.write(result)
    f_log.close()
    return total_acc
