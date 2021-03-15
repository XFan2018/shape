import time
import torch
import numpy as np
import os
from scramble_checkerboard import checkerboard_batch
import torchvision.transforms as transforms

"""
fine-tune on scrambled checker board
two dataloaders are used, one is for ground truth images, and the other is for adversarial attack images
"""

np.set_printoptions(precision=2)


def test_model(model, testloaders1, testloaders2, device, batch_size, model_path, log_path, block_size, horizontal):
    since = time.time()

    if os.path.exists(log_path):
        log = open(log_path + "/test.txt", "a+")
    else:
        os.mkdir(log_path)
        log = open(log_path + "/test.txt", "a+")

    # testing

    model.eval()  # Set model to testing mode

    running_corrects_top1 = 0
    running_attack_corrects_top1 = 0
    running_corrects_top5 = 0
    running_attack_corrects_top5 = 0
    confidence_coherent = 0
    confidence_adversarial = 0

    log = open(log_path + "/test.txt", "a+")
    # Iterate over data.
    for data_index, ((inputs1, labels1), (inputs2, labels2)) in enumerate(zip(testloaders1, testloaders2)):

        inputs = checkerboard_batch(inputs1, inputs2, block_size, horizontal)
        inputs = inputs.to(device)
        labels1 = labels1.to(device)
        labels2 = labels2.to(device)

        print("\n" + "data_index: " + str(data_index), "\n" + "-" * 10)
        # forward
        # track history
        with torch.set_grad_enabled(False):
            # Get model outputs and calculate loss
            output = model(inputs)
            confidence = torch.softmax(output, 1)
            for bat in range(output.shape[0]):
                confidence_coherent += confidence[bat][labels1[bat]]
                confidence_adversarial += confidence[bat][labels2[bat]]
            _, pred = torch.max(output, 1)
            top5, top5_pred = torch.topk(output, 5, 1)

            print(f"\nmax probability1: {_.detach()} "
                  f"\npreds1: ", pred.detach(),
                  f"\nlabels1: {labels1.detach()}"
                  f"\nlabels2: {labels2.detach()}")

        corrects_top1 = torch.sum(pred == labels1.detach())
        attack_corrects_top1 = torch.sum(pred == labels2.detach())
        corrects_top5 = 0
        attack_corrects_top5 = 0

        for i in range(len(labels1)):
            if labels1.detach()[i] in top5_pred[i]:
                corrects_top5 += 1
            if labels2.detach()[i] in top5_pred[i]:
                attack_corrects_top5 += 1

        # statistics of running loss
        running_corrects_top1 += corrects_top1
        running_attack_corrects_top1 += attack_corrects_top1
        running_corrects_top5 += corrects_top5
        running_attack_corrects_top5 += attack_corrects_top5

        ########## to delete ###########
    #        if data_index == 10:
    #            break

    running_result = f"{model_path:<30}" \
                     f"top1: {running_corrects_top1}" + ", " + f"{float(running_corrects_top1) / ((data_index + 1) * batch_size):<30.6f}" \
                     f"top5: {running_corrects_top5}" + ", " + f"{float(running_corrects_top5) / ((data_index + 1) * batch_size) :<30.6f}" \
                     f"attack top1: {running_attack_corrects_top1}" + ", " + f"{float(running_attack_corrects_top1) / ((data_index + 1) * batch_size):<30.6f}" \
                     f"attack top5: {running_attack_corrects_top5}" + ", " + f"{float(running_attack_corrects_top5) / ((data_index + 1) * batch_size):.6f}" \
                     f"confidence coherent: {confidence_coherent}" + ", " + f"confident adversarial: {confidence_adversarial}"+"\n"
    log.write(running_result)
    log.close()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
