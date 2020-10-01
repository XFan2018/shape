import torch
import torchvision
from project1.Vgg16Model_val_imagenet import ConfigTestImagenet
from project1.test import test_model


def run_test_before_finetune():
    log_path = "log_test_before_finetune_horizontal_scramble"

    # config1 = ConfigTestImagenet()

    # test_dataloader1 = config1.test_loader_imagenet_horizontal_scrambled

    model = torchvision.models.vgg16_bn(True)

    if torch.cuda.is_available():
        model.cuda()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # test_model(model, test_dataloader1, log_path, device, False, 224)

    arr = [8, 16, 28, 56, 112]
    for size in arr:
        print("start...\nsize: ", size)

        config1 = ConfigTestImagenet(size)  # arg is block size

        test_dataloader1 = config1.test_loader_imagenet_scrambled

        test_model(model, test_dataloader1, log_path, device, True, size)


if __name__ == "__main__":
    run_test_before_finetune()
