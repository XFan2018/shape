import torch
import torchvision
from project1.vgg16_model_damagenet import ConfigTestImagenet
from project1.test import test_model


def run_test_damagenet():
    log_path = "log_test_damagenet"

    config1 = ConfigTestImagenet()

    test_dataloader1 = config1.test_loader

    model = torchvision.models.vgg16_bn(True)

    if torch.cuda.is_available():
        model.cuda()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    test_model(model, test_dataloader1, log_path, device, False, 224)


if __name__ == "__main__":
    run_test_damagenet()
