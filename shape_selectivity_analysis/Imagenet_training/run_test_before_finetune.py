import torch
import torchvision
from Vgg16Model_val_imagenet import ConfigTestImagenet
from test import test_model


def run_test_before_finetune():
    log_path = "log_test_before_finetune"

    model = torchvision.models.vgg16_bn(True)

    if torch.cuda.is_available():
        model.cuda()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    acc = 1

    while acc > 0.5:

        with torch.no_grad():
            for param in model.parameters():
                param.add_(torch.randn(param.size()) * 0.001)

        config1 = ConfigTestImagenet()

        test_dataloader1 = config1.test_loader_imagenet

        acc = test_model(model, test_dataloader1, log_path, device, False, 224)

    torch.save(model, "modified_model")


if __name__ == "__main__":
    run_test_before_finetune()
