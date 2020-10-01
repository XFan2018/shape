import torch
import torchvision
from project1.Vgg16Model_val_imagenet import ConfigTestImagenet
from project2.test_before_finetune_silhouette import test_model


def run_test_before_finetune():
    config1 = ConfigTestImagenet()

    test_dataloader1 = config1.test_loader_imagenet

    model = torchvision.models.vgg16_bn(True)

    if torch.cuda.is_available():
        model.cuda()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    test_model(model, test_dataloader1, device)


if __name__ == "__main__":
    # run_test_before_finetune()
    config1 = ConfigTestImagenet()

    test_dataloader1 = config1.test_loader_imagenet
    dataset = test_dataloader1.dataset
    print(dataset.class_to_idx)
