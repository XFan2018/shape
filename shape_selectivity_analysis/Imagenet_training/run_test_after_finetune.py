import torch
import os
from project1.Vgg16Model_val_imagenet import ConfigTestImagenet
from project1.test_after_finetune import test_finetuned_model
torch.manual_seed(os.getenv("SEED"))
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def run_test_after_finetune():
    config1 = ConfigTestImagenet()

    test_dataloader1 = config1.test_loader_imagenet

    model = torch.load("finetune_model/model.pkl29")
    # model = torchvision.models.vgg16_bn(True)

    if torch.cuda.is_available():
        model.cuda()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    test_finetuned_model(model, test_dataloader1, device, False, 224)

    arr = [32, 56, 112]
    for size in arr:
        print("start...\nsize: ", size)

        config1 = ConfigTestImagenet(size)  # arg is block size

        test_dataloader2 = config1.test_loader_imagenet_scrambled

        test_finetuned_model(model, test_dataloader2, device, True, size)

    # dataiter = iter(test_dataloader1)
    #
    # img, target = dataiter.next()
    #
    # show_multi_images(img)
    #
    # print(target)


if __name__ == "__main__":
    config = ConfigTestImagenet()
    # loader = config.test_loader_imagenet
    # dataset = loader.dataset
    # print(dataset.class_to_idx)
