import torch
import torchvision
from project1.Vgg16Model_train_imagenet import ConfigTrainImagenet
from project1.finetune import train_model
from project1.fine_tune_loss import FinetuneLoss, KLLoss
import torch.nn as nn


def dfs_freeze(model):
    for name, child in model.named_children():
        if name == "features":
            for param in child.parameters():
                param.requires_grad = False
        dfs_freeze(child)


def run_train_finetune():
    config1 = ConfigTrainImagenet()

    test_dataloader = config1.train_loader_imagenet_scrambled

    criterion = FinetuneLoss()

    model = torchvision.models.vgg16_bn(True)

    model.apply(dfs_freeze)

    if torch.cuda.is_available():
        model.cuda()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    old_state_dict = {}
    for key in model.state_dict():
        old_state_dict[key] = model.state_dict()[key].clone()

    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=config1.lr,
                                momentum=config1.momentum,
                                weight_decay=config1.weight_decay)

    train_model(model,
                test_dataloader,
                criterion,
                optimizer,
                config1.epochs,
                device,
                config1.batch_size)

    new_state_dict = {}
    for key in model.state_dict():
        new_state_dict[key] = model.state_dict()[key].clone()

    # Compare params
    count = 0
    for key in old_state_dict:
        if not (old_state_dict[key] == new_state_dict[key]).all():
            print('Diff in {}'.format(key))
            count += 1
    print(count)


if __name__ == "__main__":
    run_train_finetune()
    # model = torch.load("finetune_model/model.pkl")
    # print(model)
