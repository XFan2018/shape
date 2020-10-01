import torch
import torchvision
import torch.nn as nn
from project2.finetune_FC_silhouette_and_imagenet.Vgg16Model_finetune_FC_silhouette_and_imagenet import \
    ConfigFinetuneImagenetSilhouette
from project2.finetune_FC_silhouette_and_imagenet.finetune_FC_silhouette_and_imagenet import \
    finetune_model_FC_silhouette_and_imagenet


def dfs_freeze(model):
    for name, child in model.named_children():
        if name == "features":
            for param in child.parameters():
                param.requires_grad = False
        dfs_freeze(child)


def run_finetune():
    config1 = ConfigFinetuneImagenetSilhouette()

    finetune_dataloader1 = config1.train_loader_imagenet

    criterion = nn.CrossEntropyLoss()

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

    finetune_model_FC_silhouette_and_imagenet(model=model,
                                              trainloader=finetune_dataloader1,
                                              criterion=criterion,
                                              optimizer=optimizer,
                                              num_epochs=config1.epochs,
                                              device=device,
                                              batch_size=config1.batch_size)
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
    run_finetune()
