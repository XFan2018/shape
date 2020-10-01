import torch
import torchvision
import torch.nn as nn
from project2.finetune_silhouette_and_imagenet.Vgg16Model_finetune_silhouette_and_imagenet import ConfigFinetuneImagenetSilhouette
from project2.finetune_silhouette_and_imagenet.finetune_silhouette_and_imagenet import finetune_model_silhouette_and_imagenet


def run_finetune():
    config1 = ConfigFinetuneImagenetSilhouette()

    finetune_dataloader1 = config1.train_loader_imagenet

    criterion = nn.CrossEntropyLoss()

    model = torchvision.models.vgg16_bn(True)

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=config1.lr,
                                momentum=config1.momentum,
                                weight_decay=config1.weight_decay)

    if torch.cuda.is_available():
        model.cuda()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    finetune_model_silhouette_and_imagenet(model=model,
                                           trainloader=finetune_dataloader1,
                                           criterion=criterion,
                                           optimizer=optimizer,
                                           num_epochs=config1.epochs,
                                           device=device,
                                           batch_size=config1.batch_size)


if __name__ == "__main__":
    run_finetune()
    # test_dataloader1 = config1.train_loader_imagenet
    # dataset = test_dataloader1.dataset
    # print(dataset.class_to_idx)

