import torch
import torchvision
import torch.nn as nn
from project2.finetune_silhouette.finetune_silhouette import finetune_model_silhouette_es
from project2.finetune_silhouette.Vgg16Model_finetune_silhouette import ConfigFinetuneImagenetSilhouette
from project2.finetune_silhouette.test_after_finetune_silhouette import test_model
import matplotlib.pyplot as plt


def run_finetune():
    config1 = ConfigFinetuneImagenetSilhouette("D:\\projects\\summerProject2020\\project3\\animal_silhouette_training", True)
    config2 = ConfigFinetuneImagenetSilhouette("D:\\projects\\summerProject2020\\project3\\animal_silhouette_validation", True)

    finetune_trainloader = config1.loader_imagenet
    finetune_validloader = config2.loader_imagenet

    criterion = nn.CrossEntropyLoss()

    model = torchvision.models.vgg16_bn(True)

    optimizer = torch.optim.SGD(model.parameters(),
                                lr=config1.lr,
                                momentum=config1.momentum,
                                weight_decay=config1.weight_decay)

    if torch.cuda.is_available():
        model.cuda()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model, train_loss, valid_loss, stop_point = finetune_model_silhouette_es(model=model,
                                                                             trainloader=finetune_trainloader,
                                                                             validloader=finetune_validloader,
                                                                             criterion=criterion,
                                                                             optimizer=optimizer,
                                                                             num_epochs=config1.epochs,
                                                                             device=device,
                                                                             batch_size=config1.batch_size,
                                                                             patience=20)
    return model, train_loss, valid_loss, stop_point


def run_test_after_finetune(model):
    config1 = ConfigFinetuneImagenetSilhouette("D:\\projects\\summerProject2020\\project3\\animal_silhouette_testing", False)

    test_dataloader = config1.loader_imagenet

    if torch.cuda.is_available():
        model.cuda()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    test_model(model, test_dataloader, device)


def plot(train_loss, valid_loss):
    fig = plt.figure(figsize=(10, 8))
    plt.plot(range(1, len(train_loss) + 1), train_loss, label='Training Loss')
    plt.plot(range(1, len(valid_loss) + 1), valid_loss, label='Validation Loss')

    # find position of lowest validation loss
    minposs = valid_loss.index(min(valid_loss)) + 1
    plt.axvline(minposs, linestyle='--', color='r', label='Early Stopping Checkpoint')

    plt.xlabel('epochs')
    plt.ylabel('loss')
    # plt.ylim(0, 0.5)  # consistent scale
    plt.xlim(0, len(train_loss) + 1)  # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    fig.savefig("finetune_low_resolution_silhouettes_loss_plot.png", bbox_inches='tight')


if __name__ == "__main__":
    # model, train_loss, valid_loss, stop_point = run_finetune()
    # plot(train_loss, valid_loss)
    model = torch.load("D:\\projects\\summerProject2020\\project2\\finetune_silhouette\\finetune_model_silhouette_es_low_res\\model.pkl13")
    run_test_after_finetune(model)
