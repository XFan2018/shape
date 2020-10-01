import torch
from project2.finetune_silhouette_and_imagenet.Vgg16Model_test_silhouette_and_imagenet import ConfigTestImagenet
from project2.finetune_silhouette.test_after_finetune_silhouette import test_model


def run_test_after_finetune():
    config1 = ConfigTestImagenet()

    test_dataloader1 = config1.test_loader_imagenet

    for i in range(4, 60, 5):
        model = torch.load("/Users/leo/PycharmProjects/summerProject/project2/finetune_model_silhouette/model.pkl" + str(i))

        if torch.cuda.is_available():
            model.cuda()

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        test_model(model, test_dataloader1, device, i)


if __name__ == "__main__":
    run_test_after_finetune()
    # config1 = ConfigTestImagenet()
    #
    # test_dataloader1 = config1.test_loader_imagenet
    # dataset = test_dataloader1.dataset
    # print(dataset.class_to_idx)
