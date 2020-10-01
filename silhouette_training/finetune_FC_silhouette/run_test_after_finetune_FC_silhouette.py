import torch
from project2.finetune_FC_silhouette.Vgg16Model_FC_test_silhouette import ConfigFinetuneSilhouette
from project2.finetune_FC_silhouette.test_after_finetune_FC_silhouette import test_model


def run_test_after_finetune():
    config1 = ConfigFinetuneSilhouette()

    test_dataloader1 = config1.loader_dataset

    for i in range(60):
        model = torch.load("/Users/leo/PycharmProjects/summerProject/project2/finetune_model_silhouette/model.pkl" + str(i))

        if torch.cuda.is_available():
            model.cuda()

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        test_model(model, test_dataloader1, device, i)


if __name__ == "__main__":
    run_test_after_finetune()
    # config1 = ConfigFinetuneSilhouette()
    # test_dataloader1 = config1.loader_dataset
    # dataset = test_dataloader1.dataset
    # print(dataset.class_to_idx)
