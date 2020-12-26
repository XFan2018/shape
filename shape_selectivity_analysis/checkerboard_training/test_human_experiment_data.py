import os
import torch
from torch import nn
import torchvision
import time
from os import path
import sys
sys.path.append(sys.path.append(r"D:\projects\shape"))
from settings import *
import warnings

warnings.filterwarnings("ignore")
from torchvision import transforms
from shape_selectivity_analysis.human_experiment_data_prep.category_mapping import *
from shape_selectivity_analysis.checkerboard_training.Vgg16Model_CheckerBoard_train import ConfigTrainImagenet
import argparse

parser = argparse.ArgumentParser(description="finetune with scrambled checkerboard image")
parser.add_argument("-ds", "--dataset", help="path to training dataset")
parser.add_argument("-ts", "--testset", help="path to testing dataset")
parser.add_argument("-vs", "--validset", help="path to validation dataset")
parser.add_argument("-ep", "--epochs", help="number of epochs", type=int)
parser.add_argument("-lr", "--learning_rate", help="learning rate")
parser.add_argument("-ltrp", "--log_training_path", help="log path of the training")
parser.add_argument("-ltsp", "--log_testing_path", help="log path of the testing")
parser.add_argument("-mp", "--model_path", help="model path")
parser.add_argument("-fc", "--fc_only", help="train fc only")
parser.add_argument("-bs", "--block_size", help="block size of checkerboard", type=int)
args = parser.parse_args()


def test_model_human_experiment(model, test_loader, log_path, device, model_path, batch_size):
    since = time.time()
    if path.exists(log_path):
        f_log = open(log_path + "/test_model_human_experiment.txt", "a+")
    else:
        os.mkdir(log_path)
        f_log = open(log_path + "/test_model_human_experiment.txt", "a+")

    # set evaluate mode
    model.eval()

    running_corrects = 0
    # running_corrects_top5 = 0
    # confidence_score = 0

    # test model
    for index, (inputs, labels) in enumerate(test_loader):
        labels = labels.to(device)
        inputs = inputs.to(device)

        print("\nval", str(index), "\n" + "-" * 10)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            confidence = torch.softmax(outputs, dim=1)
            # confidence_score += torch.sum(confidence[:][labels])

            # take the maximum from the 8 categories, don't care other categories
            # here preds' indices are not equal to the indices in ImageNet
            _, preds = torch.max(outputs[:, all8_cate_index], 1)

            # map preds indices back to that of ImageNet
            eight_cate_index = torch.tensor([all8_cate_index] * batch_size)
            preds = torch.tensor([eight_cate_index[i][preds[i]].item() for i in range(len(preds))])

            # convert ImageNet label to 8 category label
            for i in range(len(preds)):
                for key in index_mapping_dict.keys():
                    if preds[i].detach() in index_mapping_dict[key]:
                        preds[i] = key
            preds = preds.to(device)
            print(f"\nmax outputs: {_.detach()} "
                  f"\npreds: ", preds.detach(),
                  f"\nlabels: {labels.detach()}")

        running_corrects += torch.sum(preds.detach() == labels.detach())

        print(f"running corrects: {running_corrects}")

        # if index == 10:
        #     break

    total_acc = running_corrects.double() / len(test_loader.dataset)
    time_elapsed = time.time() - since

    tag = model_path

    result = f"{tag:<30}" + 'time_elapsed: {:4f} Acc: {:.4f}\n'.format(time_elapsed, total_acc)
    f_log.write(result)
    f_log.close()


def run_human_test_checkerboard():
    ################### prepare parameters ########################
    model = torchvision.models.vgg16_bn(pretrained=True)
    dataset_path = os.path.join(CHECKERBOARD_DATASET_HUMAN, "blocksize7")
    transform = torchvision.transforms.Compose([transforms.ToTensor(),
                                                transforms.Normalize((0.485, 0.456, 0.406),
                                                                     (0.229, 0.224, 0.225))])
    dataset = torchvision.datasets.ImageFolder(dataset_path, transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=2)
    # lr = 1e-3
    # criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        model.cuda()
    ################################################################

    test_model_human_experiment(model=model,
                                test_loader=dataloader,
                                device=device,
                                model_path=args.model_path,
                                log_path=args.log_testing_path + "_" + str(args.block_size),
                                batch_size=16)


if __name__ == "__main__":
    run_human_test_checkerboard()
    # l = [1, 2]
    # a = torch.rand((4,4))
    # _, preds = torch.max(a[:, l], 1)
    # print(a)
    # print(preds)
    # print(_)

