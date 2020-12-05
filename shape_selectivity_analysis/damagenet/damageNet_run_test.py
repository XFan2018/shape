import torch
import sys

sys.path.append(r'D:\projects\shape')
import torchvision
import argparse
from shape_selectivity_analysis.damagenet.vgg16_model_damagenet import ConfigTestImagenet
from shape_selectivity_analysis.Imagenet_training.test import test_model

parser = argparse.ArgumentParser(description="test using DamageNet dataset")
parser.add_argument("-ds", "--dataset", help="DamageNet dataset")
parser.add_argument("-lb", "--label", help="path to the DamageNet label file")
parser.add_argument("-lp", "--log_path", help="log path")
parser.add_argument("-bs", "--batch_size", help="number of batch size for dataloader")
parser.add_argument("-wk", "--workers", help="number of workers for the dataloader")
parser.add_argument("-md", "--model", help="path to the test model. If empty, use pre-trained VGG16")
args = parser.parse_args()


def run_test_damagenet():
    log_path = args.log_path

    config = ConfigTestImagenet(args.dataset, args.label, int(args.batch_size), int(args.workers))

    dataloader = config.test_loader

    if not args.model:
        model = torchvision.models.vgg16_bn(True)
    else:
        model = torch.load(args.model)

    if torch.cuda.is_available():
        model.cuda()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    test_model(model, dataloader, log_path, device, False, 224)


if __name__ == "__main__":
    run_test_damagenet()
