import torch
import torchvision
from Vgg16Model_CheckerBoard_train import ConfigTrainImagenet
from finetune_checkerboard import train_model
from finetune_gray_checkerboard import train_model_gray
from test_checkerboard import test_model
from test_checkerboard_intact import test_model_intact
from test_checkerboard_gray import test_model_gray
import matplotlib.pyplot as plt
import torch.nn as nn
import argparse

parser = argparse.ArgumentParser(description="finetune with scrambled checkerboard image")
parser.add_argument("-ds", "--dataset", help="path to training dataset")
parser.add_argument("-ts", "--testset", help="path to testing dataset")
parser.add_argument("-vs", "--validset", help="path to validation dataset")
parser.add_argument("-ep", "--epochs", help="number of epochs")
parser.add_argument("-lr", "--learning_rate", help="learning rate")
parser.add_argument("-ltrp", "--log_training_path", help="log path of the training")
parser.add_argument("-ltsp", "--log_testing_path", help="log path of the testing")
parser.add_argument("-mp", "--model_path", help="model path")
parser.add_argument("-fc", "--fc_only", help="train fc only")
parser.add_argument("-bs", "--block_size", help="block size of checkerboard")
args = parser.parse_args()
block_sizes = [112, 56, 28, 16, 8]


def dfs_freeze(model):
    for name, child in model.named_children():
        if name == "features":
            for param in child.parameters():
                param.requires_grad = False
        dfs_freeze(child)


def run_train_finetune(block_size, horizontal):
    config_train = ConfigTrainImagenet(args.dataset, int(args.epochs), float(args.learning_rate), shuffle=True)
    config_valid = ConfigTrainImagenet(args.validset, int(args.epochs), float(args.learning_rate), shuffle=False)
    train_dataloader = config_train.loader
    valid_dataloader = config_valid.loader
    model = torchvision.models.vgg16_bn(True)
    if bool(args.fc_only):
        model.apply(dfs_freeze)
    if torch.cuda.is_available():
        model.cuda()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    old_state_dict = {}
    if bool(args.fc_only):
        for key in model.state_dict():
            old_state_dict[key] = model.state_dict()[key].clone()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=config_train.lr)

    model_trained, avg_train_losses, avg_valid_losses, stop_point = train_model(model=model,
                                                                                trainloaders1=train_dataloader,
                                                                                trainloaders2=train_dataloader,
                                                                                validloaders1=valid_dataloader,
                                                                                validloaders2=valid_dataloader,
                                                                                criterion=criterion,
                                                                                optimizer=optimizer,
                                                                                num_epochs=config_train.epochs,
                                                                                device=device,
                                                                                batch_size=config_train.batch_size,
                                                                                model_path=args.model_path + "_" + str(
                                                                                    block_size),
                                                                                log_path=args.log_training_path + "_" + str(
                                                                                    block_size),
                                                                                block_size=block_size,
                                                                                fc_only=bool(args.fc_only),
                                                                                patience=20,
                                                                                horizontal=horizontal)

    if bool(args.fc_only):
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
    return model_trained, avg_train_losses, avg_valid_losses, stop_point


def run_train_finetune_gray(block_size, horizontal):
    config_train = ConfigTrainImagenet(args.dataset, int(args.epochs), float(args.learning_rate), shuffle=True)
    config_valid = ConfigTrainImagenet(args.validset, int(args.epochs), float(args.learning_rate), shuffle=False)
    train_dataloader = config_train.loader
    valid_dataloader = config_valid.loader
    model = torchvision.models.vgg16_bn(True)
    if bool(args.fc_only):
        model.apply(dfs_freeze)
    if torch.cuda.is_available():
        model.cuda()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    old_state_dict = {}
    if bool(args.fc_only):
        for key in model.state_dict():
            old_state_dict[key] = model.state_dict()[key].clone()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=config_train.lr)

    model_trained, avg_train_losses, avg_valid_losses, stop_point = train_model_gray(model=model,
                                                                                     trainloaders1=train_dataloader,
                                                                                     validloaders1=valid_dataloader,
                                                                                     criterion=criterion,
                                                                                     optimizer=optimizer,
                                                                                     num_epochs=config_train.epochs,
                                                                                     device=device,
                                                                                     batch_size=config_train.batch_size,
                                                                                     model_path=args.model_path + "_" + str(
                                                                                         block_size),
                                                                                     log_path=args.log_training_path + "_" + str(
                                                                                         block_size),
                                                                                     block_size=block_size,
                                                                                     fc_only=bool(args.fc_only),
                                                                                     patience=20,
                                                                                     horizontal=horizontal)

    if bool(args.fc_only):
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
    return model_trained, avg_train_losses, avg_valid_losses, stop_point


def run_test_scramble_checkerboard(model_trained, block_size, stop_point, horizontal):
    config1 = ConfigTrainImagenet(args.testset)
    config2 = ConfigTrainImagenet(args.testset, shuffle=True)
    test_dataloader1 = config1.loader
    test_dataloader2 = config2.loader
    model = torchvision.models.vgg16_bn(True)
    if torch.cuda.is_available():
        model.cuda()
        model_trained.cuda()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # test pre-trained model
    test_model(model=model,
               testloaders1=test_dataloader1,
               testloaders2=test_dataloader2,
               device=device,
               batch_size=config1.batch_size,
               model_path="model_0",
               log_path=args.log_testing_path,
               block_size=block_size,
               horizontal=horizontal)

    # test finetuned model
    test_model(model=model_trained,
               testloaders1=test_dataloader1,
               testloaders2=test_dataloader2,
               device=device,
               batch_size=config1.batch_size,
               model_path="model_" + str(block_size) + str(stop_point),
               log_path=args.log_testing_path,
               block_size=block_size,
               horizontal=horizontal)


def run_test_intact(model_trained, block_size, stop_point):
    config1 = ConfigTrainImagenet(args.testset)
    test_dataloader = config1.intact_loader
    model = model_trained
    if torch.cuda.is_available():
        model.cuda()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_model_intact(model=model,
                      test_loader=test_dataloader,
                      log_path=args.log_testing_path + "_intact",
                      device=device,
                      model_path=args.model_path + "_" + str(block_size),
                      model_number=str(stop_point))


def run_test_scrambled(model_trained, block_size, stop_point, horizontal):
    config1 = ConfigTrainImagenet(args.testset, block_size=block_size)
    test_loader = config1.scrambled_loader
    model = model_trained
    if torch.cuda.is_available():
        model.cuda()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_model_intact(model=model,
                      test_loader=test_loader,
                      log_path=args.log_testing_path + "_scrambled",
                      device=device,
                      model_path=args.model_path + "_" + str(block_size),
                      model_number=str(stop_point))


def run_test_gray(model_trained, block_size, stop_point, intact, horizontal):
    config1 = ConfigTrainImagenet(args.testset, block_size=block_size)
    test_loader = config1.loader
    model = model_trained
    if torch.cuda.is_available():
        model.cuda()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_model_gray(model=model,
                    test_loader=test_loader,
                    log_path=args.log_testing_path + "_gray",
                    device=device,
                    block_size=block_size,
                    model_path=args.model_path + "_" + str(block_size),
                    model_number=str(stop_point),
                    intact=intact,
                    horizontal=horizontal)


def run(block_size, horizontal):
    model_trained, train_loss, valid_loss, stop_point = run_train_finetune(block_size, horizontal)
    plot(block_size, train_loss, valid_loss)
    run_test_scramble_checkerboard(model_trained, block_size, stop_point, horizontal)
    run_test_intact(model_trained, block_size, stop_point)
    run_test_scrambled(model_trained, block_size, stop_point, horizontal)
    run_test_gray(model_trained, block_size, stop_point, True, horizontal)
    run_test_gray(model_trained, block_size, stop_point, False, horizontal)
    model_trained = torchvision.models.vgg16_bn(True)
    run_test_gray(model_trained, block_size, 0, True, horizontal)
    run_test_gray(model_trained, block_size, 0, False, horizontal)


def run_gray(block_size, horizontal):
    model_trained, train_loss, valid_loss, stop_point = run_train_finetune_gray(block_size, horizontal)
    plot(block_size, train_loss, valid_loss)
    run_test_scramble_checkerboard(model_trained, block_size, stop_point, horizontal)
    run_test_intact(model_trained, block_size, stop_point)
    run_test_scrambled(model_trained, block_size, stop_point, horizontal)
    run_test_gray(model_trained, block_size, stop_point, True, horizontal)
    run_test_gray(model_trained, block_size, stop_point, False, horizontal)
    model_trained = torchvision.models.vgg16_bn(True)
    run_test_gray(model_trained, block_size, 0, True, horizontal)
    run_test_gray(model_trained, block_size, 0, False, horizontal)


def plot(block_size, train_loss, valid_loss):
    fig = plt.figure(figsize=(10, 8))
    plt.plot(range(1, len(train_loss) + 1), train_loss, label='Training Loss')
    plt.plot(range(1, len(valid_loss) + 1), valid_loss, label='Validation Loss')

    # find position of lowest validation loss
    minposs = valid_loss.index(min(valid_loss)) + 1
    plt.axvline(minposs, linestyle='--', color='r', label='Early Stopping Checkpoint')

    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.ylim(0, 0.5)  # consistent scale
    plt.xlim(0, len(train_loss) + 1)  # consistent scale
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    fig.savefig(args.log_training_path + "_" + str(block_size) + '_loss_plot.png', bbox_inches='tight')


if __name__ == "__main__":
    run_gray(int(args.block_size), True)
