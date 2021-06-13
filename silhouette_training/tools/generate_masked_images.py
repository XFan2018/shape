import logging
from pathlib import Path
from typing import Any, List

import torch
import torchvision
from PIL import Image
from bagnets import pytorchnet
from torchvision import transforms

delimiter = "\\"
file_path = r"D:\projects\shape_dataset\imagenet_val\n01440764\ILSVRC2012_val_00007197.JPEG"
save_path = r"C:\Users\x44fa\Dropbox\York_summer_project\occluded_images"
logging.basicConfig(format="%(asctime)s — logger %(name)s — %(levelname)s — %(message)s",
                    datefmt='%m/%d/%Y %I:%M:%S %p',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

extension = "png"


class MaskImageGenerator:
    def __init__(self, dataset: str, dst: str, model: Any):
        self.dataset = dataset
        self.dst = dst
        self.switcher = {
            20: {
                "offset": 22,
                "interval": 20 * 2,
                "img_number": 26
            },

            30: {
                "offset": 7,
                "interval": 30 * 2,
                "img_number": 17
            },

            40: {
                "offset": 12,
                "interval": 40 * 2,
                "img_number": 10
            },

            50: {
                "offset": 37,
                "interval": 50 * 2,
                "img_number": 5
            },

            60: {
                "offset": 22,
                "interval": 60 * 2,
                "img_number": 5
            },

            70: {
                "offset": 7,
                "interval": 70 * 2,
                "img_number": 5
            }
        }
        self.isolation = "isolation"
        self.combination = "combination"
        self.model = model
        self.data_transforms = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def mask_patch(self, img: Image.Image, patch_size: int):
        """
        :return:
                 patch size = 20
                 a list of 25 + 1 + 1 = 27 images will be returned (type = <class 'PIL.Image.Image'>)
                 1st "1" is the image masked with all patches simultaneously, 2nd "1" is the original image
        """
        patch_size_list = [20, 30, 40, 50, 60, 70]
        if patch_size not in patch_size_list:
            raise Exception("patch size must be one of 20, 30, 40, 50, 60, 70 numbers")

        list_img = []  # a list of 27 images, see return for details
        list_img_pixel_map = []  # 27 pixel maps for the 27 images
        img = img.resize((224, 224))  # original image is re-sized to 224
        pixel_map = img.load()
        for _ in range(self.switcher[patch_size]["img_number"]):
            # create n images and pixel maps from original images
            new_img = Image.new(img.mode, img.size)
            pixel_map_new = new_img.load()

            # assign pixel values so that they are all identical to the original images
            for i in range(new_img.size[0]):
                for j in range(new_img.size[1]):
                    pixel_map_new[i, j] = pixel_map[i, j]

            # append them to the list
            list_img.append(new_img)
            list_img_pixel_map.append(pixel_map_new)

        # create iterator for the two lists (images and pixel maps)
        iter_img_pixel_map = iter(list_img_pixel_map)

        # generate the masks location
        list_offset_row = range(self.switcher[patch_size]["offset"], 224 - self.switcher[patch_size]["offset"],
                                self.switcher[patch_size]["interval"])
        list_offset_col = range(self.switcher[patch_size]["offset"], 224 - self.switcher[patch_size]["offset"],
                                self.switcher[patch_size]["interval"])
        for offset_row in list_offset_row:
            for offset_col in list_offset_col:

                # image masked with individual patch
                pixel_map_individual_patch = next(iter_img_pixel_map)

                # image masked with all patches simultaneously
                pixel_map_all_patch = list_img_pixel_map[self.switcher[patch_size]["img_number"] - 1]
                for i in range(offset_row, offset_row + patch_size):
                    for j in range(offset_col, offset_col + patch_size):
                        try:
                            pixel_map_individual_patch[i, j] = (0, 0, 0)
                            pixel_map_all_patch[i, j] = (0, 0, 0)
                        except TypeError:
                            pixel_map_individual_patch = (0,)
                            pixel_map_all_patch = (0,)

        # add original image to the end of the list
        list_img.append(img)
        return list_img

    # generate two occluded datasets for left side and right side of the eq, repectively
    # Isolation dataset contains many images with an individual patch for one data
    # Combination dataset contains one image with all patches for one data
    def make_dataset(self, patch_size: int, original=False, combination=False, isolation=False):
        self.copy_dataset_folders(patch_size)
        for idx, img_class_path in enumerate(Path.iterdir(Path(self.dataset))):
            image_class = str(img_class_path).split(delimiter)[-1]
            logger.info(image_class)
            for img_path in Path.iterdir(Path(self.dataset).joinpath(image_class)):
                image_name = str(img_path).split(delimiter)[-1]
                image_name = image_name.replace(".JPEG", "")
                print(idx)
                logger.info(image_name)
                logger.info(img_path)
                img = Image.open(img_path)
                *isolation_images, combination_image, original_image = self.mask_patch(img, patch_size)

                if original:
                    # save original image
                    original_image.save(Path(self.dst).
                                        joinpath(f"original").
                                        joinpath(image_class).
                                        joinpath(image_name + f".{extension}"), format=extension)
                if combination:
                    # save combination image
                    combination_image.save(Path(self.dst).
                                           joinpath(f"{self.combination}_{patch_size}").
                                           joinpath(image_class).
                                           joinpath(image_name + f".{extension}"), format=extension)
                if isolation:
                    # save isolation images
                    for i, isolation_image in enumerate(isolation_images):
                        isolation_image.save(Path(self.dst).
                                             joinpath(f"{self.isolation}_{patch_size}").
                                             joinpath(image_class).
                                             joinpath(f"{image_name}_{i}.{extension}"), format=extension)

    # copy class folders of the dataset to another path
    def copy_dataset_folders(self, patch_size: int):
        for d in Path.iterdir(Path(self.dataset)):
            class_dir = str(d).split(delimiter)[-1]
            if class_dir.startswith("."):
                continue
            logger.info(class_dir)
            try:
                Path(self.dst).joinpath(f"{self.isolation}_{patch_size}").joinpath(class_dir).mkdir(mode=644,
                                                                                                    parents=True)
            except FileExistsError:
                logger.info("file already exists")

            try:
                Path(self.dst).joinpath(f"{self.combination}_{patch_size}").joinpath(class_dir).mkdir(mode=644,
                                                                                                      parents=True)

            except FileExistsError:
                logger.info("file already exists")

            try:
                Path(self.dst).joinpath(f"original").joinpath(class_dir).mkdir(mode=644,
                                                                               parents=True)
            except FileExistsError:
                logger.info("file already exists")

    # calculate LHS: mean logits of images that are manipulated by an individual patch
    def calculate_lhs(self, patch_size: int) -> List:
        result = []
        dataset = Path(self.dst).joinpath(f"{self.isolation}_{patch_size}")
        data = torchvision.datasets.ImageFolder(dataset, self.data_transforms)
        batch_size = self.switcher[patch_size]["img_number"] - 1
        dataloader = torch.utils.data.DataLoader(data,
                                                 batch_size=batch_size,
                                                 shuffle=False,
                                                 num_workers=2)
        self.model = self.model.to(self.device)
        self.model.eval()
        for i, (image, label) in enumerate(dataloader):
            image = image.to(self.device)
            label = label.to(self.device)
            with torch.set_grad_enabled(False):
                outputs = self.model(image)
                val = outputs[torch.arange(batch_size), label]
                mean_val = torch.mean(val)
                result.append(mean_val.item())
                logger.info(f"iter: {i}, mean_val: {mean_val}, label: {label}")
        return result

    # calculate RHS: logits of the images that are manipulated by all patches at once
    def calculate_rhs(self, patch_size: int) -> List:
        result = []
        dataset = Path(self.dst).joinpath(f"{self.combination}_{patch_size}")
        data = torchvision.datasets.ImageFolder(dataset, self.data_transforms)
        dataloader = torch.utils.data.DataLoader(data,
                                                 shuffle=False,  # batch size default is 1
                                                 num_workers=2)
        self.model = self.model.to(self.device)
        self.model.eval()
        for i, (image, label) in enumerate(dataloader):
            image = image.to(self.device)
            label = label.to(self.device)
            with torch.set_grad_enabled(False):
                outputs = self.model(image)
                val = outputs[torch.tensor(0), label]
                val = torch.squeeze(val)
                result.append(val.item())
                logger.info(f"iter: {i}, val: {val}, label: {label}")
        return result

    def calculate_org_logit(self) -> List:
        result = []
        dataset = Path(self.dst).joinpath("original")
        data = torchvision.datasets.ImageFolder(dataset, self.data_transforms)
        dataloader = torch.utils.data.DataLoader(data,
                                                 shuffle=False,  # batch size default is 1
                                                 num_workers=2)
        self.model = self.model.to(self.device)
        self.model.eval()
        for i, (image, label) in enumerate(dataloader):
            image = image.to(self.device)
            label = label.to(self.device)
            with torch.set_grad_enabled(False):
                outputs = self.model(image)
                val = outputs[torch.tensor(0), label]
                val = torch.squeeze(val)
                result.append(val.item())
                logger.info(f"iter: {i}, val: {val}, label: {label}")
        return result

    def pickle_data(self, data: List, file_name: str):
        import pickle
        with open(file_name, 'wb') as f:
            pickle.dump(data, f)

    def load_data(self, file_name: str) -> List:
        import pickle
        with open(file_name, 'rb') as f:
            return pickle.load(f)


MODEL = pytorchnet.bagnet33(pretrained=True)
DATASET = r"D:\projects\shape_dataset\imagenet_val_testing_dataset_Copy"
DESTINATION = r"C:\Users\x44fa\Dropbox\York_summer_project\occluded_images"


def get_corrcoef(patch_size: int, generate_org=False):
    generator = MaskImageGenerator(DATASET, DESTINATION, MODEL)
    generator.make_dataset(patch_size, combination=True, isolation=True)
    rhs = generator.calculate_rhs(patch_size)
    generator.pickle_data(rhs, f'rhs_{patch_size}.pkl')
    lhs = generator.calculate_lhs(patch_size)
    generator.pickle_data(lhs, f'lhs_{patch_size}.pkl')
    if generate_org:
        org = generator.calculate_org_logit()
        generator.pickle_data(org, f'org.pkl')



if __name__ == "__main__":
# import numpy as np
#
# generator = MaskImageGenerator(DATASET, DESTINATION, MODEL)
# rhs = generator.load_data('rhs_new2.pkl')
# lhs = generator.load_data('lhs_new2.pkl')
# org = generator.load_data('org_new2.pkl')
# rhs = np.array(rhs)
# lhs = np.array(lhs)
# org = np.array(org)
# RHS = rhs - org
# LHS = lhs - org
# pass
    for patch_size in [20, 30, 40, 50, 70]:
        get_corrcoef(patch_size)
