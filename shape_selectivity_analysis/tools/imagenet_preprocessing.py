import os
import random
import shutil

random.seed(os.getenv("SEED"))


def copy_folders(src, dst):
    """
    :param src: path of the original folder
    :param dst: path to place the copy folder
    """
    arr = os.listdir(src)
    arr = sorted(arr)
    for d in arr:
        if d.startswith("."):
            continue
        print(d)
        os.makedirs(os.path.join(dst, d))


# support method
def create_demo_files(path):
    """
    :param path: place to create 10 demo files
    """
    arr = os.listdir(path)
    os.chdir(path)
    for d in arr:
        if d == ".DS_Store":
            continue
        for i in range(10):
            open(d + "/demo" + d + str(i), "w")


def split_folder(original, testing, ratio):
    """
    :param original: original dataset (original dataset will become training dataset)
    :param testing:  testing dataset
    :param ratio:    ratio of # in testing set to # in original set
    """

    copy_folders(original, testing)
    folders_in_original_dataset = os.listdir(original)
    for folder in folders_in_original_dataset:
        if folder == ".DS_Store":
            continue
        files = os.listdir(os.path.join(original, folder))
        k = int(len(files) * ratio)
        testing_files = random.sample(files, k)
        print(k)
        for file in testing_files:
            os.rename(os.path.join(original, folder, file), os.path.join(testing, folder, file))


def create_subset(original_folder_path, new_folder_path, subset_number):
    copy_folders(original_folder_path, new_folder_path)
    folders_in_original_dataset = os.listdir(original_folder_path)
    for folder in folders_in_original_dataset:
        if folder == ".DS_Store":
            continue
        files = os.listdir(os.path.join(original_folder_path, folder))
        new_files = random.sample(files, subset_number)
        for file in new_files:
            shutil.copy(os.path.join(original_folder_path, folder, file), os.path.join(new_folder_path, folder))


# copy_folders(r"D:\projects\shape_dataset\imagenet_val", r"D:\projects\shape\shape_selectivity_analysis\tools\demo")
# split_folder(r"D:\projects\shape_dataset\animal_dataset", r"D:\projects\shape_dataset\animal_dataset_validation", 0.2)
# create_demo_files(r"D:\projects\shape\shape_selectivity_analysis\tools\demo")


if __name__ == "__main__":
    original_folder_path = r"/home/xingye/train"
    new_folder_path = r"/home/xingye/ImageNet_subset_50000"
    subset_number = 50
    create_subset(original_folder_path, new_folder_path, subset_number)