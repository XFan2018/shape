import os
import random

def copy_folders(src, dst):
    """
    :param src: path of the folder to be copied
    :param dst: path to place the copy folder
    """
    arr = os.listdir(src)
    arr = sorted(arr)
    for d in arr:
        if d.startswith("."):
            continue
        print(d)
        os.makedirs(dst + "\\" + d)


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
        files = os.listdir(original + "\\" + folder)
        k = int(len(files) * ratio)
        testing_files = random.sample(files, k)
        print(k)
        for file in testing_files:
            os.rename(original + "\\" + folder + "\\" + file, testing + "\\" + folder + "\\" + file)




split_folder("D:\\projects\\summerProject2020\\project3\\animal_silhouette_training", "D:\\projects\\summerProject2020\\project3\\animal_silhouette_validation", 0.25)