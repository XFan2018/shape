import os

from dotenv import load_dotenv
from pathlib import Path
load_dotenv(verbose=True)
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)
shape_path = os.path.dirname(os.path.realpath(__file__))
prefix = "/Users/leo/Dropbox/shape_dataset/"


IMAGENET_DIR = r"D:\projects\shape_dataset\imagenet_val"
IMAGENET_16CAT_DIR = r"D:\projects\shape_dataset\imagenet_16categories"
CHECKERBOARD_PREP = r"D:\projects\shape_dataset\checkerboard_prep_new"


CHECKERBOARD_DATASET_HUMAN = r"D:\projects\shape_dataset\checkerboard_dataset_human_png"
CHECKERBOARD_DATASET_HUMAN_LATTICE_BLACK = r"D:\projects\shape_dataset\checkerboard_lattice_dataset_human_black"
CHECKERBOARD_DATASET_HUMAN_LATTICE_GRAY = r"D:\projects\shape_dataset\checkerboard_lattice_dataset_human_gray"
CHECKERBOARD_GRAY_DATASET_HUMAN = r"D:\projects\shape_dataset\checkerboard_gray_dataset_human_png"
CHECKERBOARD_GRAY_JUMBLED_DATASET_HUMAN = r"D:\projects\shape_dataset\checkerboard_jumbled_gray_dataset_human_png"
INTACT_DATASET_HUMAN = r"D:\projects\shape_dataset\intact_dataset_human_png"
JUMBLED_DATASET_HUMAN = r"D:\projects\shape_dataset\jumbled_dataset_human_png"
