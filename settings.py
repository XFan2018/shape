import os

from dotenv import load_dotenv
from pathlib import Path
import logging
from logging import handlers
import sys
load_dotenv(verbose=True)
env_path = rf"{os.path.split(__file__)[0]}\.env"
print(env_path)
load_dotenv(dotenv_path=env_path)
shape_path = os.path.dirname(os.path.realpath(__file__))
prefix = "/Users/leo/Dropbox/shape_dataset/"

logger = logging.getLogger("mylogger")
logger.setLevel(logging.DEBUG)
user = os.getenv("USER")
pwd = os.getenv("PASSWORD")
print(type(user), user, pwd)
em_handler = handlers.SMTPHandler(
    mailhost=("smtp.gmail.com", 587),
    fromaddr="kakashikatake98@gmail.com",
    toaddrs=["kakashikatake98@gmail.com"],
    subject="Shape Project New Logging Message",
    credentials=(user, pwd),
    secure=()
)
em_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
em_handler.setLevel(logging.DEBUG)
logger.addHandler(em_handler)

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