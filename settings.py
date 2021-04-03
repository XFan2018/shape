import os

from dotenv import load_dotenv
from pathlib import Path
import logging
from logging import handlers

load_dotenv(verbose=True)
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)
shape_path = os.path.dirname(os.path.realpath(__file__))
prefix = "/Users/leo/Dropbox/shape_dataset/"
IMAGENET_DIR = fr"{prefix}imagenet_val"
IMAGENET_16CAT_DIR = fr"{prefix}imagenet_16categories"
CHECKERBOARD_PREP = rf"{prefix}checkerboard_prep_new"
CHECKERBOARD_DATASET_HUMAN = rf"{prefix}checkerboard_dataset_human_png"
CHECKERBOARD_DATASET_HUMAN_LATTICE_BLACK = fr"{prefix}checkerboard_lattice_dataset_human_black_png"
CHECKERBOARD_DATASET_HUMAN_LATTICE_GRAY = fr"{prefix}checkerboard_lattice_dataset_human_gray_png"
CHECKERBOARD_GRAY_DATASET_HUMAN = fr"{prefix}checkerboard_gray_dataset_human_png"
INTACT_DATASET_HUMAN = fr"{prefix}intact_dataset_human_png"
JUMBLED_DATASET_HUMAN = fr"{prefix}jumbled_dataset_human_png"

logger = logging.getLogger("mylogger")
logger.setLevel(logging.DEBUG)
em_handler = handlers.SMTPHandler(
    mailhost="smtp.gmail.com",
    fromaddr="xingyecoin@gmail.com",
    toaddrs="xingye@my.yorku.ca",
    subject="Shape Project New Logging Message",
    credentials=("xingye", "feng5830648"),
    secure=None
)
em_handler.setLevel(logging.DEBUG)
logger.addHandler(em_handler)