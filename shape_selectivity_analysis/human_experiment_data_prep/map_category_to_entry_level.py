from shape_selectivity_analysis.human_experiment_data_prep.category_mapping import *
from settings import IMAGENET_DIR, IMAGENET_16CAT_DIR
import shutil
import os


'''
map find-grained categories from imagenet to entry-level 16 categories
refer to file 'category_mapping.py'
'''
dir_arr = os.listdir(IMAGENET_DIR)
for dir in dir_arr:
    for cate in mapping_dict.keys():
        if dir in mapping_dict[cate]:
            for img in os.listdir(os.path.join(IMAGENET_DIR, dir)):
                print(img)
                src = os.path.join(IMAGENET_DIR, dir, img)
                dst = os.path.join(IMAGENET_16CAT_DIR, cate)
                shutil.copy(src, dst)


