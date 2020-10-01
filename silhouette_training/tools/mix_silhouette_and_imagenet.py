import os
import shutil
animal_label = {
    "bird": 17,
    "butterfly": 326,
    "cat": 281,
    "crocodile": 49,
    "cow": 345,
    "dog": 207,
    "dolphine": 148,
    "duck": 97,
    "elephant": 385,
    "fish": 0,
    "hen": 8,
    "leopard": 288,
    "monkey": 373,
    "rabbit": 331,
    "rat": 333,
    "spider": 74,
    "tortoise": 37
}

animal_label_list = [val for val in animal_label.values()]
imagenet_dataset = "/Users/leo/PycharmProjects/summerProject/project2/imagenet_val_testing_dataset"

silhouette_dataset = "/Users/leo/PycharmProjects/summerProject/project2/animal_silhouette_training"

imagenet_dir_list = os.listdir(imagenet_dataset)
imagenet_dir_list = sorted(imagenet_dir_list)
animal_dir_list = os.listdir(silhouette_dataset)

for animal_dir in animal_dir_list:
    if animal_dir.startswith("."):
        continue
    id = animal_label[animal_dir]
    print(id)
    print(imagenet_dir_list[id])
    src = silhouette_dataset + "/" + animal_dir + "/"
    animal_image_list = os.listdir(src)
    dst = imagenet_dataset + "/" + imagenet_dir_list[id] + "/"
    for img in animal_image_list:
        if img.endswith(".tif"):
            shutil.copy(src + img, dst)

    # shutil.copy(src, dst)





