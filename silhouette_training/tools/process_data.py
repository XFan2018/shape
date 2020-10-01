import os
import csv

path1 = "D:\\projects\\summerProject2020\\project2\\finetune_silhouette\\log_test_after_finetune_silhouette_low_resolution"
silhouette_category_ids = [17, 326, 281, 49, 345, 207, 148, 97, 385, 0, 8, 288, 373, 331, 333, 74, 37]
filename = "../finetune_silhouette/finetune_all_layers_on_silhouettes_low_resolution"  # csv file name
result = [("category", "epoch0", 'epoch5', 'epoch10', 'epoch15', 'epoch20', 'epoch25', 'epoch30', 'epoch35', 'epoch40', 'epoch45',
           'epoch50', 'epoch55', 'epoch60', 'epoch65', 'epoch70', 'epoch75', 'epoch80')]
with open(os.path.join(
        "D:\\projects\\summerProject2020\project2\\finetune_silhouette\\log_test_before_finetune_silhouette_resize224\\test_after_finetune_model0.txt")) as f:
    lines = f.readlines()
    for line in lines[1:]:
        category_id = int(line.split()[1])
        if category_id in silhouette_category_ids:
            correct = line.split()[3]
            data = (category_id, correct, )
            result.append(data)

with open(filename, 'w', newline='') as f:
    writer = csv.writer(f)
    for data in result:
        writer.writerow(data)

finetune_result = []
dir_list = os.listdir(path1)
dir_list = sorted(dir_list, key=lambda x: int("".join([c for c in x if c.isdigit()])))
for file in dir_list:
    if file.startswith("."):
        continue
    file_id = int("".join([c for c in file if c.isdigit()]))
    if file_id in range(4, 80, 5):
        with open(os.path.join(path1, file)) as r_f:
            print(file)
            lines = r_f.readlines()
            col = []
            for line in lines[1:]:
                category_id = int(line.split()[1])
                if category_id in silhouette_category_ids:
                    col.append(line.split()[3])
            finetune_result.append(col)
print(finetune_result)


for col_id, col in enumerate(finetune_result):
    with open(filename+str(col_id), 'w', newline='') as f:
        writer = csv.writer(f)
        with open(filename, 'r') as csv_f:
            filename = filename+str(col_id)
            reader = csv.reader(csv_f, delimiter=',')
            for idx, row in enumerate(reader):
                print(idx, row)
                if idx == 0:
                    writer.writerow(tuple(row))
                else:
                    print(len(row))
                    row.append(col[idx-1])  # [col_id+2] start to fill the 3rd col, [idx-1] first row is skip for the reader
                    writer.writerow(row)



