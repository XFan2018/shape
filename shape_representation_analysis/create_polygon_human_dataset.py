from pathlib import Path
from .sparse_coding import im2poly
from PIL import Image

# windows platform
delimiter = "\\"

class CreatePolygonHumanDataset:
    def __init__(self, dataset_path: str, ext: str):
        self.dataset = dataset_path
        self.ext = ext

    def draw_polygon_image(self):
        """
        Create another folder besides the original dataset path named <original dataset name>_human
        There are folders for each category which contains the images for human experiment generated from
        the shape dataset (e.g. Animal Dataset)
        """
        p = Path(self.dataset)
        human_dataset_path = Path(self.dataset + "_human")
        for category_path in p.iterdir():
            # get the name of each category
            category = str(category_path).split(delimiter)[-1]
            # generate category path of human dataset
            category_path_human = human_dataset_path.joinpath(category)
            category_path_human.mkdir(mode=644, parents=True, exist_ok=True)

            # generate images for human dataset
            




if __name__ == '__main__':
    create_dataset = CreatePolygonHumanDataset(r"C:\Users\x44fa\Dropbox\shape_dataset\animal_dataset", "tif")
    create_dataset.draw_polygon_image()
