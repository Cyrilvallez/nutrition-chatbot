import os

from PIL import Image
import pandas as pd

from helpers import utils

MEAL_DATASET = os.path.join(utils.IMAGE_FOLDER, 'meals')

class ImageDataset(object):
    
    def __init__(self, path: str = MEAL_DATASET):

        self.path = path
        self.image_names = [image for image in os.listdir(self.path) if not image.startswith('.')]
        self.image_paths = [os.path.join(self.path, image) for image in self.image_names]

    def __len__(self) -> int:

        return len(self.image_paths)
    
    def __getitem__(self, key: int | slice) -> Image.Image | list[Image.Image]:

        if isinstance(key, int):
            return Image.open(self.image_paths[key]).convert('RGB'), self.image_names[key]
        
        elif isinstance(key, slice):
            return [Image.open(image).convert('RGB') for image in self.image_paths[key]], self.image_names[key]
        
        else:
            raise ValueError('Cannot slice with this type.')
    
    def __iter__(self):
        """Create a simple generator over the samples.
        """

        for i in range(len(self)):
            yield self[i]


class NutriQuestions(object):

    def __init__(self):

        self.path = os.path.join(utils.ROOT_FOLDER, 'data', 'nutri_questions.xlsx')
        self.data = pd.read_excel(self.path).to_dict(orient='records')

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, key: int | slice) -> dict | list[dict]:
        return self.data[key]
    
    def __iter__(self):
        """Create a simple generator over the samples.
        """

        for i in range(len(self)):
            yield self[i]