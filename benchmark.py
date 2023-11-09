import os

from engine import IdeficsModel
from helpers import datasets
from helpers import utils

image_dataset = datasets.ImageDataset()

model_name = 'idefics-9B'
model = IdeficsModel(model_name)

completions = {}
for img, name in image_dataset:
    completions[name] = model.process_image(img)

utils.save_json(completions, f'meal_dataset_{model_name}.json')