import os

from engine import IdeficsModel
from helpers import datasets

image_dataset = datasets.ImageDataset()

model_name = 'idefics-9B'
model = IdeficsModel(model_name)

completions = {}
for img, name in image_dataset:
    completions[name] = model.process_image(img)