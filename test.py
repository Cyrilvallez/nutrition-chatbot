import torch
from transformers import IdeficsForVisionText2Text, AutoProcessor, StoppingCriteriaList
from PIL import Image

import model
import template

model_name = "idefics-9B"

model_ = model.IdeficsModel(model_name)


test_image = Image.open('test_images/fish_chips.jpeg')
baseline = template.FEW_SHOT_INSTRUCTION
baseline_prompt = [test_image, baseline]

baseline_output = model_.generate_text(baseline_prompt)
few_shot_output = model_.process_image(test_image)

print(f'Baseline:\n{baseline_output}')
print(f'Few-shot:\n{few_shot_output}')