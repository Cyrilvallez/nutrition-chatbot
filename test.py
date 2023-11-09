import torch
from transformers import IdeficsForVisionText2Text, AutoProcessor, StoppingCriteriaList
from PIL import Image

import engine
from engine.template import FEW_SHOT_INSTRUCTION

model_name = "idefics-9B"

model_ = engine.IdeficsModel(model_name)


test_image = Image.open('test_images/fish_chips.jpeg')
baseline = FEW_SHOT_INSTRUCTION
baseline_prompt = [test_image, baseline]

baseline_output = model_.generate_text(baseline_prompt)
few_shot_output = model_.process_image(test_image)

print(f'Baseline:\n{baseline_output}')
print(f'Few-shot:\n{few_shot_output}')