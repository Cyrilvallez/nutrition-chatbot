from PIL import Image

import engine
from engine.template import FEW_SHOT_INSTRUCTION

model_name = "idefics-9B-instruct"

model_ = engine.IdeficsModel(model_name)

test_image = Image.open('test_images/fish_chips.jpeg')


baseline = FEW_SHOT_INSTRUCTION
baseline_prompt = ["User:", test_image, baseline + "<end_of_utterance>", "\nAssistant:"]


baseline_output = model_.generate_text(baseline_prompt)
few_shot_output = model_.process_image(test_image)

print(f'Baseline:\n{baseline_output}')
print(f'Few-shot:\n{few_shot_output}')