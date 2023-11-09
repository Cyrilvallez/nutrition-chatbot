import torch
from transformers import IdeficsForVisionText2Text, AutoProcessor, StoppingCriteriaList
from PIL import Image

import config
import template
import stopping

checkpoint = "HuggingFaceM4/idefics-9b"
# checkpoint = "HuggingFaceM4/idefics-80b"
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

processor = AutoProcessor.from_pretrained(checkpoint)
model = IdeficsForVisionText2Text.from_pretrained(checkpoint, torch_dtype=dtype, device_map="auto")

gen_config = config.create_idefics_config(processor, max_new_tokens=500)
prompt_template = template.FewShotIdeficsTemplate()
baseline = template.FEW_SHOT_INSTRUCTION

test_image = Image.open('test_images/fish_chips.jpeg')

few_shot_prompt = prompt_template.get_prompt(test_image)
baseline_prompt = [test_image, baseline]

few_shot_input = processor(few_shot_prompt, return_tensors="pt").to("cuda")
baseline_input = processor(baseline_prompt, return_tensors="pt").to("cuda")

few_shot_input_length = few_shot_input["input_ids"].shape[-1]
baseline_input_length = baseline_input["input_ids"].shape[-1]

few_shot_criteria = StoppingCriteriaList([stopping.TextPatternStopping(few_shot_input_length, processor)])
baseline_criteria = StoppingCriteriaList([stopping.TextPatternStopping(few_shot_input_length, processor)])

few_shot_ids = model.generate(**few_shot_input, generation_config=gen_config, stopping_criteria=few_shot_criteria)
baseline_ids = model.generate(**baseline_input, generation_config=gen_config, stopping_criteria=baseline_criteria)

few_shot_output = few_shot_ids[:, few_shot_input_length:]
baseline_output = baseline_ids[:, baseline_input_length:]

few_shot_output = stopping.post_process_sequences(few_shot_output, processor)
baseline_output = stopping.post_process_sequences(baseline_output, processor)

print(f'Baseline:\n{baseline_output}')
print(f'Few-shot:\n{few_shot_output}')