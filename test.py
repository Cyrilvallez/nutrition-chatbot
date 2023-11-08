import torch
from transformers import IdeficsForVisionText2Text, AutoProcessor

import config

checkpoint = "HuggingFaceM4/idefics-9b"

processor = AutoProcessor.from_pretrained(checkpoint)
dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
model = IdeficsForVisionText2Text.from_pretrained(checkpoint, torch_dtype=dtype, device_map="auto")

gen_config = config.create_idefics_config(processor, max_new_tokens=100)