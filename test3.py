from engine import Llama2ChatModel
from engine.template import LLAMA2_DEFAULT_SYSTEM_PROMPT, LLAMA2_NUTRITION_SYSTEM_PROMPT

model_name = 'llama2-13B-chat'
model = Llama2ChatModel(model_name)

conversation_default = model.get_empty_conversation(system_prompt=LLAMA2_DEFAULT_SYSTEM_PROMPT)
conversation_nutrition = model.get_empty_conversation(system_prompt=LLAMA2_NUTRITION_SYSTEM_PROMPT)

prompt1 = "Hello! Who are you?"
prompt2 = "I am in a bit of overweight... Could you help me find a healthy meal for tonight?"

model.generate_conversation(prompt1, conv_history=conversation_default)
model.generate_conversation(prompt1, conv_history=conversation_nutrition)

model.generate_conversation(prompt2, conv_history=conversation_default)
model.generate_conversation(prompt2, conv_history=conversation_nutrition)

print(f'Default:\n{str(conversation_default)}')
print('\n\n\n\n')
print(f'Nutrition:\n{str(conversation_nutrition)}')