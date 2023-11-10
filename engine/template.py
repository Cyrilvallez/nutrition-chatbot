import os
import uuid
import copy

from PIL import Image

from helpers import utils

# Path to the image folder
IMAGE_FOLDER = os.path.join(utils.ROOT_FOLDER, 'few_shot_images')

FEW_SHOT_IMAGES = (
    Image.open(os.path.join(IMAGE_FOLDER, 'burger.jpeg')),
    Image.open(os.path.join(IMAGE_FOLDER, 'pizza.jpeg')),
    Image.open(os.path.join(IMAGE_FOLDER, 'mountain.jpeg')),
    Image.open(os.path.join(IMAGE_FOLDER, 'fondue.jpeg')),
    Image.open(os.path.join(IMAGE_FOLDER, 'tahiti.jpeg')),
    Image.open(os.path.join(IMAGE_FOLDER, 'rice.jpeg')),
)

FEW_SHOT_INSTRUCTION = ("Does the image represent human food? Only answer by 'Yes' or 'No'. If your answer is "
                        "'Yes', give the name of the meal and describe in details its ingredients. Given the  "
                        "meal name and the ingredients, estimate how many calories the meal represent.")

FEW_SHOT_RESPONSES = (
    ("Yes.\nThe meal on the image is a hamburger.\nThe ingredients are: bun, lettuce, tomatoes, onions, steak."
     "\nThe estimated amount of calories for this hamburger is 500-800 kcal."),

    ("Yes.\nThe meal on the image is a pizza.\nThe ingredients are: pizza dough, tomatoes, basilic, cheese, " 
     "tomato sauce.\nThe estimated amount of calories for this pizza is 700-1000 kcal."),

    'No',

    ("Yes.\nThe meal on the image is a swiss fondue.\nThe ingredients are: melted cheese, bread.\nThe estimated "
     "amount of calories for a portion of fondue is 800-1100 kcal."),

    'No',

    ("Yes.\nThe meal on the image is a bowl of rice with tofu and vegetables.\nThe ingredients are: rice, tofu "
     "bell peppers, and mushrooms.\nThe estimated amount of calories for this meal is 200-500 kcal."),
)


LLAMA2_DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. "
    "Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. "
    "Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make "
    "any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't "
    "know the answer to a question, please don't share false information."
)


LLAMA2_NUTRITION_SYSTEM_PROMPT = (
    "You are a helpful, respectful, honest and world-renowned nutritionist assistant. You provide advices to people on food "
    "and nutrition, and how they impact health. You try to improve their lives by providing healthy food solutions "
    "and recipes. If someone tells you about a specific health condition such as diabetes, obesity, underweight, or "
    "food allergy, you always provide adequate advice. Your answers should not include any harmful, unethical, "
    "racist, sexist, toxic, dangerous, or illegal content.\n\nIf a question does not make any sense, or is not "
    "factually coherent, explain why instead of answering something not correct. If you don't know the answer to "
    "a question, please don't share false information or advice."
)



class FewShotIdeficsTemplate(object):

    def __init__(self, shots: int | None = None, instruct: bool = False,
                 images: list[str] | list[Image.Image] | None = FEW_SHOT_IMAGES, instruction: str = FEW_SHOT_INSTRUCTION,
                 responses: list[str] | None = FEW_SHOT_RESPONSES):

        self.images = images if images is not None else []
        self.few_shot_responses = responses if responses is not None else []

        if len(self.images) != len(self.few_shot_responses):
            raise ValueError('The number of few shot images must match the number of few shot description of these images.')

        if shots is None:
            self.shots = len(self.images)
        else:
            self.shots = shots if shots <= len(self.images) else len(self.images)

        self.instruct = instruct
        self.eou_token = "<end_of_utterance>"
        self.instruction = instruction

    def get_prompt(self, image: str | Image.Image) -> list[str | Image.Image]:
        """Format the prompt with few-shot examples.

        Parameters
        ----------
        image : str | PIL.Image
            Image input to the model.

        Returns
        -------
        list[str | Image.Image]
            The formatted prompt.
        """

        prompt = []
        for i in range(self.shots + 1):
            if i == 0:
                prompt.append('User:')
            else:
                prompt.append('\nUser:')
            # few shot examples
            if i < self.shots:
                prompt.append(self.images[i])
                if self.instruct:
                    text = self.instruction + self.eou_token + '\nAssistant: ' + self.few_shot_responses[i] + self.eou_token
                else:
                    text = self.instruction + '\nAssistant: ' + self.few_shot_responses[i]
            # actual user image prompt
            else:
                prompt.append(image)
                if self.instruct:
                    text = self.instruction + self.eou_token + '\nAssistant:'
                else:
                    text = self.instruction + '\nAssistant:'

            prompt.append(text)

        return prompt
    



class GenericConversationTemplate(object):
    """Class used to store a conversation with a model."""

    def __init__(self, eos_token: str, system_prompt: str = ''):

        # Conversation history
        self.user_history_text = []
        self.model_history_text = []

        # system prompt
        self.system_prompt = system_prompt

        # eos token
        self.eos_token = eos_token

        # Extra eos tokens
        self.extra_eos_tokens = []

        # Some templates need an additional space when using `get_last_turn_continuation_prompt`
        self.add_space_to_continuation_prompt = False

        # create unique identifier (used in gradio flagging)
        self.id = str(uuid.uuid4())


    def __len__(self) -> int:
        """Return the length of the current conversation.
        """
        return len(self.user_history_text)
    
    
    def __iter__(self):
        """Create a generator over (user_input, model_answer) tuples for all turns in the conversation.
        """
        # Generate over copies so that the object in the class cannot change during iteration
        for user_history, model_history in zip(self.user_history_text.copy(), self.model_history_text.copy()):
            yield user_history, model_history
    

    def __str__(self) -> str:
        """Format the conversation as a string.
        """

        N = len(self)

        if N == 0:
            return "The conversation is empty."
        
        else:
            out = ''
            for i, (user, model) in enumerate(self):
                out += f'>> User: {user}\n'
                if model is not None:
                    out += f'>> Model: {model}'
                # Skip 2 lines between turns
                if i < N - 1:
                    out += '\n\n'

            return out
        

    def set_system_prompt(self, system_prompt: str):
        self.system_prompt = system_prompt
        

    def append_user_message(self, user_prompt: str):
        """Append a new user message, and set the corresponding answer of the model to `None`.

        Parameters
        ----------
        user_prompt : str
            The user message.
        """

        if None in self.model_history_text:
            raise ValueError('Cannot append a new user message before the model answered to the previous messages.')

        self.user_history_text.append(user_prompt)
        self.model_history_text.append(None)


    def append_model_message(self, model_output: str):
        """Append a new model message, by modifying the last `None` value in-place. Should always be called after
        `append_user_message`, before a new call to `append_user_message`.

        Parameters
        ----------
        model_output : str
            The model message.
        """

        if self.model_history_text[-1] is None:
            self.model_history_text[-1] = model_output
        else:
            raise ValueError('It looks like the last user message was already answered by the model.')
        

    def append_to_last_model_message(self, model_output: str):
        """Append text to the last model message, in case the `max_new_tokens` was set to a value too low
        to finish the model answer.

        Parameters
        ----------
        model_output : str
            The model message.
        """

        if self.model_history_text[-1] is None:
            raise ValueError('The last user message was never answered. You should use `append_model_message`.')
        else:
            self.model_history_text[-1] += model_output
        

    def get_prompt(self) -> str:
        """Format the prompt representing the conversation that we will feed to the tokenizer.
        """

        # This seems to be the accepted way to treat inputs for conversation with a model that was not specifically
        # fine-tuned for conversation. This is the DialoGPT way of handling conversation, but is in fact reused by
        # all other tokenizers that we use.

        prompt = ''

        for user_message, model_response in self:

            prompt += user_message + self.eos_token
            if model_response is not None:
                prompt += model_response + self.eos_token

        return prompt
    

    def get_last_turn_continuation_prompt(self) -> str:
        """Format the prompt to feed to the model in order to continue the last turn of the model output, in case
        `max_new_tokens` was set to a low value and the model did not finish its output.
        """

        if len(self) == 0:
            raise RuntimeError('Cannot continue the last turn on an empty conversation.')
    
        if self.model_history_text[-1] is None:
            raise RuntimeError('Cannot continue an empty last turn.')
        
        # Use a copy since we will modify the last model turn
        conv_copy = copy.deepcopy(self)
        last_model_output = conv_copy.model_history_text[-1]
        # Set it to None to mimic the behavior of an un
        conv_copy.model_history_text[-1] = None

        # Get prompt of conversation without the last model turn
        prompt = conv_copy.get_prompt()
        # Reattach last turn, with or without additional space
        if self.add_space_to_continuation_prompt:
            prompt += ' ' + last_model_output
        else:
            prompt += last_model_output

        return prompt
    

    def get_extra_eos(self) -> list[str]:
        return self.extra_eos_tokens
    

    def erase_conversation(self):
        """Reinitialize the conversation.
        """

        self.user_history_text = []
        self.model_history_text = []

    
    def set_conversation(self, past_user_inputs: list[str], past_model_outputs: list[str]):
        """Set the conversation.
        """

        self.user_history_text = past_user_inputs
        self.model_history_text = past_model_outputs


    def to_gradio_format(self) -> list[list[str, str]]:
        """Convert the current conversation to gradio chatbot format.
        """

        if len(self) == 0:
            return [[None, None]]

        return [list(conv_turn) for conv_turn in self]
        


# reference: https://github.com/facebookresearch/llama/blob/1a240688810f8036049e8da36b073f63d2ac552c/llama/generation.py#L212
class Llama2ChatConversationTemplate(GenericConversationTemplate):

    def __init__(self, eos_token: str = '</s>', system_prompt: str = LLAMA2_NUTRITION_SYSTEM_PROMPT):

        super().__init__(eos_token, system_prompt)

        # Override value
        self.add_space_to_continuation_prompt = True

        self.bos_token = '<s>'

        self.system_template = '<<SYS>>\n{system_prompt}\n<</SYS>>\n\n'

        self.user_token = '[INST]'
        self.assistant_token = '[/INST]'


    def get_prompt(self) -> str:
        """Format the prompt representing the conversation that we will feed to the tokenizer.
        """

        # If we are not using system prompt, do not add the template formatting with empty prompt
        if self.system_prompt.strip() != '':
            system_prompt = self.system_template.format(system_prompt=self.system_prompt.strip())
        else:
            system_prompt = ''
        prompt = ''

        for i, (user_message, model_response) in enumerate(self):

            if i == 0:
                # Do not add bos_token here as it will be added automatically at the start of the prompt by 
                # the tokenizer 
                prompt += self.user_token + ' ' + system_prompt + user_message.strip() + ' '
            else:
                prompt += self.bos_token + self.user_token + ' ' + user_message.strip() + ' '
            if model_response is not None:
                prompt += self.assistant_token + ' ' + model_response.strip() + ' ' + self.eos_token
            else:
                prompt += self.assistant_token

        return prompt