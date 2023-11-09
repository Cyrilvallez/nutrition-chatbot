import os

from PIL import Image

# Path to the root of the project
ROOT_FOLDER = os.path.dirname(__file__)

# Path to the image folder
IMAGE_FOLDER = os.path.join(ROOT_FOLDER, 'images')

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