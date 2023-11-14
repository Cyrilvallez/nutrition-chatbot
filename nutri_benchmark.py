import argparse

from engine import ChatModel
from helpers import datasets
from helpers import utils


def main(model_name):

    dataset = datasets.NutriQuestions()
    model = ChatModel(model_name)

    completions = []
    for sample in dataset:
        conv = model.get_empty_conversation()
        prompt = sample['Questions']
        answer = model.generate_conversation(prompt, conv_history=conv, max_new_tokens=1024)
        answer = answer.model_history_text[-1]
        completions.append({**sample, 'answer': answer})

    utils.save_jsonl(completions, f'nutri_questions_{model_name}.jsonl')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Nutri Questions')
    parser.add_argument('model', type=str, help='The model to run.')
    args = parser.parse_args()

    main(args.model)
