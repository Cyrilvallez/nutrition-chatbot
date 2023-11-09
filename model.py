from PIL import Image

import torch

import loader
import template
import config
import stopping


class IdeficsModel(object):

    def __init__(self, model_name, quantization_8bits: bool = False, quantization_4bits: bool = False,
                 dtype: torch.dtype | None = None, max_fraction_gpu_0: float = 0.8, max_fraction_gpus: float = 0.8,
                 device_map: dict | str | None = None, gpu_rank: int = 0):
        
        self.model, self.processor = loader.load_model_and_processor(model_name, quantization_8bits=quantization_8bits,
                                                                     quantization_4bits=quantization_4bits, dtype=dtype,
                                                                     max_fraction_gpu_0=max_fraction_gpu_0,
                                                                     max_fraction_gpus=max_fraction_gpus,
                                                                     device_map=device_map, gpu_rank=gpu_rank)
        
        self.model_name = model_name
        self.quantization_8bits = quantization_8bits
        self.quantization_4bits = quantization_4bits
        # May be different from the dtype given in the arguments so use the model attribute
        self.dtype = self.model.dtype
        
        self.is_instruct = self.model_name.rsplit('-', 1)[1] == 'instruct'


    def generate_text(
            self,
            prompt: list[str | Image.Image],
            max_new_tokens: int = 512,
            min_new_tokens: int | None = None,
            do_sample: bool = False,
            top_k: int | None = 50,
            top_p: float | None = 0.9,
            temperature: float = 0.8,
            stopping_patterns: list[str] | tuple[str] | None = stopping.IDEFICS_STOP_PATTERNS,
            truncate_prompt_from_output: bool = True,
            post_process_output: bool = True,
            **kwargs
        ) -> str:
        """Generate a single text completion based on `prompt`.

        Prompt formatting parameters
        ----------
        prompt : list[str  |  Image.Image]
            The prompt to the model.

        Generation parameters
        ---------------------
        max_new_tokens : int, optional
            Maximum number of new tokens to generate, by default 512
        min_new_tokens : int | None, optional
            The minimum number of tokens to generate, by setting the probability of EOS token to 0. Giving `None`
            is the same as giving 0. By default `None`.
        do_sample : bool, optional
            Whether to perform sampling or greedy search generation, by default False, i.e. greedy search.
        top_k : int | None, optional
            How many tokens with max probability to consider for random sampling, by default 50. Not used if 
            `do_sample=False`. You can deactivate top_k sampling by providing `top_k=0` or `top_k=None`. Note 
            that if you provide both `top_k` and `top_p`, the `top_k` is applied before. By default 50
        top_p : float | None, optional
            The probability density covering the new tokens to consider for random sampling, by default 0.9. Not used if 
            `do_sample=False`. You can deactivate top_p sampling by providing `top_p=1` or `top_p=None`. Note 
            that if you provide both `top_k` and `top_p`, the `top_k` is applied before. By default 0.90
        temperature : float, optional
            How to cool down the probability distribution. Value between 1 (no cooldown) and 0 (greedy search,
            no randomness), by default 0.8. Passing 0 is equivalent to setting `do_sample=False`.
        stopping_patterns : list[str] | tuple[str] | None, optional
            The list of patterns to use to stop generation, by default stopping.IDEFICS_STOP_PATTERNS

        Output formatting parameters
        ----------------------------
        truncate_prompt_from_output : bool, optional
            Whether to remove the prompt from the model answer or not, by default True.
        post_process_output : bool, optional
            Whether to post-process the outputs, i.e. truncate according to the `stopping_patterns`. By default True.

        Returns
        -------
        str
            The generated text sequence.
        """

        generation_config = config.create_idefics_config(self.processor, max_new_tokens=max_new_tokens,
                                                         min_new_tokens=min_new_tokens, do_sample=do_sample,
                                                         top_k=top_k, top_p=top_p, temperature=temperature)
        
        input = self.processor(prompt, return_tensors='pt')
        input_length = input['input_ids'].shape[-1]
        if torch.cuda.is_available():
            input = input.to('cuda')

        stopping_criteria = stopping.create_stopping_criteria(input_length, self.processor, stopping_patterns)

        output = self.model.generate(**input, generation_config=generation_config, stopping_criteria=stopping_criteria,
                                     **kwargs)
        truncated_output = output[:, input_length:]

        if post_process_output:
            generated_text = stopping.post_process_sequences(truncated_output, self.processor, stopping_patterns)
        else:
            generated_text = self.processor.batch_decode(truncated_output, skip_special_tokens=True)

        if not truncate_prompt_from_output:
            generated_text = self.processor.batch_decode(output[:, 0:input_length], skip_special_tokens=True) \
                + generated_text
            
        return generated_text[0]



    def process_image(
            self,
            image: str | Image.Image,
            shots: int | None = None,
            few_shot_images: list[str | Image.Image] | None = template.FEW_SHOT_IMAGES,
            few_shot_instruction: str | None = template.FEW_SHOT_INSTRUCTION,
            few_shot_answers: list[str] | None = template.FEW_SHOT_RESPONSES,
            max_new_tokens: int = 512,
            min_new_tokens: int | None = None,
            do_sample: bool = False,
            top_k: int | None = 50,
            top_p: float | None = 0.9,
            temperature: float = 0.8,
            stopping_patterns: list[str] | tuple[str] | None = stopping.IDEFICS_STOP_PATTERNS,
            truncate_prompt_from_output: bool = True,
            post_process_output: bool = True,
            **kwargs
        ) -> str:
        """Generate a single text completion based on a new `image` and few-shot examples.

        Prompt formatting parameters
        ----------
        image : str | Image.Image
            The image to process (describe).
        shots : int | None, optional
            The number of few-shot examples to use. If `None`, will use the maximum number of available examples.
            By default `None`.
        few_shot_images : list[str  |  Image.Image] | None, optional
            The images to use in the few-shot examples, by default template.FEW_SHOT_IMAGES
        few_shot_instruction : str | None, optional
            The instruction to repeat for each image in the few-shot examples, by default template.FEW_SHOT_INSTRUCTION
        few_shot_answers : list[str] | None, optional
            The expected output to use in the few-shot examples, by default template.FEW_SHOT_RESPONSES

        Generation parameters
        ---------------------
        max_new_tokens : int, optional
            Maximum number of new tokens to generate, by default 512
        min_new_tokens : int | None, optional
            The minimum number of tokens to generate, by setting the probability of EOS token to 0. Giving `None`
            is the same as giving 0. By default `None`.
        do_sample : bool, optional
            Whether to perform sampling or greedy search generation, by default False, i.e. greedy search.
        top_k : int | None, optional
            How many tokens with max probability to consider for random sampling, by default 50. Not used if 
            `do_sample=False`. You can deactivate top_k sampling by providing `top_k=0` or `top_k=None`. Note 
            that if you provide both `top_k` and `top_p`, the `top_k` is applied before. By default 50
        top_p : float | None, optional
            The probability density covering the new tokens to consider for random sampling, by default 0.9. Not used if 
            `do_sample=False`. You can deactivate top_p sampling by providing `top_p=1` or `top_p=None`. Note 
            that if you provide both `top_k` and `top_p`, the `top_k` is applied before. By default 0.90
        temperature : float, optional
            How to cool down the probability distribution. Value between 1 (no cooldown) and 0 (greedy search,
            no randomness), by default 0.8. Passing 0 is equivalent to setting `do_sample=False`.
        stopping_patterns : list[str] | tuple[str] | None, optional
            The list of patterns to use to stop generation, by default stopping.IDEFICS_STOP_PATTERNS

        Output formatting parameters
        ----------------------------
        truncate_prompt_from_output : bool, optional
            Whether to remove the prompt from the model answer or not, by default True.
        post_process_output : bool, optional
            Whether to post-process the outputs, i.e. truncate according to the `stopping_patterns`. By default True.

        Returns
        -------
        str
            The description of the image.
        """

        few_shot_template = template.FewShotIdeficsTemplate(shots=shots, instruct=self.is_instruct, images=few_shot_images,
                                                            instruction=few_shot_instruction, responses=few_shot_answers)
        
        prompt = few_shot_template.get_prompt(image)

        return self.generate_text(prompt, max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens,do_sample=do_sample,
                                  top_k=top_k, top_p=top_p, temperature=temperature, stopping_patterns=stopping_patterns,
                                  truncate_prompt_from_output=truncate_prompt_from_output, post_process_output=post_process_output,
                                  **kwargs)