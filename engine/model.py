import copy
import abc

import torch
from PIL import Image

from engine import loader
from engine import template
from engine import config
from engine import stopping
from helpers import utils


class IdeficsModel(object):

    def __init__(self, model_name, quantization_8bits: bool = False, quantization_4bits: bool = False,
                 dtype: torch.dtype | None = None, max_fraction_gpu_0: float = 0.8, max_fraction_gpus: float = 0.8,
                 device_map: dict | str | None = None, gpu_rank: int = 0):
        
        if model_name not in loader.IDEFICS_MODELS_MAPPING.keys():
            raise ValueError('You did not provide a valid idefics model name.')
        
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

        # In this case, the model is on multiple devices
        if hasattr(self.model, 'hf_device_map'):
            gpu_devices = set(self.model.hf_device_map.values())
            gpu_devices.discard('cpu')
            gpu_devices.discard('disk')
            
            self.input_device = min(gpu_devices) if len(gpu_devices) > 0 else 'cpu'
            
        # In this case, the model is on a single device
        else:
            device = next(self.model.parameters()).get_device()
            self.input_device = 'cpu' if device == -1 else device


    def generate_text(
            self,
            prompt: list[str | Image.Image],
            max_new_tokens: int = 512,
            min_new_tokens: int | None = None,
            do_sample: bool = False,
            top_k: int | None = 50,
            top_p: float | None = 0.9,
            temperature: float = 0.8,
            seed: int | None = None,
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
        seed : int | None, optional
            An optional seed to force the generation to be reproducible. By default False.
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

        if seed is not None:
            utils.set_all_seeds(seed)

        generation_config = config.create_idefics_config(self.processor, max_new_tokens=max_new_tokens,
                                                         min_new_tokens=min_new_tokens, do_sample=do_sample,
                                                         top_k=top_k, top_p=top_p, temperature=temperature)
        
        input = self.processor(prompt, return_tensors='pt')
        input_length = input['input_ids'].shape[-1]
        if torch.cuda.is_available():
            input = input.to(device=self.input_device)

        stopping_criteria = stopping.create_stopping_criteria(input_length, self.processor, stopping_patterns)

        output = self.model.generate(**input, generation_config=generation_config, stopping_criteria=stopping_criteria,
                                     **kwargs)
        truncated_output = output[:, input_length:]

        if post_process_output:
            generated_text = stopping.post_process_sequences(truncated_output, self.processor, stopping_patterns)
        else:
            generated_text = self.processor.batch_decode(truncated_output, skip_special_tokens=True)

        if not truncate_prompt_from_output:
            original_prompt = self.processor.batch_decode(output[:, 0:input_length], skip_special_tokens=True)[0]
            generated_text = [original_prompt + sequence for sequence in generated_text]
            
        if len(generated_text) == 1:
            return generated_text[0]
        else:
            return generated_text



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
            seed: int | None = None,
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
        seed : int | None, optional
            An optional seed to force the generation to be reproducible. By default False.
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
                                  top_k=top_k, top_p=top_p, temperature=temperature, seed=seed, stopping_patterns=stopping_patterns,
                                  truncate_prompt_from_output=truncate_prompt_from_output, post_process_output=post_process_output,
                                  **kwargs)
    




class ChatModel(object):

    def __init__(self, model_name, quantization_8bits: bool = False, quantization_4bits: bool = False,
                 dtype: torch.dtype | None = None, max_fraction_gpu_0: float = 0.8, max_fraction_gpus: float = 0.8,
                 device_map: dict | str | None = None, gpu_rank: int = 0):
        
        self.model, self.tokenizer = loader.load_model_and_processor(model_name, quantization_8bits=quantization_8bits,
                                                                     quantization_4bits=quantization_4bits, dtype=dtype,
                                                                     max_fraction_gpu_0=max_fraction_gpu_0,
                                                                     max_fraction_gpus=max_fraction_gpus,
                                                                     device_map=device_map, gpu_rank=gpu_rank)
        
        self.model_name = model_name
        self.quantization_8bits = quantization_8bits
        self.quantization_4bits = quantization_4bits
        # May be different from the dtype given in the arguments so use the model attribute
        self.dtype = self.model.dtype

        # In this case, the model is on multiple devices
        if hasattr(self.model, 'hf_device_map'):
            gpu_devices = set(self.model.hf_device_map.values())
            gpu_devices.discard('cpu')
            gpu_devices.discard('disk')

            self.input_device = min(gpu_devices) if len(gpu_devices) > 0 else 'cpu'
            
        # In this case, the model is on a single device
        else:
            device = next(self.model.parameters()).get_device()
            self.input_device = 'cpu' if device == -1 else device
        

    def generate_text(
            self,
            prompt: str,
            max_new_tokens: int = 512,
            min_new_tokens: int | None = None,
            do_sample: bool = True,
            top_k: int | None = 50,
            top_p: float | None = 0.9,
            temperature: float = 0.8,
            seed: int | None = None,
            stopping_patterns: list[str] | tuple[str] | None = None,
            truncate_prompt_from_output: bool = True,
            post_process_output: bool = True,
            **kwargs
        ) -> str | list[str]:
        """Generate a single auto-regresive text completion based on `prompt`.

        Input parameters
        ----------------

        prompt : str
            Input to the model.

        Generation parameters
        ---------------------

        max_new_tokens : int, optional
            Maximum number of new tokens to generate, by default 512
        min_new_tokens : int | None, optional
            The minimum number of tokens to generate, by setting the probability of EOS token to 0. Giving `None`
            is the same as giving 0. By default `None`.
        do_sample : bool, optional
            Whether to perform sampling or greedy search generation, by default True, i.e. sampling.
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
        seed : int | None, optional
            An optional seed to force the generation to be reproducible. By default False.
        stopping_patterns : list[str] | tuple[str] | None, optional
            The list of patterns to use to stop generation, by default None

        Output formatting parameters
        ----------------------------
        truncate_prompt_from_output : bool, optional
            Whether to remove the prompt from the model answer or not, by default True.
        post_process_output : bool, optional
            Whether to post-process the outputs, i.e. truncate according to the `stopping_patterns`. By default True.

        Returns
        -------
        str
            The text completion.
        """
    
        if seed is not None:
            utils.set_all_seeds(seed)

        # Override the default `self.model.generation_config` with our config to be sure of the generation mode
        generation_config = config.create_config(self.tokenizer, max_new_tokens=max_new_tokens,
                                                 min_new_tokens=min_new_tokens, do_sample=do_sample, top_k=top_k,
                                                 top_p=top_p, temperature=temperature)

        # Tokenize the prompt
        input = self.tokenizer.encode(prompt, return_tensors='pt')
        input_length = input.shape[-1]
        if torch.cuda.is_available():
            input = input.to(device=self.input_device)

        # Create the stopping criteria
        stopping_criteria = stopping.create_stopping_criteria(input_length, self.tokenizer, stopping_patterns)

        outputs = self.model.generate(input, generation_config=generation_config, stopping_criteria=stopping_criteria,
                                      **kwargs)
                
        # Truncate the prompt from the output
        truncated_outputs = outputs[:, input_length:]

        # Post-process the sequences according to stopping patterns and extra eos
        if post_process_output:
            generated_text = stopping.post_process_sequences(truncated_outputs, self.tokenizer, stopping_patterns)
        else:
            generated_text = self.tokenizer.batch_decode(truncated_outputs, skip_special_tokens=True)
        
        # reattach the prompt if needed
        if not truncate_prompt_from_output:
            generated_text = [prompt + sequence for sequence in generated_text]

        # In this case return a str instead of list[str]
        if len(generated_text) == 1:
            return generated_text[0]
        else:
            return generated_text
        


    def generate_conversation(
            self,
            prompt: str,
            system_prompt: str | None = None,
            conv_history: template.GenericConversationTemplate | None = None,
            max_new_tokens: int = 512,
            min_new_tokens: int | None = None,
            do_sample: bool = True,
            top_k: int = 50,
            top_p: float = 0.9,
            temperature: float = 0.8,
            seed: int | None = None,
            stopping_patterns: list[str] | tuple[str] | None = None,
            truncate_if_conv_too_long: bool = True,
            **kwargs
    ) -> template.GenericConversationTemplate:
        """Generate a conversation turn between a user and the model, according to new user input `prompt`.

        Input parameters
        ----------------
        prompt : str
            The new prompt of the user to the model.
        system_prompt : str | None, optional
            An optional system prompt to guide the style of the model answers. The default is `None` which uses
            the default template system prompt.
        conv_history : template.GenericConversationTemplate | None, optional
            An optional existing conversation object, representing the current dialogue between the user and
            the model. The default is `None`.

        Generation parameters
        ---------------------

        max_new_tokens : int, optional
            How many new tokens to generate, by default 512.
        min_new_tokens : int | None, optional
            The minimum number of tokens to generate, by setting the probability of EOS token to 0. Giving `None`
            is the same as giving 0. By default `None`.
        do_sample : bool, optional
            Whether to introduce randomness in the generation, by default True.
        top_k : int | None, optional
            How many tokens with max probability to consider for random sampling, by default 50. Not used if 
            `do_sample=False`. You can deactivate top_k sampling by providing `top_k=0` or `top_k=None`. Note 
            that if you provide both `top_k` and `top_p`, the `top_k` is applied before. By default 50.
        top_p : float | None, optional
            The probability density covering the new tokens to consider for random sampling, by default 0.9. Not used if 
            `do_sample=False`. You can deactivate top_p sampling by providing `top_p=1` or `top_p=None`. Note 
            that if you provide both `top_k` and `top_p`, the `top_k` is applied before. By default 0.9.
        temperature : float, optional
            How to cool down the probability distribution. Value between 1 (no cooldown) and 0 (greedy search,
            no randomness), by default 0.9. Passing 0 is equivalent to setting `do_sample=False`. By default 0.8.
        seed : int | None, optional
            An optional seed to force the generation to be reproducible. By default `None`.
        stopping_patterns : list[str] | tuple[str] | None, optional
            The list of patterns to use to stop generation, by default None
        truncate_if_conv_too_long : bool, optional
            Whether to truncate the conversation history if it becomes larger than the model maximum capacity,
            by default True.

        Returns
        -------
        GenericConversation
            A conversation object, with the dialogue history updated with the current turn.
        """

        if seed is not None:
            utils.set_all_seeds(seed)

        # Override the default `self.model.generation_config` with our config to be sure of the generation mode
        generation_config = config.create_config(self.tokenizer, max_new_tokens=max_new_tokens,
                                                 min_new_tokens=min_new_tokens, do_sample=do_sample, top_k=top_k,
                                                 top_p=top_p, temperature=temperature)

        # Check that the history is not empty
        if conv_history is None:
            conv_history = self.get_empty_conversation()

        # Set system prompt
        if system_prompt is not None:
            conv_history.set_system_prompt(system_prompt)

        # Add the prompt to the current conversation
        conv_history.append_user_message(prompt)

        # Generate and tokenize the full prompt
        if truncate_if_conv_too_long:
            truncated_conv = self.truncate_conversation(conv_history, max_new_tokens, continuation=False)
            full_prompt = truncated_conv.get_prompt()
        else:
            full_prompt = conv_history.get_prompt()

        input = self.tokenizer.encode(full_prompt, return_tensors='pt')
        input_length = input.shape[-1]
        if torch.cuda.is_available():
            input = input.to(device=self.input_device)

        # Create the stopping criteria in case the model has some extra eos tokens to process
        stopping_criteria  = stopping.create_stopping_criteria(input_length, self.tokenizer, stopping_patterns)

        outputs = self.model.generate(input, generation_config=generation_config, stopping_criteria=stopping_criteria,
                                      num_return_sequences=1, **kwargs)
                
        # Truncate the prompt from the output
        truncated_outputs = outputs[:, input_length:]

        # Post-process the sequences according to potential extra eos tokens
        response = stopping.post_process_sequences(truncated_outputs, self.tokenizer, stopping_patterns)
        
        # Append output to the conv
        conv_history.append_model_message(response[0])

        return conv_history
    

    def continue_last_conversation_turn(
            self,
            conv_history: template.GenericConversationTemplate,
            max_new_tokens: int = 128,
            do_sample: bool = True,
            top_k: int = 50,
            top_p: float = 0.9,
            temperature: float = 0.8,
            seed: int | None = None,
            stopping_patterns: list[str] | tuple[str] | None = None,
            truncate_if_conv_too_long: bool = True,
            **kwargs
    ) -> template.GenericConversationTemplate:
        """Continue the last conversation turn if the model stopped too early due to `max_new_tokens` being too
        low.

        Input parameters
        ----------------
        conv_history : template.GenericConversationTemplate
            An existing conversation object, representing the current dialogue between the user and
            the model.

        Generation parameters
        ---------------------

        max_new_tokens : int, optional
            How many new tokens to generate, by default 128.
        do_sample : bool, optional
            Whether to introduce randomness in the generation, by default True.
        top_k : int | None, optional
            How many tokens with max probability to consider for random sampling, by default 50. Not used if 
            `do_sample=False`. You can deactivate top_k sampling by providing `top_k=0` or `top_k=None`. Note 
            that if you provide both `top_k` and `top_p`, the `top_k` is applied before. By default 50.
        top_p : float | None, optional
            The probability density covering the new tokens to consider for random sampling, by default 0.9. Not used if 
            `do_sample=False`. You can deactivate top_p sampling by providing `top_p=1` or `top_p=None`. Note 
            that if you provide both `top_k` and `top_p`, the `top_k` is applied before. By default 0.9.
        temperature : float, optional
            How to cool down the probability distribution. Value between 1 (no cooldown) and 0 (greedy search,
            no randomness), by default 0.9. Passing 0 is equivalent to setting `do_sample=False`. By default 0.8.
        seed : int | None, optional
            An optional seed to force the generation to be reproducible. By default `None`.
        stopping_patterns : list[str] | tuple[str] | None, optional
            The list of patterns to use to stop generation, by default None
        truncate_if_conv_too_long : bool, optional
            Whether to truncate the conversation history if it becomes larger than the model maximum capacity,
            by default True.

        Returns
        -------
        template.GenericConversationTemplate
            A conversation object, with the dialogue history updated with the current turn.
        """

        if seed is not None:
            utils.set_all_seeds(seed)

        # Override the default `self.model.generation_config` with our config to be sure of the generation mode
        generation_config = config.create_config(self.tokenizer, max_new_tokens=max_new_tokens,
                                                 min_new_tokens=None, do_sample=do_sample, top_k=top_k,
                                                 top_p=top_p, temperature=temperature)

        # Generate and tokenize the full prompt
        if truncate_if_conv_too_long:
            truncated_conv = self.truncate_conversation(conv_history, max_new_tokens, continuation=True)
            full_prompt = truncated_conv.get_last_turn_continuation_prompt()
        else:
            full_prompt = conv_history.get_last_turn_continuation_prompt()
        input = self.tokenizer.encode(full_prompt, return_tensors='pt')
        input_length = input.shape[-1]
        if torch.cuda.is_available():
            input = input.to(device=self.input_device)

        # Create the stopping criteria in case the model has some extra eos tokens to process
        stopping_criteria  = stopping.create_stopping_criteria(input_length, self.tokenizer, stopping_patterns)

        outputs = self.model.generate(input, generation_config=generation_config, stopping_criteria=stopping_criteria,
                                      num_return_sequences=1, **kwargs)
                
        # Truncate the prompt from the output
        truncated_outputs = outputs[:, input_length:]

        # Add space if it is missing due to Llama tokenization process
        first_token = self.tokenizer.convert_ids_to_tokens(int(truncated_outputs[0, 0]))
        llama_space_character = b'\xe2\x96\x81'.decode()
        if first_token.startswith(llama_space_character):
            add_space = True
        else:
            add_space = False

        # Post-process the sequences
        response = stopping.post_process_sequences(truncated_outputs, self.tokenizer, stopping_patterns)
        
        # Append output to the conv
        if add_space:
            conv_history.append_to_last_model_message(' ' + response[0])
        else:
            conv_history.append_to_last_model_message(response[0])

        return conv_history
    

    def get_empty_conversation(self, system_prompt: str | None = None) -> template.GenericConversationTemplate:
        """Return a new empty conversation with the template of the current model."""
        if system_prompt is None:
            return template.TEMPLATE_MAPPING[self.model_name]()
        else:
            return template.TEMPLATE_MAPPING[self.model_name](system_prompt=system_prompt)
    

    def get_context_size(self) -> int:
        """Return the maximum context size for the current model."""
        return loader.get_model_context_size(self.model_name)
   

    def truncate_conversation(self, conversation: template.GenericConversationTemplate, max_new_tokens: int,
                              continuation: bool = False) -> template.GenericConversationTemplate:
        """Truncate the current conversation by removing the oldest messages so that the length of the prompt
        + the `max_new_tokens` fit the maximum context length that the model can handle.

        Parameters
        ----------
        conversation : template.GenericConversationTemplate
            The current conversation.
        max_new_tokens : int
            How many new tokens to generate.
        continuation : bool, optional
            Whether we continue the last conversation turn, or create a new one. By default `False`.

        Returns
        -------
        template.GenericConversationTemplate
            The truncated conversation.
        """

        if len(conversation) == 0:
            raise ValueError('Cannot truncate an empty conversation.')
        
        context_size = self.get_context_size()

        new_conv = copy.deepcopy(conversation)
        if continuation:
            full_prompt = new_conv.get_last_turn_continuation_prompt()
        else:
            full_prompt = new_conv.get_prompt()
        input = self.tokenizer.encode(full_prompt, return_tensors='pt')
        input_length = input.shape[-1]

        while input_length + max_new_tokens >= context_size:
            del new_conv.user_history_text[0]
            del new_conv.model_history_text[0]

            if len(new_conv) == 0:
                raise RuntimeError('The entire conversation got truncated to fit the context size.')

            if continuation:
                full_prompt = new_conv.get_last_turn_continuation_prompt()
            else:
                full_prompt = new_conv.get_prompt()
            input = self.tokenizer.encode(full_prompt, return_tensors='pt')
            input_length = input.shape[-1]

        return new_conv
    


class DummyModel(object):
    """Dummy model only used for debugging purposes on the chatbot."""

    def __init__(self):
        self.tokenizer = 'dummy_tokenizer_attribute'

    def generate_text(
            self,
            prompt: str,
            max_new_tokens: int = 512,
            min_new_tokens: int | None = None,
            do_sample: bool = True,
            top_k: int | None = 50,
            top_p: float | None = 0.9,
            temperature: float = 0.8,
            seed: int | None = None,
            stopping_patterns: list[str] | tuple[str] | None = None,
            truncate_prompt_from_output: bool = True,
            post_process_output: bool = True,
            **kwargs
        ) -> str | list[str]:
        
        return 'This is a test'
        


    def generate_conversation(
            self,
            prompt: str,
            system_prompt: str | None = None,
            conv_history: template.GenericConversationTemplate | None = None,
            max_new_tokens: int = 512,
            min_new_tokens: int | None = None,
            do_sample: bool = True,
            top_k: int = 50,
            top_p: float = 0.9,
            temperature: float = 0.8,
            seed: int | None = None,
            stopping_patterns: list[str] | tuple[str] | None = None,
            truncate_if_conv_too_long: bool = True,
            **kwargs
    ) -> template.GenericConversationTemplate:
        
        # Check that the history is not empty
        if conv_history is None:
            conv_history = self.get_empty_conversation()
        # Add the prompt to the current conversation
        conv_history.append_user_message(prompt)
        conv_history.append_model_message('This is a test asnwer.')
        return conv_history
    

    def continue_last_conversation_turn(
            self,
            conv_history: template.GenericConversationTemplate,
            max_new_tokens: int = 128,
            do_sample: bool = True,
            top_k: int = 50,
            top_p: float = 0.9,
            temperature: float = 0.8,
            seed: int | None = None,
            stopping_patterns: list[str] | tuple[str] | None = None,
            truncate_if_conv_too_long: bool = True,
            **kwargs
    ) -> template.GenericConversationTemplate:
        
        conv_history.append_to_last_model_message(' This is a continuation test asnwer.')
        return conv_history
    

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
            seed: int | None = None,
            stopping_patterns: list[str] | tuple[str] | None = stopping.IDEFICS_STOP_PATTERNS,
            truncate_prompt_from_output: bool = True,
            post_process_output: bool = True,
            **kwargs
        ) -> str:

        return 'This is a test.'
    

    def get_empty_conversation(self, system_prompt: str = template.LLAMA2_NUTRITION_SYSTEM_PROMPT) -> template.GenericConversationTemplate:
        """Return a new empty conversation with the template of the current model."""
        return template.Llama2ChatConversationTemplate(system_prompt=system_prompt)
    