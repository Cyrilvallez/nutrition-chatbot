import re
import math
import warnings

import torch
from transformers import AutoModelForCausalLM, IdeficsForVisionText2Text, AutoProcessor, AutoTokenizer

def _infer_model_sizes(name_mapping: dict[str, str]) -> dict[str, float]:
    """Infer the number of parameters of all model names (dict keys) and return them as {key: #params}

    Parameters
    ----------
    name_mapping : dict[str, str]
        A dictionary whose keys are the model names.

    Returns
    -------
    dict[str, float]
        A mapping from names to number of parameters.
    """

    # The following regex matches any digits possibly separated with a dot ('.') which is immeditely
    # followed by a 'B' or 'M' to capture the model size following our model name convention. Parenthesis 
    # allow to capture given groups of the regex thanks to the match object .group() method.
    pattern = r'([0-9]+(?:\.[0-9]+)?)([BM])'

    out = {}

    for model_name in name_mapping.keys():
        match = re.search(pattern, model_name)
        if match:
            matched_number = match.group(1)
            matched_letter = match.group(2)
            # Model size in billion (B) of parameters
            model_size = float(matched_number) if matched_letter == 'B' else float(matched_number)/1e3
            out[model_name] = model_size
        else:
            raise ValueError('The model number of parameters cannot be inferred from its name.')
    
    return out
    


# Idefics models
IDEFICS_MODELS_MAPPING = {
    'idefics-9B': 'HuggingFaceM4/idefics-9b',
    'idefics-80B': 'HuggingFaceM4/idefics-80b',
    'idefics-9B-instruct': 'HuggingFaceM4/idefics-9b-instruct',
    'idefics-80B-instruct': 'HuggingFaceM4/idefics-80b-instruct',
}
IDEFICS_MODELS_DTYPES = {model: torch.bfloat16 for model in IDEFICS_MODELS_MAPPING.keys()}
IDEFICS_MODELS_SIZES = _infer_model_sizes(IDEFICS_MODELS_MAPPING)
IDEFICS_MODELS_CONTEXT_SIZE = {model: 2048 for model in IDEFICS_MODELS_MAPPING.keys()}


# Llama2-chat models
LLAMA2_MODELS_MAPPING = {
    'llama2-7B-chat': 'meta-llama/Llama-2-7b-chat-hf',
    'llama2-13B-chat': 'meta-llama/Llama-2-13b-chat-hf',
    'llama2-70B-chat': 'meta-llama/Llama-2-70b-chat-hf',
}
LLAMA2_MODELS_DTYPES = {model: torch.float16 for model in LLAMA2_MODELS_MAPPING.keys()}
LLAMA2_MODELS_SIZES = _infer_model_sizes(LLAMA2_MODELS_MAPPING)
LLAMA2_MODELS_CONTEXT_SIZE = {model: 4096 for model in LLAMA2_MODELS_MAPPING.keys()}
LLAMA2_MODELS_ADDITIONAL_TOKENIZER_KWARGS = {model: {'use_fast': False} for model in LLAMA2_MODELS_MAPPING.keys()}


# Combine all model attributes
ALL_MODELS_MAPPING = {
    **IDEFICS_MODELS_MAPPING,
    **LLAMA2_MODELS_MAPPING,
}
ALL_MODELS_DTYPES = {
    **IDEFICS_MODELS_DTYPES,
    **LLAMA2_MODELS_DTYPES,
}
ALL_MODELS_SIZES = {
    **IDEFICS_MODELS_SIZES,
    **LLAMA2_MODELS_SIZES,
}
ALL_MODELS_CONTEXT_SIZE = {
    **IDEFICS_MODELS_CONTEXT_SIZE,
    **LLAMA2_MODELS_CONTEXT_SIZE,
}
ALL_MODELS_ADDITIONAL_TOKENIZER_KWARGS = {
    **LLAMA2_MODELS_ADDITIONAL_TOKENIZER_KWARGS,
}


# Summarize all supported model names and dtypes
ALLOWED_MODELS = tuple(ALL_MODELS_MAPPING.keys())
ALLOWED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def get_model_dtype(model_name: str) -> torch.dtype:
    """Return the default dtype used by the model.

    Parameters
    ----------
    model_name : str
        The name of the model.

    Returns
    -------
    torch.dtype
        The default dtype.
    """

    if model_name not in ALLOWED_MODELS:
        raise ValueError(f'The model name must be one of {*ALLOWED_MODELS,}.')
    
    return ALL_MODELS_DTYPES[model_name]


def get_model_size(model_name: str) -> float:
    """Return the approximate number of params of the model, in billions.

    Parameters
    ----------
    model_name : str
        The name of the model.

    Returns
    -------
    float
        The number of parameters.
    """

    if model_name not in ALLOWED_MODELS:
        raise ValueError(f'The model name must be one of {*ALLOWED_MODELS,}.')
    
    return ALL_MODELS_SIZES[model_name]


def get_model_context_size(model_name: str) -> int:
    """Return the maximum context size used by the model.

    Parameters
    ----------
    model_name : str
        The name of the model.

    Returns
    -------
    int
        The context size.
    """

    if model_name not in ALLOWED_MODELS:
        raise ValueError(f'The model name must be one of {*ALLOWED_MODELS,}.')
    
    return ALL_MODELS_CONTEXT_SIZE[model_name]


def estimate_model_gpu_footprint(model_name, quantization_8bits: bool = False, quantization_4bits: bool = False,
                                 dtype: torch.dtype | None = None, max_fraction_gpu_0: float = 0.8,
                                 max_fraction_gpus: float = 0.8) -> tuple[int, dict]:
    """Estimate the minimum number of gpus needed to perform inference with a model, given the maximum gpu memory
    proportion `max_fraction_gpu_0` and `max_fraction_gpus` that we allow for the model. This relies on
    simple heuristics. This also computes the corresponding `memory_map` to use when creating a `device_map`.

    Parameters
    ----------
    model_name : str
        The model name.
    quantization_8bits : bool
        Whether the model will be loaded in 8 bits mode, by default False.
    quantization_4bits : bool
        Whether the model will be loaded in 4 bits mode, by default False.
    dtype : torch.dtype | None, optional
        The dtype to use for the model. If not provided, we use the dtype with which the model was trained
        if it is known, else we use float32, by default None.
    max_fraction_gpu_0 : float, optional
        The maximum fraction of the gpu 0 memory to reserve for the model. The default is 0.8.
    max_fraction_gpus : float, optional
        The maximum fraction of the other gpus memory to reserve for the model. The default is 0.8.

    Returns
    -------
    tuple[int, dict]
        Tuple containing the minimum number of gpus needed, the `memory_map`, i.e. a dictionary mapping each gpu
        needed to the maximum size reserved by the model for this gpu.
    """

    if max_fraction_gpu_0 < 0 or max_fraction_gpus < 0:
        raise ValueError('The maximum fraction of gpu memory to use cannot be negative.')
    
    if max_fraction_gpu_0 > 0.95 or max_fraction_gpus > 0.95:
        raise ValueError(('The maximum fraction of gpu memory to use cannot be larger than 0.95 because some '
                         'memory need to stay free for the forward pass and other computations.'))
    
    # Silently use 4bits when both are True
    if quantization_4bits and quantization_8bits:
        quantization_8bits = False

    # If not provided take the default one
    if dtype is None:
        dtype = get_model_dtype(model_name)

    if quantization_4bits:
        size_multiplier = 1/2
    elif quantization_8bits:
        size_multiplier = 1
    elif (dtype == torch.float16) or (dtype == torch.bfloat16):
        size_multiplier = 2
    else:
        size_multiplier = 4

    # Estimate of the memory size of the model
    rough_model_size_estimate = get_model_size(model_name) * size_multiplier
    
    # We assume that we always have identical gpus when using multiple gpus
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    # If cuda is not available, run this function assuming A100 40GB gpus
    else:
        gpu_memory = 40

    # Say we only have access to a portion of that memory for our model
    gpu_0_available_memory = max_fraction_gpu_0 * gpu_memory
    gpus_available_memory = max_fraction_gpus * gpu_memory

    # Heuristic: if the remainder is smaller than 2% of gpu_memory, do not add a gpu 
    if rough_model_size_estimate <= gpu_0_available_memory + 0.02 * gpu_memory:
        return 1, None
    
    else:
        max_memory_map = {0: f'{math.ceil(gpu_0_available_memory)}GiB'}
        to_fit_on_other_gpus = rough_model_size_estimate - gpu_0_available_memory
        additional_gpus_needed = int(to_fit_on_other_gpus // gpus_available_memory)

        # Heuristic: if the remainder is smaller than 2% of each gpu_memory, do not add a gpu and distill
        # the small excess between existing gpus
        if to_fit_on_other_gpus % gpus_available_memory >= (0.02 * gpu_memory) * additional_gpus_needed:
            additional_gpus_needed += 1
            available_gpu_size = math.ceil(gpus_available_memory)
        else:
            # Add the 2% to the gpus size requirements
            available_gpu_size = math.ceil((max_fraction_gpus + 0.02) * gpu_memory)

        gpu_needed = 1 + additional_gpus_needed
        for i in range(1, gpu_needed):
            max_memory_map[i] = f'{available_gpu_size}GiB'

        return gpu_needed, max_memory_map
    



def load_model(model_name: str, quantization_8bits: bool = False, quantization_4bits: bool = False,
               dtype: torch.dtype | None = None, max_fraction_gpu_0: float = 0.8, max_fraction_gpus: float = 0.8,
               device_map: dict | str | None = None, gpu_rank: int = 0):
    """Load one of the supported pretrained model.

    Parameters
    ----------
    model_name : str
        The model name.
    quantization_8bits : bool
        Whether the model will be loaded in 4 bits mode, by default False. This argument is mutually exclusive
        with `quantization_4bits`.
    quantization_4bits : bool
        Whether the model will be loaded in 4 bits mode, by default False. This argument is mutually exclusive
        with `quantization_8bits`.
    dtype : torch.dtype | None, optional
        The dtype to use for the model. If not provided, we use the dtype with which the model was trained
        if it is known, else we use float32, by default None.
    max_fraction_gpu_0 : float, optional
        The maximum fraction of the gpu 0 memory to reserve for the model. The default is 0.8. This is only
        used if `device_map` is `None`.
    max_fraction_gpus : float, optional
        The maximum fraction of the other gpus memory to reserve for the model. The default is 0.8. This is only
        used if `device_map` is `None`.
    device_map : dict | str | None, optional
        The device map to decide how to split the model between available devices, by default None. If not
        provided, the model dispatch to GPU(s) is made according to `max_fraction_gpu_0` and `max_fraction_gpus`
        in such a way to use the smallest number of gpus that respect these two values.
    gpu_rank : int, optional
        The gpu rank on which to put the model ONLY if it can fit on a single gpu. This is ignored if `device_map`
        is provided. By default 0.

    Returns
    -------
        The model.
    """

    if model_name not in ALLOWED_MODELS:
        raise ValueError(f'The model name must be one of {*ALLOWED_MODELS,}.')
    
    # Set the dtype if not provided
    if dtype is None:
        dtype = get_model_dtype(model_name)

    if dtype not in ALLOWED_DTYPES:
        raise ValueError(f'The dtype must be one of {*ALLOWED_DTYPES,}.')
    
    if quantization_8bits and quantization_4bits:
        raise ValueError(('You cannot load a model with both `quantization_8bits` and `quantization_4bits`. '
                         'Please choose one'))
    
    # torch.float16 is not supported on cpu
    if not torch.cuda.is_available() and dtype != torch.float32:
        dtype = torch.float32
    
    # Override quantization if we don't have access to GPUs
    if not torch.cuda.is_available() and (quantization_8bits or quantization_4bits):
        quantization_4bits = False
        quantization_8bits = False
        warnings.warn('There are no GPUs available. The model will NOT be quantized.', RuntimeWarning)

    # Flag to know if the model is quantized
    quantization = quantization_8bits or quantization_4bits

    # Override dtype if we quantize the model as only float16 is acceptable for quantization
    dtype = torch.float16 if quantization else dtype

    additional_kwargs = {}

    # Flag that will be set to True if we don't even need a device_map and can just put the model on one gpu
    only_move_to_one_gpu = False
    
    # Automatically find the best device_map depending on the model size and gpu size.
    # Try to minimize the number of gpus to use because using more will slow inference (but allow larger
    # batch size -> hard trade-off to find). Indeed, the parallelism of device_map is naive and gpus are only
    # used sequentially
    if (device_map is None) and torch.cuda.is_available():
    
        min_gpu_needed, max_memory_map = estimate_model_gpu_footprint(model_name, quantization_8bits=quantization_8bits,
                                                                      quantization_4bits=quantization_4bits, dtype=dtype,
                                                                      max_fraction_gpu_0=max_fraction_gpu_0,
                                                                      max_fraction_gpus=max_fraction_gpus)
        gpu_number = torch.cuda.device_count()

        if min_gpu_needed > gpu_number:
            raise RuntimeError(("The model seems too big for the gpu resources you have. To offload to the cpu as well, "
                               "explicitly pass a `device_map`, e.g. device_map='balanced'."))
        
        # In this case we don't need a device_map, we just move the model to the 1st gpu. Most models are 
        # relatively small and should fall on this category.
        if min_gpu_needed == 1:
            only_move_to_one_gpu = True
        # In this case, we need more than 1 gpu so we create a device_map between different gpus. However, 
        # we minimize the number of gpus used with the max_memory arg instead of naively using device_map='balanced'
        # between all gpus, because the parallelism is not optimized and thus using a lot of gpus is not efficient
        # if not needed
        else:
            additional_kwargs['max_memory'] = max_memory_map
            # Providing 'balanced' dispatch correctly with respect to the max_memory_map we provide
            device_map = 'balanced'

    # Load model
    if model_name in IDEFICS_MODELS_MAPPING.keys():
        model = IdeficsForVisionText2Text.from_pretrained(ALL_MODELS_MAPPING[model_name], device_map=device_map,
                                                          torch_dtype=dtype, load_in_8bit=quantization_8bits,
                                                          load_in_4bit=quantization_4bits, low_cpu_mem_usage=True,
                                                          **additional_kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(ALL_MODELS_MAPPING[model_name], device_map=device_map,
                                                    torch_dtype=dtype, load_in_8bit=quantization_8bits,
                                                    load_in_4bit=quantization_4bits, low_cpu_mem_usage=True,
                                                    **additional_kwargs)
    
    # If the flag is active we directly put our model on one gpu without using any device_map (this is 
    # more efficient). But if the model is quantized, this is already done automatically because quantization
    # happen only on gpu
    if only_move_to_one_gpu and not quantization:
        # This operation is in-place for nn.Module
        model.cuda(gpu_rank)

    # Convert to better transformer to use Pytorch optimizations if supported by the model
    try:
        model = model.to_bettertransformer()
    except:
        pass
        
    model.eval()

    return model



def load_processor_or_tokenizer(model_name: str):
    """Load a pretrained processor or tokenizer corresponding to one of the supported models.

    Parameters
    ----------
    model_name : str
        The model name.

    Returns
    -------
        The tokenizer.
    """

    if model_name not in ALLOWED_MODELS:
        raise ValueError(f'The model name must be one of {*ALLOWED_MODELS,}.') 
    
    if model_name in ALL_MODELS_ADDITIONAL_TOKENIZER_KWARGS.keys():
        additional_kwargs = ALL_MODELS_ADDITIONAL_TOKENIZER_KWARGS[model_name]
    else:
        additional_kwargs = {}
    
    if model_name in IDEFICS_MODELS_MAPPING.keys():
        processor = AutoProcessor.from_pretrained(ALL_MODELS_MAPPING[model_name], **additional_kwargs)
        return processor
    else:
        tokenizer = AutoTokenizer.from_pretrained(ALL_MODELS_MAPPING[model_name], **additional_kwargs)
        return tokenizer
    

def load_model_and_processor(model_name: str, quantization_8bits: bool = False, quantization_4bits: bool = False,
                             dtype: torch.dtype | None = None, max_fraction_gpu_0: float = 0.8,
                             max_fraction_gpus: float = 0.8, device_map: dict | str | None = None,
                             gpu_rank: int = 0) -> tuple:
    """Load both a model and corresponding processor (or tokenizer).

    Parameters
    ----------
    model_name : str
        The model name.
    quantization_8bits : bool
        Whether the model will be loaded in 4 bits mode, by default False. This argument is mutually exclusive
        with `quantization_4bits`.
    quantization_4bits : bool
        Whether the model will be loaded in 4 bits mode, by default False. This argument is mutually exclusive
        with `quantization_8bits`.
    dtype : torch.dtype | None, optional
        The dtype to use for the model. If not provided, we use the dtype with which the model was trained
        if it is known, else we use float32, by default None.
    max_fraction_gpu_0 : float, optional
        The maximum fraction of the gpu 0 memory to reserve for the model. The default is 0.8. This is only
        used if `device_map` is `None`.
    max_fraction_gpus : float, optional
        The maximum fraction of the other gpus memory to reserve for the model. The default is 0.8. This is only
        used if `device_map` is `None`.
    device_map : dict | str | None, optional
        The device map to decide how to split the model between available devices, by default None. If not
        provided, the model dispatch to GPU(s) is made according to `max_fraction_gpu_0` and `max_fraction_gpus`
        in such a way to use the smallest number of gpus that respect these two values.
    gpu_rank : int, optional
        The gpu rank on which to put the model ONLY if it can fit on a single gpu. This is ignored if `device_map`
        is provided. By default 0.

    Returns
    -------
    tuple
        The model and processor (tokenizer).
    """

    return (load_model(model_name, quantization_8bits=quantization_8bits, quantization_4bits=quantization_4bits,
                       dtype=dtype, max_fraction_gpu_0=max_fraction_gpu_0, max_fraction_gpus=max_fraction_gpus,
                       device_map=device_map, gpu_rank=gpu_rank),
            load_processor_or_tokenizer(model_name))