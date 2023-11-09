from transformers import GenerationConfig, ProcessorMixin, PreTrainedTokenizerBase


def create_config(tokenizer: PreTrainedTokenizerBase, max_new_tokens: int, min_new_tokens: int | None = None,
                  do_sample: bool = False, top_k: int | None = 50, top_p: float | None = 0.9,
                  temperature: float = 0.8) -> GenerationConfig:
    """Create a new `GenerationConfig` object to pass to `model.generate()` to control the generation strategy
    of the idefics model.
    This is needed because by default `generate()` uses `model.generation_config` if the `generation_config`
    parameter is not provided, which may conflict with some of our parameters and thus provide incorrect
    or suprising results.
    
    Parameters
    ----------
    tokenizer : PreTrainedTokenizerBase
        The tokenizer used by the model.
    max_new_tokens : int
        How many new tokens to generate.
    min_new_tokens : int | None, optional
        The minimum number of tokens to generate, by setting the probability of EOS token to 0. It is useful to
        force the model to generate an output, instead of immediately generating EOS. By default None.
    do_sample : bool, optional
        Whether to introduce randomness in the generation. By default False.
    top_k : int | None, optional
        How many tokens with max probability to consider for random sampling. Not used if 
        `do_sample=False`. You can deactivate top_k sampling by providing `top_k=0` or `top_k=None`. Note 
        that if you provide both `top_k` and `top_p`, the `top_k` is applied before. By default 50.
    top_p : float | None, optional
        The probability density covering the new tokens to consider for random sampling. Not used if 
        `do_sample=False`. You can deactivate top_p sampling by providing `top_p=1` or `top_p=None`. Note 
        that if you provide both `top_k` and `top_p`, the `top_k` is applied before. By default 0.9.
    temperature : float, optional
        How to cool down the probability distribution. Value between 1 (no cooldown) and 0 (greedy search,
        no randomness). Passing 0 is equivalent to setting `do_sample=False`. By default 0.8.

    Returns
    -------
    GenerationConfig
        Config which controls the text generation.
    """

    # Setting the temperature to 0 is equivalent to greedy search thus we explicitly set do_sample=False
    if temperature == 0:
        do_sample = False

    eos_token_id = tokenizer.eos_token_id
    bos_token_id = tokenizer.bos_token_id
    pad_token_id = tokenizer.pad_token_id

    if min_new_tokens is not None:
        min_new_tokens = min_new_tokens if min_new_tokens > 0 else None

    # create the config
    generation_config = GenerationConfig(eos_token_id=eos_token_id, bos_token_id=bos_token_id,
                                         pad_token_id=pad_token_id, max_new_tokens=max_new_tokens,
                                         min_new_tokens=min_new_tokens, do_sample=do_sample)
    
    # Add parameters to the config
    if do_sample:
        unused = generation_config.update(top_k=top_k, top_p=top_p, temperature=temperature)
        assert len(unused) == 0, 'There is a typo in some generation config parameters.'

    return generation_config



def create_idefics_config(processor: ProcessorMixin | PreTrainedTokenizerBase, max_new_tokens: int,
                          min_new_tokens: int | None = None, do_sample: bool = False, top_k: int | None = 50,
                          top_p: float | None = 0.9, temperature: float = 0.8) -> GenerationConfig:
    """Create a new `GenerationConfig` object to pass to `model.generate()` to control the generation strategy
    of the idefics model.
    This is needed because by default `generate()` uses `model.generation_config` if the `generation_config`
    parameter is not provided, which may conflict with some of our parameters and thus provide incorrect
    or suprising results.
    
    Parameters
    ----------
    processor : ProcessorMixin
        The processor used by the model.
    max_new_tokens : int
        How many new tokens to generate.
    min_new_tokens : int | None, optional
        The minimum number of tokens to generate, by setting the probability of EOS token to 0. It is useful to
        force the model to generate an output, instead of immediately generating EOS. By default None.
    do_sample : bool, optional
        Whether to introduce randomness in the generation. By default False.
    top_k : int | None, optional
        How many tokens with max probability to consider for random sampling. Not used if 
        `do_sample=False`. You can deactivate top_k sampling by providing `top_k=0` or `top_k=None`. Note 
        that if you provide both `top_k` and `top_p`, the `top_k` is applied before. By default 50.
    top_p : float | None, optional
        The probability density covering the new tokens to consider for random sampling. Not used if 
        `do_sample=False`. You can deactivate top_p sampling by providing `top_p=1` or `top_p=None`. Note 
        that if you provide both `top_k` and `top_p`, the `top_k` is applied before. By default 0.9.
    temperature : float, optional
        How to cool down the probability distribution. Value between 1 (no cooldown) and 0 (greedy search,
        no randomness). Passing 0 is equivalent to setting `do_sample=False`. By default 0.8.

    Returns
    -------
    GenerationConfig
        Config which controls the text generation.
    """

    bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids
    gen_config = create_config(processor.tokenizer, max_new_tokens=max_new_tokens, min_new_tokens=min_new_tokens,
                               do_sample=do_sample, top_k=top_k, top_p=top_p, temperature=temperature)
    
    # Add the bad words ids to the config
    unused = gen_config.update(bad_words_ids=bad_words_ids)
    assert len(unused) == 0, 'There is a typo in some generation config parameters.'

    return gen_config