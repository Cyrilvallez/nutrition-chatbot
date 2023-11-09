import torch
import numpy as np
from transformers import PreTrainedTokenizerBase, ProcessorMixin, StoppingCriteria


IDEFICS_STOP_PATTERNS = (
    '\nUser:',
    '<end_of_utterance>',
)


class TextPatternStopping(StoppingCriteria):
    """Stop generation upon meeting any of the `stopping_patterns`.
    """

    def __init__(self, prompt_ids_length: int, processor: ProcessorMixin | PreTrainedTokenizerBase,
                 stopping_patterns: list[str] | tuple[str] = IDEFICS_STOP_PATTERNS):

        super().__init__()
        self.prompt_ids_length = prompt_ids_length
        self.processor = processor
        self.patterns = tuple(stopping_patterns)


    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        """Return `True` if all sequences are finished being generated (i.e. there is at least one stopping
        pattern or eos in each sequence). Unfortunately, this cannot return a list of boolean to inform
        the generation function which sequences are done or not, and append <pad-token> to the finished
        sequences.

        Parameters
        ----------
        input_ids : torch.LongTensor
            Outputs ids of the model.
        scores : torch.FloatTensor
            Scores.

        Returns
        -------
        bool
            `True` if all sequences are done being generated, `False` otherwise.
        """

        # If this was initialized without patterns immediately return False
        if len(self.patterns) == 0:
            return False

        outputs = input_ids[:, self.prompt_ids_length:]
        generated_sequences = self.processor.batch_decode(outputs, skip_special_tokens=False)
        
        done_sequences = []

        for sequence in generated_sequences:
            done = any([pattern in sequence for pattern in self.patterns])
            done_sequences.append(done)

        return all(done_sequences)
        
        


def post_process_stopping_patterns(prompt_truncated_generated_sequences: list[str],
                                   stopping_patterns: list[str] | tuple[str] | None) -> list[str]:
    """Post-process the outputs of a model to truncate according to a list of patterns upon which we stop
    generation (this is needed because the StoppingCriteria cannot immediately stop the generation of each
    sequence upon meeting a pattern in the case of more than 1 `num_return_sequences`).

    Parameters
    ----------
    prompt_truncated_generated_sequences : list[str]
        Decoded PROMPT-TRUNCATED outputs of a model. Passing the full decoded outputs may induce errors in the logic.
    stopping_patterns : list[str] | tuple[tr] | None,
        The list of patterns to use to stop generation.

    Returns
    -------
    list[str]
        The truncated outputs to meet the criteria of the stopping patterns.
    """

    # If there are no stopping patterns
    if stopping_patterns is None or len(stopping_patterns) == 0:
        return prompt_truncated_generated_sequences

    generated_sequences_curated = []
    
    for sequence in prompt_truncated_generated_sequences:
        
        stop_index = len(sequence)

        # Scan the sequence for each pattern, and return the minimum index such that none of the patterns are
        # in the sequence
        for pattern in stopping_patterns:
            index = sequence.find(pattern)
            if index != -1:
                stop_index = min(stop_index, index)

        curated_sequence = sequence[0:stop_index]
        generated_sequences_curated.append(curated_sequence)

    return generated_sequences_curated



def post_process_sequences(prompt_truncated_outputs: torch.Tensor, processor: ProcessorMixin | PreTrainedTokenizerBase,
                           stopping_patterns: list[str] | tuple[str] | None = IDEFICS_STOP_PATTERNS) -> list[str]:
    """Apply all steps of post-processing to the prompt-truncated outputs of a model.

    Parameters
    ----------
    prompt_truncated_outputs : torch.Tensor
        The PROMPT-TRUNCATED output of a model. Passing the full outputs may induce errors in the logic.
    processor : ProcessorMixin | PreTrainedTokenizerBase
        The processor or tokenizer used by the model.
    stopping_patterns : list[str] | tuple[tr] | None,
        The list of patterns to use to stop generation.

    Returns
    -------
    list[str]
        The post-processed generated sequences.
    """

    # Decode sequences
    prompt_truncated_sequences = processor.batch_decode(prompt_truncated_outputs, skip_special_tokens=True)
    # Truncate according to the patterns
    final_sequences = post_process_stopping_patterns(prompt_truncated_sequences, stopping_patterns)

    if len(final_sequences) == 1:
        return final_sequences[0]

    return final_sequences

