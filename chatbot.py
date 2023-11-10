import os
import queue
import copy
from concurrent.futures import ThreadPoolExecutor

from transformers import TextIteratorStreamer
import gradio as gr

from engine import IdeficsModel, Llama2ChatModel, TextContinuationStreamer, GenericConversationTemplate
from engine.template import LLAMA2_NUTRITION_SYSTEM_PROMPT
from helpers import utils

# Load both models at the beginning
IDEFICS_VERSION = 'idefics-9B'
IDEFICS = IdeficsModel(IDEFICS_VERSION, gpu_rank=0)

LLAMA2_VERSION = 'llama2-13B-chat'
LLAMA2 = Llama2ChatModel(LLAMA2_VERSION, gpu_rank=1)

# File where the valid credentials are stored
CREDENTIALS_FILE = os.path.join(utils.ROOT_FOLDER, '.gradio_login.txt')

# This will be a mapping between users and current conversation, to reload them with page reload
CACHED_CONVERSATIONS = {}

# Need to define one logger per user
LOGGERS = {}


def chat_generation(conversation: GenericConversationTemplate, prompt: str, max_new_tokens: int,
                    do_sample: bool, top_k: int | None, top_p: float | None,
                    temperature: float) -> tuple[GenericConversationTemplate, str, list[list[str, str]]]:
    """Chat generation with streamed output.

    Parameters
    ----------
    prompt : str
        Prompt to the model.
    conversation : GenericConversation
        Current conversation. This is the value inside a gr.State instance.
    max_new_tokens : int, optional
        Maximum new tokens to generate, by default 60

    Yields
    ------
    Iterator[tuple[GenericConversation, str, list[list[str, str]]]]
        Corresponds to the tuple of components (conversation, prompt, output)
    """

    timeout = 20

    # To show text as it is being generated
    streamer = TextIteratorStreamer(LLAMA2.tokenizer, skip_prompt=True, timeout=timeout, skip_special_tokens=True)

    conv_copy = copy.deepcopy(conversation)
    conv_copy.append_user_message(prompt)
    
    # We need to launch a new thread to get text from the streamer in real-time as it is being generated. We
    # use an executor because it makes it easier to catch possible exceptions
    with ThreadPoolExecutor(max_workers=1) as executor:
        # This will update `conversation` in-place
        future = executor.submit(LLAMA2.generate_conversation, prompt, conv_history=conversation,
                                 max_new_tokens=max_new_tokens, do_sample=do_sample, top_k=top_k, top_p=top_p,
                                 temperature=temperature, seed=None, stopping_patterns=None,
                                 truncate_if_conv_too_long=True, streamer=streamer)
        
        # Get results from the streamer and yield it
        try:
            generated_text = ''
            for new_text in streamer:
                generated_text += new_text
                # Update model answer (on a copy of the conversation) as it is being generated
                conv_copy.model_history_text[-1] = generated_text
                # The first output is an empty string to clear the input box, the second is the format output
                # to use in a gradio chatbot component
                yield conversation, '', conv_copy.to_gradio_format()

        # If for some reason the queue (from the streamer) is still empty after timeout, we probably
        # encountered an exception
        except queue.Empty:
            e = future.exception()
            if e is not None:
                raise gr.Error(f'The following error happened during generation: {repr(e)}')
            else:
                raise gr.Error(f'Generation timed out (no new tokens were generated after {timeout} s)')
    
    
    # Update the chatbot with the real conversation (which may be slightly different due to postprocessing)
    yield conversation, '', conversation.to_gradio_format()



def continue_generation(conversation: GenericConversationTemplate, additional_max_new_tokens: int,
                        do_sample: bool, top_k: int | None, top_p: float | None,
                        temperature: float) -> tuple[GenericConversationTemplate, list[list[str, str]]]:
    """Continue the last turn of the conversation, with streamed output.

    Parameters
    ----------
    conversation : GenericConversation
        Current conversation. This is the value inside a gr.State instance.
    max_new_tokens : int, optional
        Maximum new tokens to generate, by default 60

    Yields
    ------
    Iterator[tuple[GenericConversation, list[list[str, str]]]]
        Corresponds to the tuple of components (conversation, output)
    """
   
    timeout = 20

    # To show text as it is being generated
    streamer = TextContinuationStreamer(LLAMA2.tokenizer, skip_prompt=True, timeout=timeout, skip_special_tokens=True)

    conv_copy = copy.deepcopy(conversation)
    
    # We need to launch a new thread to get text from the streamer in real-time as it is being generated. We
    # use an executor because it makes it easier to catch possible exceptions
    with ThreadPoolExecutor(max_workers=1) as executor:
        # This will update `conversation` in-place
        future = executor.submit(LLAMA2.continue_last_conversation_turn, conv_history=conversation,
                                 max_new_tokens=additional_max_new_tokens, do_sample=do_sample, top_k=top_k,
                                 top_p=top_p, temperature=temperature, seed=None, stopping_patterns=None,
                                 truncate_if_conv_too_long=True, streamer=streamer)
        
        # Get results from the streamer and yield it
        try:
            generated_text = conv_copy.model_history_text[-1]
            for new_text in streamer:
                generated_text += new_text
                # Update model answer (on a copy of the conversation) as it is being generated
                conv_copy.model_history_text[-1] = generated_text
                # The first output is an empty string to clear the input box, the second is the format output
                # to use in a gradio chatbot component
                yield conversation, '', conv_copy.to_gradio_format()

        # If for some reason the queue (from the streamer) is still empty after timeout, we probably
        # encountered an exception
        except queue.Empty:
            e = future.exception()
            if e is not None:
                raise gr.Error(f'The following error happened during generation: {repr(e)}')
            else:
                raise gr.Error(f'Generation timed out (no new tokens were generated after {timeout} s)')
    
    
    # Update the chatbot with the real conversation (which may be slightly different due to postprocessing)
    yield conversation, conversation.to_gradio_format()



def authentication(username: str, password: str) -> bool:
    """Simple authentication method.

    Parameters
    ----------
    username : str
        The username provided.
    password : str
        The password provided.

    Returns
    -------
    bool
        Return True if both the username and password match some credentials stored in `CREDENTIALS_FILE`. 
        False otherwise.
    """

    with open(CREDENTIALS_FILE, 'r') as file:
        # Read lines and remove whitespaces
        lines = [line.strip() for line in file.readlines() if line.strip() != '']

    valid_usernames = lines[0::2]
    valid_passwords = lines[1::2]

    if username in valid_usernames:
        index = valid_usernames.index(username)
        # Check that the password also matches at the corresponding index
        if password == valid_passwords[index]:
            return True
    
    return False
    

def clear_chatbot(username: str) -> tuple[GenericConversationTemplate, str, str, list[list[str]]]:
    """Erase the conversation history and reinitialize the elements.

    Parameters
    ----------
    username : str
        The username of the current session.

    Returns
    -------
    tuple[GenericConversation, str, str, list[list[str]]]
        Corresponds to the tuple of components (conversation, conv_id, output)
    """

    # Create new global conv object (we need a new unique id)
    conversation = LLAMA2.get_empty_conversation(system_prompt=LLAMA2_NUTRITION_SYSTEM_PROMPT)
    # Cache value
    CACHED_CONVERSATIONS[username] = conversation
    return conversation, conversation.id, conversation.to_gradio_format()



def loading(request: gr.Request) -> tuple[GenericConversationTemplate, str, str, str, str, list[list[str]]]:
    """Retrieve username and all cached values at load time, and set the elements to the correct values.

    Parameters
    ----------
    request : gr.Request
        Request sent to the app.

    Returns
    -------
    tuple[GenericConversation, str, str, str, str, list[list[str]]]
        Corresponds to the tuple of components (conversation, conv_id, username, output)
    """

    # Retrieve username
    if request is not None:
        username = request.username
    else:
        raise RuntimeError('Impossible to find username on startup.')
    
    # Check if we have cached a value for the conversation to use
    if username in CACHED_CONVERSATIONS.keys():
        actual_conv = CACHED_CONVERSATIONS[username]
    else:
        actual_conv = LLAMA2.get_empty_conversation(system_prompt=LLAMA2_NUTRITION_SYSTEM_PROMPT)
        CACHED_CONVERSATIONS[username] = actual_conv
        LOGGERS[username] = gr.CSVLogger()

    conv_id = actual_conv.id
    
    return actual_conv, conv_id, username, actual_conv.to_gradio_format()
    



# Define generation parameters and model selection
max_new_tokens = gr.Slider(32, 4096, value=512, step=32, label='Max new tokens',
                           info='Maximum number of new tokens to generate.')
max_additional_new_tokens = gr.Slider(16, 512, value=128, step=16, label='Max additional new tokens',
                           info='Maximum number of new tokens to generate when using "Continue last answer" feature.')
do_sample = gr.Checkbox(value=True, label='Sampling', info=('Whether to incorporate randomness in generation. '
                                                            'If not selected, perform greedy search.'))
top_k = gr.Slider(0, 200, value=50, step=5, label='Top-k',
               info='How many tokens with max probability to consider. 0 to deactivate.')
top_p = gr.Slider(0, 1, value=0.90, step=0.01, label='Top-p',
              info='Probability density threshold for new tokens. 1 to deactivate.')
temperature = gr.Slider(0, 1, value=0.8, step=0.01, label='Temperature',
                        info='How to cool down the probability distribution.')


# Define elements of the chatbot Tab
prompt = gr.Textbox(placeholder='Write your prompt here.', label='Prompt', lines=2)
output = gr.Chatbot(label='Conversation')
generate_button = gr.Button('Generate text', variant='primary')
continue_button = gr.Button('Continue last answer', variant='primary')
clear_button = gr.Button('Clear conversation')


# State variable to keep one conversation per session (default value does not matter here -> it will be set
# by loading() method anyway)
conversation = gr.State(LLAMA2.get_empty_conversation())


# Define NON-VISIBLE elements: they are only used to keep track of variables and save them to the callback.
username = gr.Textbox('', label='Username', visible=False)
conv_id = gr.Textbox('', label='Conversation id', visible=False)


# Define the inputs for the main inference
inputs_to_generation = [conversation, prompt, max_new_tokens, do_sample, top_k, top_p, temperature]

inputs_to_continuation = [conversation, max_additional_new_tokens, do_sample, top_k, top_p, temperature]

# Define inputs for the logging callbacks
inputs_to_callback = [max_new_tokens, max_additional_new_tokens, do_sample, top_k, top_p, temperature,
                      output, conv_id, username]


# Some prompt examples
prompt_examples = [
    "Please write a function to multiply 2 numbers `a` and `b` in Python.",
    "Hello, what's your name?",
    "What's the meaning of life?",
    "How can I write a Python function to generate the nth Fibonacci number?",
    ("Here is my data {'Name':['Tom', 'Brad', 'Kyle', 'Jerry'], 'Age':[20, 21, 19, 18], 'Height' :"
     " [6.1, 5.9, 6.0, 6.1]}. Can you provide Python code to plot a bar graph showing the height of each person?"),
]


demo = gr.Blocks(title='Nutrition Chatbot')

with demo:

    # State variable
    conversation.render()

    # Variables we track with usual components: they do not need to be State variables -- will not be visible
    conv_id.render()
    username.render()

    # Need to wrap everything in a row because we want two side-by-side columns
    with gr.Row():

        with gr.Column(scale=5):

            prompt.render()
            with gr.Row():
                generate_button.render()
                clear_button.render()
                continue_button.render()
            output.render()

            gr.Markdown("### Prompt Examples")
            gr.Examples(prompt_examples, inputs=prompt)

        # Second column defines model selection and generation parameters
        with gr.Column(scale=1):
            
            # Accordion for generation parameters
            with gr.Accordion("Text generation parameters", open=False):
                do_sample.render()
                with gr.Row():
                    max_new_tokens.render()
                    max_additional_new_tokens.render()
                with gr.Row():
                    top_k.render()
                    top_p.render()
                with gr.Row():
                    temperature.render()


    # Perform chat generation when clicking the button
    generate_event1 = generate_button.click(chat_generation, inputs=inputs_to_generation,
                                            outputs=[conversation, prompt, output])

    # Add automatic callback on success (args[-1] is the username)
    generate_event1.success(lambda *args: LOGGERS[args[-1]].flag(args, flag_option=f'generation'),
                            inputs=inputs_to_callback, preprocess=False)
    
    # Continue generation when clicking the button
    generate_event2 = continue_button.click(continue_generation, inputs=inputs_to_continuation,
                                            outputs=[conversation, output])
    
    # Add automatic callback on success (args[-1] is the username)
    generate_event2.success(lambda *args: LOGGERS[args[-1]].flag(args, flag_option=f'continuation'),
                            inputs=inputs_to_callback, preprocess=False)
    
    # Clear the output box when clicking the button
    clear_button.click(clear_chatbot, inputs=[username], outputs=[conversation, conv_id, output])
    
    # Correctly set all variables and callback at load time
    loading_events = demo.load(loading, outputs=[conversation, conv_id, username, output])
    loading_events.then(lambda username: LOGGERS[username].setup(inputs_to_callback, flagging_dir=f'chatbot_logs/{username}'),
                        inputs=username)


if __name__ == '__main__':
    demo.queue(concurrency_count=4).launch(share=True, auth=authentication, blocked_paths=[CREDENTIALS_FILE])
