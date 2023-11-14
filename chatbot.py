import os
import queue
import copy
import tempfile
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

import torch
from transformers import TextIteratorStreamer
import gradio as gr

from engine import IdeficsModel, Llama2ChatModel, TextContinuationStreamer, GenericConversationTemplate
from engine.model import DummyModel
from engine.template import LLAMA2_USER_TRANSITION, LLAMA2_MODEL_TRANSITION, parse_idefics_output, get_custom_system_prompt, get_fake_turn
from helpers import utils

# Load both models at the beginning
IDEFICS_VERSION = 'idefics-9B'
IDEFICS = IdeficsModel(IDEFICS_VERSION, gpu_rank=0) if torch.cuda.is_available() else DummyModel()

LLAMA2_VERSION = 'llama2-13B-chat'
LLAMA2 = Llama2ChatModel(LLAMA2_VERSION, gpu_rank=1) if torch.cuda.is_available() else DummyModel()

# File where the valid credentials are stored
CREDENTIALS_FILE = os.path.join(utils.ROOT_FOLDER, '.gradio_login.txt')

# This will be a mapping between users and current conversation, to reload them with page reload
CACHED_CONVERSATIONS = {}
# This will be a mapping between users and current outputs, to reload them with page reload
CACHED_OUTPUTS = {}
# This will be a mapping between users and current initial question answers, to reload them with page reload
CACHED_QUESTION_ANSWERS = {}

# Need to define one logger per user
LOGGERS = {}

# Path to NutriBot thumbnail
THUMBNAIL = os.path.join(utils.ROOT_FOLDER, 'avatars', 'nutribot_cropped.png')


def chat_generation(conversation: GenericConversationTemplate, gradio_output: list[list], prompt: str,
                    max_new_tokens: int, do_sample: bool, top_k: int, top_p: float, temperature: float,
                    append_prompt_to_gradio: bool = True) -> tuple[GenericConversationTemplate, str, list[list], list[list]]:
    """Chat generation with streamed output.

    Parameters
    ----------
    conversation : GenericConversation
        Current conversation. This is the value inside a gr.State instance.
    gradio_output : list[list]
        Current conversation to be displayed. This is the value inside a gr.State instance.
    prompt : str
        Prompt to the model.
    max_new_tokens : int
        Maximum new tokens to generate.
    do_sample : bool
        Whether to introduce randomness in the generation.
    top_k : int
        How many tokens with max probability to consider for randomness.
    top_p : float
        The probability density covering the new tokens to consider for randomness.
    temperature : float
        How to cool down the probability distribution. Value between 1 (no cooldown) and 0 (greedy search,
        no randomness).
    append_prompt_to_gradio : bool, optional
        Whether to append the prompt to the gradio output, by default True.

    Yields
    ------
    Iterator[tuple[GenericConversation, str, list[list]
        Corresponds to the tuple of components (conversation, prompt, chatbot, gradio_output)
    """

    timeout = 20

    if not torch.cuda.is_available():
        out = LLAMA2.generate_conversation(prompt, conv_history=conversation)
        gradio_output.append(conversation.get_last_turn())
        print(gradio_output)
        return conversation, '', gradio_output, gradio_output

    # To show text as it is being generated
    streamer = TextIteratorStreamer(LLAMA2.tokenizer, skip_prompt=True, timeout=timeout, skip_special_tokens=True)

    output_copy = copy.deepcopy(gradio_output)
    if append_prompt_to_gradio:
        output_copy.append([prompt, None])
    
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
                output_copy[-1][1] = generated_text
                # The first output is an empty string to clear the input box, the second is the format output
                # to use in a gradio chatbot component
                yield conversation, '', output_copy, output_copy

        # If for some reason the queue (from the streamer) is still empty after timeout, we probably
        # encountered an exception
        except queue.Empty:
            e = future.exception()
            if e is not None:
                raise gr.Error(f'The following error happened during generation: {repr(e)}')
            else:
                raise gr.Error(f'Generation timed out (no new tokens were generated after {timeout} s)')
            
    # Update the component with the final value
    if append_prompt_to_gradio:
        gradio_output.append(conversation.get_last_turn())
    else:
        gradio_output[-1][1] = conversation.model_history_text[-1]
    
    # Update the chatbot with the real conversation (which may be slightly different due to postprocessing)
    yield conversation, '', gradio_output, gradio_output



def continue_generation(conversation: GenericConversationTemplate, gradio_output: list[list],
                        additional_max_new_tokens: int, do_sample: bool, top_k: int, top_p: float,
                        temperature: float) -> tuple[GenericConversationTemplate, list[list[str, str]]]:
    """Continue the last turn of the conversation, with streamed output.

    Parameters
    ----------
    conversation : GenericConversation
        Current conversation. This is the value inside a gr.State instance.
    gradio_output : list[list]
        Current conversation to be displayed. This is the value inside a gr.State instance.
    additional_max_new_tokens : int
        Maximum new tokens to generate.
    do_sample : bool
        Whether to introduce randomness in the generation.
    top_k : int
        How many tokens with max probability to consider for randomness.
    top_p : float
        The probability density covering the new tokens to consider for randomness.
    temperature : float
        How to cool down the probability distribution. Value between 1 (no cooldown) and 0 (greedy search,
        no randomness).

    Yields
    ------
    Iterator[tuple[GenericConversation, list[list[str, str]]]]
        Corresponds to the tuple of components (conversation, chatbot, gradio_output)
    """

    # If we just uploaded an image, do nothing
    if conversation.user_history_text[-1].startswith(LLAMA2_USER_TRANSITION) and \
        conversation.model_history_text[-1].startswith(LLAMA2_MODEL_TRANSITION):
        return conversation, gradio_output, gradio_output
   
    timeout = 20

    # To show text as it is being generated
    streamer = TextContinuationStreamer(LLAMA2.tokenizer, skip_prompt=True, timeout=timeout, skip_special_tokens=True)

    output_copy = copy.deepcopy(gradio_output)
    
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
            generated_text = output_copy[-1][1]
            for new_text in streamer:
                generated_text += new_text
                # Update model answer (on a copy of the conversation) as it is being generated
                output_copy[-1][1] = generated_text
                # The first output is an empty string to clear the input box, the second is the format output
                # to use in a gradio chatbot component
                yield conversation, output_copy, output_copy

        # If for some reason the queue (from the streamer) is still empty after timeout, we probably
        # encountered an exception
        except queue.Empty:
            e = future.exception()
            if e is not None:
                raise gr.Error(f'The following error happened during generation: {repr(e)}')
            else:
                raise gr.Error(f'Generation timed out (no new tokens were generated after {timeout} s)')
    
    # Update the component with the final value
    gradio_output[-1][1] = conversation.model_history_text[-1]
    
    # Update the chatbot with the real conversation (which may be slightly different due to postprocessing)
    yield conversation, gradio_output, gradio_output



def upload_image(file: tempfile.TemporaryFile, conversation: GenericConversationTemplate,
                 gradio_output: list[list]) -> tuple[GenericConversationTemplate, list[list], list[list]]:
    """Load the uploaded image, process it, and feed output to Llama2.

    Parameters
    ----------
    file : tempfile.TemporaryFile
        The file as returned by the UploadButton.

    Returns
    -------
        Corresponds to the tuple of components (conversation, chatbot, gradio_output)
    """

    image = Image.open(file.name). convert('RGB')

    try:
        out = IDEFICS.process_image(image)
        parsed_output = parse_idefics_output(out)
    except BaseException as e:
        raise gr.Error(f'The following error happened during image processing: {repr(e)}. Please choose another image.')

    if parsed_output['is_food']:
        # user_turn, model_turn = get_fake_turn(parsed_output, LLAMA2_USER_TRANSITION, LLAMA2_MODEL_TRANSITION)
        # conversation.append_user_message(user_turn)
        # conversation.append_model_message(model_turn)
        # gradio_output.append([(file.name,), model_turn])

        gradio_output.append([(file.name,), None])
        user_turn, _ = get_fake_turn(parsed_output, LLAMA2_USER_TRANSITION, LLAMA2_MODEL_TRANSITION)
        yield conversation, '', gradio_output, gradio_output
        yield from chat_generation(conversation, gradio_output, user_turn, 512, True, 50, 0.9, 0.8, False)
    else:
        gr.Warning("The image you just uploaded does not depict food. We only allow images of meals or "
                   "beverages.")
        
    yield conversation, '', gradio_output, gradio_output



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
    


def clear_chatbot(username: str) -> tuple[GenericConversationTemplate, str, list[list[str]]]:
    """Erase the conversation history and reinitialize the elements.

    Parameters
    ----------
    username : str
        The username of the current session.

    Returns
    -------
    tuple[GenericConversation, str, list[list[str]]]
        Corresponds to the tuple of components (conversation, conv_id, chatbot, gradio_output)
    """

    # Get the system prompt from the cached medical conditions
    medical_conditions = CACHED_QUESTION_ANSWERS[username]
    system_prompt = get_custom_system_prompt(medical_conditions)
    
    # Create the new objects
    conversation = LLAMA2.get_empty_conversation(system_prompt=system_prompt)
    gradio_output = []
    # Cache value
    CACHED_CONVERSATIONS[username] = conversation
    CACHED_OUTPUTS[username] = gradio_output

    return conversation, conversation.id, gradio_output, gradio_output



def loading(request: gr.Request) -> tuple[GenericConversationTemplate, str, str, list[list], list[list], dict, dict]:
    """Retrieve username and all cached values at load time, and set the elements to the correct values.

    Parameters
    ----------
    request : gr.Request
        Request sent to the app.

    Returns
    -------
    tuple[GenericConversation, str, str, list[list[str]], dict, dict]
        Corresponds to the tuple of components (conversation, conv_id, username, chatbot, gradio_output, initial_ui, main_ui)
    """

    # Retrieve username
    if request is not None:
        username = request.username
    else:
        raise RuntimeError('Impossible to find username on startup.')
    
    # Check if we have cached a value for the conversation to use
    if username in CACHED_CONVERSATIONS.keys():
        actual_conv = CACHED_CONVERSATIONS[username]
        actual_output = CACHED_OUTPUTS[username]

        medical_conditions = CACHED_QUESTION_ANSWERS[username]
        assert actual_conv.system_prompt == get_custom_system_prompt(medical_conditions), 'Error in loading past conversation'
        main_ui = True
    else:
        # System prompt does not matter here, ot will be set after the questions have been answered
        actual_conv = LLAMA2.get_empty_conversation()
        actual_output = []
        CACHED_CONVERSATIONS[username] = actual_conv
        CACHED_OUTPUTS[username] = actual_output
        LOGGERS[username] = gr.CSVLogger()
        main_ui = False

    conv_id = actual_conv.id
    
    return actual_conv, conv_id, username, actual_output, actual_output, gr.update(visible=not main_ui), gr.update(visible=main_ui)
 

def validate_questions(username: str, conversation: GenericConversationTemplate, age: int | None, size: int | None,
                       weight: float | None, sex: str | None,
                       conditions: str | None) -> tuple[GenericConversationTemplate, dict, dict]:
    """Validate the initial question answers and set the conversation system prompt accordingly. Change the
    UI to the main UI.

    Parameters
    ----------
    username : str
        The username of the current session.
    conversation : GenericConversation
        Current conversation. This is the value inside a gr.State instance.
    age : int | None
        Age of the user.
    size : int | None
        Size of the user.
    weight : float | None
        Weight of the user.
    sex : str | None
        Sex of the user.
    conditions : str | None
        Conditions of the user.

    Returns
    -------
    tuple[GenericConversationTemplate, dict, dict]
        Corresponds to the tuple of components (conversation, initial_ui, main_ui)
    """

    missing = []
    if age is None:
        missing.append('Age')
    if size is None:
        missing.append('Size')
    if weight is None:
        missing.append('Weight')
    if sex is None:
        missing.append('Sex')

    if len(missing) > 0:
        raise gr.Error(f'You must still specify {*missing,}')
    
    if conditions is None:
        conditions = ''
    
    medical_conditions = {'age': age, 'size': size, 'weight': weight, 'sex': sex,
                                         'conditions': conditions}
    CACHED_QUESTION_ANSWERS[username] = medical_conditions

    system_prompt = get_custom_system_prompt(medical_conditions)
    conversation.set_system_prompt(system_prompt)
    
    return conversation, gr.update(visible=False), gr.update(visible=True)
    


# Define generation parameters and model selection
max_new_tokens = gr.Slider(32, 4096, value=512, step=32, label='Max new tokens',
                           info='Maximum number of new tokens to generate.')
max_additional_new_tokens = gr.Slider(16, 512, value=128, step=16, label='Max additional new tokens',
                           info='Maximum number of new tokens to generate when using "Continue last answer" feature.')
do_sample = gr.Checkbox(value=True, label='Random sampling', info=('Whether to incorporate randomness in generation. '
                                                                   'If not selected, perform greedy search.'))
top_k = gr.Slider(0, 200, value=50, step=5, label='Top-k',
               info='How many tokens with max probability to consider. 0 to deactivate.')
top_p = gr.Slider(0, 1, value=0.90, step=0.01, label='Top-p',
              info='Probability density threshold for new tokens. 1 to deactivate.')
temperature = gr.Slider(0, 1, value=0.8, step=0.01, label='Temperature',
                        info='How to cool down the probability distribution.')


# Define elements of the chatbot
prompt = gr.Textbox(placeholder='Write your prompt here.', label='Prompt', lines=2)
chatbot = gr.Chatbot(label='NutriBot', height=500, avatar_images=(None, THUMBNAIL))
generate_button = gr.Button('▶️ Submit', variant='primary')
continue_button = gr.Button('🔂 Continue last answer', variant='primary')
clear_button = gr.Button('🗑 Clear conversation')
upload_button = gr.UploadButton("📁 Upload image", file_types=['image'], variant='primary')


# Elements of the initial questions
age = gr.Number(value=30, label='Age', precision=0, minimum=2, maximum=120)
size = gr.Number(value=160, label='Size (cm)', precision=0, minimum=20, maximum=240)
weight = gr.Number(value=56, label='Weight (kg)', precision=None, minimum=3, maximum=350, step=0.1)
sex = gr.Radio(choices=['male', 'female'], value=None, label='Sex', scale=1)
conditions = gr.Textbox(label='Special conditions', placeholder='E.g. diabetes, food allergy...', scale=3)
validate_button = gr.Button('Validate answers', variant='primary')


# State variable to keep one conversation per session (default value does not matter here -> it will be set
# by loading() method anyway)
conversation = gr.State(LLAMA2.get_empty_conversation())
# This is a duplicate of the chatbot value, to be able to reload it if we reload the page
gradio_output = gr.State([])


# Define NON-VISIBLE elements: they are only used to keep track of variables and save them to the callback.
username = gr.Textbox('', label='Username', visible=False)
conv_id = gr.Textbox('', label='Conversation id', visible=False)


# Define the inputs for the main inference
inputs_to_generation = [conversation, gradio_output, prompt, max_new_tokens, do_sample, top_k, top_p, temperature]

inputs_to_continuation = [conversation, gradio_output, max_additional_new_tokens, do_sample, top_k, top_p, temperature]

# Define inputs for the logging callbacks
inputs_to_callback = [max_new_tokens, max_additional_new_tokens, do_sample, top_k, top_p, temperature,
                      chatbot, conv_id, username]

# Define inputs to initial questions
inputs_to_questions = [age, size, weight, sex, conditions]


# Define "fake" columns to easily make all of their internal components visible/hidden
initial_ui = gr.Column(visible=True)
main_ui = gr.Column(visible=False)


# Some prompt examples
prompt_examples = [
    "Hello, who are you?",
    "How healthy is a cheeseburger?",
    "Any ideas for a healthy meal for tonight?",
]



demo = gr.Blocks(title='Nutrition Chatbot')

with demo:

    # State variable
    conversation.render()
    gradio_output.render()

    # Variables we track with usual components: they do not need to be State variables -- will not be visible
    conv_id.render()
    username.render()

    # Visible UI
    # Starts by displaying image and text
    gr.Markdown('# <center>NutriBot: your nutritionist assistant</center>')
    with gr.Row(variant='panel'):
        with gr.Column(scale=1):
            gr.Image(THUMBNAIL, show_label=False, show_download_button=False, container=True)
        with gr.Column(scale=5):
            gr.Markdown(
                """
                ### This demo showcases **NutriBot**, a nutritionist assistant chatbot.  
                It can answer all your questions, and will provide personalized advices based on what you tell him.
                You can also upload food or beverage images, and it will automatically recognize what are on those
                images.  
                   
                   
                ⛔️ **Limitations:** This chatbot is not an authorized medical tool, and should not be used as such.
                    Its responses should not be considered as medical advice. If you have a specific medical condition,
                    it is always best to consult with a qualified healthcare professional for personalized advice.  
                """
                )
    
    # Fake column to group the initial questions UI inside a single entity to eqsily activate/deactivate visibility
    with initial_ui.render():
        gr.Markdown("### To start with the chatbot, please begin by answering the following questions for a better and more customized service:")
        # Initial questions
        with gr.Row():
            age.render()
            size.render()
            weight.render()
        with gr.Row():
            sex.render()
            conditions.render()
        validate_button.render()

    # Fake column to group the main UI inside a single entity to easily activate/deactivate visibility
    with main_ui.render():

        chatbot.render()
        prompt.render()

        with gr.Row():
            generate_button.render()
            upload_button.render()
            continue_button.render()
            clear_button.render()

        # Accordion for generation parameters
        with gr.Accordion("Text generation parameters", open=False):
            do_sample.render()
            with gr.Group():
                max_new_tokens.render()
                max_additional_new_tokens.render()
            with gr.Group():
                top_k.render()
                top_p.render()
                temperature.render()

        gr.Markdown("### Prompt Examples")
        gr.Examples(prompt_examples, inputs=prompt)

    # Validate the initial questions
    validate_button.click(validate_questions, inputs=[username, conversation, *inputs_to_questions],
                          outputs=[conversation, initial_ui, main_ui], queue=False)

    # Perform chat generation when clicking the button
    generate_event1 = generate_button.click(chat_generation, inputs=inputs_to_generation,
                                            outputs=[conversation, prompt, chatbot, gradio_output])

    # Add automatic callback on success (args[-1] is the username)
    generate_event1.success(lambda *args: LOGGERS[args[-1]].flag(args, flag_option=f'generation'),
                            inputs=inputs_to_callback, preprocess=False)
    
    # Continue generation when clicking the button
    generate_event2 = continue_button.click(continue_generation, inputs=inputs_to_continuation,
                                            outputs=[conversation, chatbot, gradio_output])
    
    # Add automatic callback on success (args[-1] is the username)
    generate_event2.success(lambda *args: LOGGERS[args[-1]].flag(args, flag_option=f'continuation'),
                            inputs=inputs_to_callback, preprocess=False)
    
    # Load an image to the image component
    # upload_event = upload_button.upload(upload_image, inputs=[upload_button, conversation, gradio_output],
    #                                     outputs=[conversation, chatbot, gradio_output],
    #                                     cancels=[generate_event1, generate_event2])

    upload_event = upload_button.upload(upload_image, inputs=[upload_button, conversation, gradio_output],
                                        outputs=[conversation, prompt, chatbot, gradio_output],
                                        cancels=[generate_event1, generate_event2])
    
    # Add automatic callback on success (args[-1] is the username)
    upload_event.success(lambda *args: LOGGERS[args[-1]].flag(args, flag_option=f'image_upload'),
                         inputs=inputs_to_callback, preprocess=False)
    
    # Clear the chatbot box when clicking the button
    clear_button.click(clear_chatbot, inputs=[username], outputs=[conversation, conv_id, chatbot, gradio_output],
                       queue=False)
    
    # Correctly set all variables and callback at load time
    loading_events = demo.load(loading, outputs=[conversation, conv_id, username, chatbot, gradio_output, initial_ui, main_ui],
                               queue=False)
    loading_events.then(lambda username: LOGGERS[username].setup(inputs_to_callback, flagging_dir=f'chatbot_logs/{username}'),
                        inputs=username, queue=False)
    
    # Change visibility of generation parameters if we perform greedy search
    do_sample.input(lambda value: [gr.update(visible=value) for _ in range(3)], inputs=do_sample,
                    outputs=[top_k, top_p, temperature], queue=False)


if __name__ == '__main__':
    demo.queue(concurrency_count=4).launch(share=True, auth=authentication, blocked_paths=[CREDENTIALS_FILE])
