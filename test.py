
import tempfile
from PIL import Image

import gradio as gr


def upload_file(file: tempfile.TemporaryFile):
    # file_paths = [file.name for file in files]
    # return file_paths
    image = Image.open(file.name).convert('RGB')
    return image


with gr.Blocks() as demo:
    # file_output = gr.Markdown()
    file_output = gr.Image()
    upload_button = gr.UploadButton("Click to Upload a File", file_types=["image"])
    upload_button.upload(upload_file, upload_button, file_output)

demo.queue(default_concurrency_limit=4).launch(server_name='127.0.0.1', server_port=7875)

	