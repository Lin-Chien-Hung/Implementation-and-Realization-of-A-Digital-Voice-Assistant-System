"""This module is utils of text generation.
"""

from pathlib import Path
from threading import Thread

from transformers import Pipeline

MODEL_PATH = Path.joinpath(
    Path(__file__).parent.resolve(), "Llama3", "Llama3-8B-Chinese-Chat"
)
ZH_TW_PROMPT = """
               你是一名智能助手，名為「人工智慧語音助理」，請使用繁體中文回答使用者的問題。
               1. 回答格式不能為條列的方式。
               2. 回覆的答案中不能包含英文。
               3. 台灣總統現在是賴清德
               注意事項：只能從你本身的知識來做回答，絕對不允許自行在答案中添加編造、偽造成分。
               """
EN_PROMPT = """
            You are a helpful assistant named Braslab Assistant.
            """


def generate(pipeline: Pipeline, inputs: str, *args, **kwargs):
    """Generating response of 'inputs' with LLM.

    Args:
        pipeline (Pipeline): Inference pipeline.
        inputs (str): Input text.
    """
    conversation = [
        {
            "role": "system",
            "content": ZH_TW_PROMPT,
        },
        {"role": "user", "content": inputs},
    ]
    prompt = pipeline.tokenizer.apply_chat_template(
        conversation, add_generation_prompt=True, tokenize=False
    )
    thread = Thread(
        target=pipeline,
        args=(prompt,) + args,
        kwargs=kwargs,
    )
    thread.start()
