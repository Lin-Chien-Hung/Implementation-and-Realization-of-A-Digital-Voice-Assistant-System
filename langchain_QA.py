import os

# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

"""
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryMemory
from langchain.memory import ConversationBufferMemory

from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser

from langchain_community.vectorstores import FAISS
"""
# -------------載模型 start-------------

MODEL_PATH = "./model/NPL_Model/Llama3-8B-Chinese-Chat"
# model_path = './model/NPL_Model/Llama3-8B-English'


# 4 bit用量化模板
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)


# 8 bit用量化模板
"""
quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
    )
"""


model_4bit = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    attn_implementation="flash_attention_2",
    quantization_config=quantization_config,
)


# 20240416 added
# 參考: https://stackoverflow.com/questions/69609401/suppress-huggingface-logging-warning-setting-pad-token-id-to-eos-token-id
# model_4bit.generation_config.pad_token_id = model_4bit.generation_config.eos_token_id

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
tokenizer.eos_token_id = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>"),
]

llm_pipeline = pipeline(
    "text-generation",
    model=model_4bit,
    tokenizer=tokenizer,
    device_map="auto",
    torch_dtype=torch.float16,
)

llm_pipeline.model.config.pad_token_id = llm_pipeline.model.config.eos_token_id

# -------------載模型 end---------------


# -------------指令模板 start-------------

# llama-3中文模板
template_string = """
<|start_header_id|>system<|end_header_id|>

現在，你是一名智能助手，名為「人工智慧語音助理」，請直接回答使用者的問題，並使用繁體中文回答。
1.回答的過程當中不能使用條列式。
2.回復的答案中不能有任何英文字。
3.回答請在一段句子內完成，句號後面不得在生成。
注意事項:只能從你本身的知識來做回答，絕對不允許自行在答案中添加編造、偽造成分，且回復的答案中不能有任何英文字。<|eot_id|><|start_header_id|>user<|end_header_id|>

問題:{user_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""

# #llama-3英文模板
# template_string = """
# <|begin_of_text|><|start_header_id|>system<|end_header_id|>

# You are a helpful assistant named Braslab Assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>

# question:{user_input} answer:<|eot_id|><|start_header_id|>assistant<|end_header_id|>
# """


# -------------指令模板 end---------------


def LLM(pre_str):
    user_input = str(pre_str)
    messages = [
        {
            "role": "system",
            "content": """
                       現在，你是一名智能助手，名為「人工智慧語音助理」，請直接回答使用者的問題，並使用繁體中文回答。
                       1. 回答的過程當中不能使用條列的方式。
                       2. 回覆的答案中不能包含英文。
                       3. 回答請在一段句子內完成，句號後面不得在生成。
                       注意事項：只能從你本身的知識來做回答，絕對不允許自行在答案中添加編造、偽造成分。
                       """,
        },
        {"role": "user", "content": user_input},
    ]
    prompt = llm_pipeline.tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    
    result = llm_pipeline(
        prompt,
        do_sample=True,
        num_return_sequences=1,
        max_length=200,
        repetition_penalty=1.1,
        temperature=0.2,
        truncation=True,
        top_k=50,
        top_p=0.9,
    )
    result = result[0]["generated_text"][len(prompt) :]
    print("問題回復 : " + "\033[91m" + str(result) + "\033[0m")
    print("-" * 60)

    return result
