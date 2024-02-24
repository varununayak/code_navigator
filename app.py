import glob
from pathlib import Path

import gradio as gr
import os

import torch
from llama_index.core import set_global_tokenizer, ServiceContext, SimpleDirectoryReader, VectorStoreIndex, Settings, \
    StorageContext, load_index_from_storage, PromptTemplate
from llama_index.legacy.llms import HuggingFaceLLM
from llama_index.core.callbacks.base import CallbackManager

from accelerate.utils import BnbQuantizationConfig


assert (torch.cuda.is_available())
_MODEL_NAME = "microsoft/phi-2"
_CODE_FILES = glob.glob(str(Path("./data/") / '*'), recursive=True)
_QUERY = "What does the python function 'foo' do?"
PERSIST_DIR = "./storage"

#OPEN_AI_KEY = "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"

#os.environ['OPENAI_API_KEY'] = OPEN_AI_KEY

class ChatBot:


    def __init__(self):
        bnb_quantization_config = BnbQuantizationConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        Settings.callback_manager = CallbackManager()
        Settings.chunk_size = 512
        llm = HuggingFaceLLM(
            context_window=2048,
            max_new_tokens=256,
            generate_kwargs={"temperature": 0.1, "do_sample": True},
            tokenizer_name=_MODEL_NAME,
            model_name=_MODEL_NAME,
            device_map="auto",
            tokenizer_kwargs={"max_length": 2048},
            # quantization_config=bnb_quantization_config,
            #query_wrapper_prompt=query_wrapper_prompt,
            # uncomment this if using CUDA to reduce memory usage
            model_kwargs={
            "torch_dtype": torch.float16}
        )
        Settings.llm = llm

        # load the documents and create the index
        documents = SimpleDirectoryReader(input_files=_CODE_FILES).load_data()
        index = VectorStoreIndex.from_documents(documents, llm=llm, embed_model='local')
        self.query_engine = index.as_query_engine(llm=llm)

    def chat(self, message: str, history: str = None):
        response = self.query_engine.query(message)
        return str(response)

chatbot = ChatBot()
gr.ChatInterface(chatbot.chat).launch(share=True)