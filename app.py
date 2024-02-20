import glob
from pathlib import Path

import gradio as gr
import os
from llama_index.core import set_global_tokenizer, ServiceContext, SimpleDirectoryReader, VectorStoreIndex, Settings, \
    StorageContext, load_index_from_storage

_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v0.6"
_CODE_FILES = glob.glob(str(Path("/home/varun/personal/code_navigator/data/") / '*'), recursive=True)
_QUERY = "What does the python function 'foo' do?"
PERSIST_DIR = "./storage"

OPEN_AI_KEY = "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"

os.environ['OPENAI_API_KEY'] = OPEN_AI_KEY

class ChatBot:


    def __init__(self):
        # check if storage already exists
        if not os.path.exists(PERSIST_DIR):
            print("Creating Index")
            # load the documents and create the index
            documents = SimpleDirectoryReader(input_files=_CODE_FILES).load_data()
            index = VectorStoreIndex.from_documents(documents)
            # store it for later
            index.storage_context.persist(persist_dir=PERSIST_DIR)
        else:
            # load the existing index
            storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
            index = load_index_from_storage(storage_context)
            print("Using stored index")


        documents = SimpleDirectoryReader(input_files=_CODE_FILES).load_data()
        self.index = VectorStoreIndex.from_documents(documents)


    def chat(self, message: str, history: str):
        query_engine = self.index.as_chat_engine()
        response = query_engine.chat(message)
        return str(response)

chatbot = ChatBot()
gr.ChatInterface(chatbot.chat).launch()