import glob
from pathlib import Path

import gradio as gr
import os
from llama_index.core import set_global_tokenizer, ServiceContext, SimpleDirectoryReader, VectorStoreIndex, Settings, \
    StorageContext, load_index_from_storage

_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v0.6"
_CODE_FILES = glob.glob(str(Path("/home/varun/personal/code_navigator/data/") / '*'), recursive=True)
_QUERY = "What does the python function 'foo' do?"

OPEN_AI_KEY = "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"

os.environ['OPENAI_API_KEY'] = OPEN_AI_KEY

# check if storage already exists
PERSIST_DIR = "./storage"
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
index = VectorStoreIndex.from_documents(documents)

query_engine = index.as_chat_engine()

response = query_engine.chat(_QUERY)

print(response)

response = query_engine.chat("Can you give me an example usage in python?")

print(response)

# def chat(message: str, history: str):
#     return f"Message: {message}. History {history}"

# gr.ChatInterface(chat).launch()