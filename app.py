import glob
from pathlib import Path

import gradio as gr
import os
from llama_index.core import set_global_tokenizer, ServiceContext, SimpleDirectoryReader, VectorStoreIndex, Settings, \
    StorageContext, load_index_from_storage, PromptTemplate
from llama_index.legacy.llms import HuggingFaceLLM

_MODEL_NAME = "mistralai/Mixtral-8x7B-Instruct-v0.1"
_CODE_FILES = glob.glob(str(Path("/home/varun/personal/code_navigator/data/") / '*'), recursive=True)
_QUERY = "What does the python function 'foo' do?"
PERSIST_DIR = "./storage"

#OPEN_AI_KEY = "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"

#os.environ['OPENAI_API_KEY'] = OPEN_AI_KEY

class ChatBot:


    def __init__(self):
        query_wrapper_prompt = PromptTemplate(
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n{query_str}\n\n### Response:"
        )
        llm = HuggingFaceLLM(
            context_window=2048,
            max_new_tokens=256,
            generate_kwargs={"temperature": 0.25, "do_sample": False},
            tokenizer_name=_MODEL_NAME,
            model_name=_MODEL_NAME,
            device_map="auto",
            tokenizer_kwargs={"max_length": 2048},
            # query_wrapper_prompt=query_wrapper_prompt,
            # uncomment this if using CUDA to reduce memory usage
            # model_kwargs={"torch_dtype": torch.float16}
        )
        Settings.chunk_size = 512
        Settings.llm = llm

        # load the documents and create the index
        documents = SimpleDirectoryReader(input_files=_CODE_FILES).load_data()
        index = VectorStoreIndex.from_documents(documents, llm=llm, embed_model='local')
        # store it for later
        index.storage_context.persist(persist_dir=PERSIST_DIR)

        self.query_engine = index.as_chat_engine(llm=llm)

    def chat(self, message: str, history: str = None):
        response = self.query_engine.chat(message)
        return str(response)

chatbot = ChatBot()
print(chatbot.chat("Hello"))
gr.ChatInterface(chatbot.chat).launch()