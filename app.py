import gradio as gr

def chat(message: str, history: str):
    return f"Message: {message}. History {history}"

gr.ChatInterface(chat).launch()