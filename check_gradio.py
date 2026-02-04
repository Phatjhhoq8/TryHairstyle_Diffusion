
try:
    import gradio
    print(f"Gradio version: {gradio.__version__}")
    import huggingface_hub
    print(f"HuggingFace Hub version: {huggingface_hub.__version__}")
except Exception as e:
    print(f"Error: {e}")
