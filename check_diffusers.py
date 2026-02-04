
try:
    import diffusers
    print(f"Diffusers version: {diffusers.__version__}")
    from diffusers import AutoPipelineForInpainting
    print("AutoPipelineForInpainting imported successfully")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
