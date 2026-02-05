
path = r"backend/models/stable-diffusion/sd15/unet/quarantine/diffusion_pytorch_model.safetensors"
try:
    with open(path, "rb") as f:
        header = f.read(100)
        print(f"First 100 bytes: {header}")
        try:
            print(f"Decoded: {header.decode('utf-8', errors='ignore')}")
        except:
            pass
except Exception as e:
    print(f"Error: {e}")
