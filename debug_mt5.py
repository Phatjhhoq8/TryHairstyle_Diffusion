
import sys
try:
    print("1. Testing sentencepiece import...")
    import sentencepiece
    print(f"   Success. Version: {sentencepiece.__version__}")
except ImportError as e:
    print(f"   Failed: {e}")

try:
    print("\n2. Testing transformers import...")
    import transformers
    print(f"   Success. Version: {transformers.__version__}")
    
    print("\n3. Testing MT5Tokenizer import...")
    from transformers import MT5Tokenizer
    print("   Success: MT5Tokenizer imported.")
except ImportError as e:
    print(f"   Failed: {e}")
except Exception as e:
    print(f"   Error: {e}")

print("\n4. Checking sys.modules...")
import sys
print(f"   sentencepiece in modules: {'sentencepiece' in sys.modules}")

try:
    print("\n5. Testing diffusers pipeline import...")
    from diffusers import DiffusionPipeline
    print("   Success: diffusers imported.")
except ImportError as e:
    print(f"   Failed diffusers: {e}")

