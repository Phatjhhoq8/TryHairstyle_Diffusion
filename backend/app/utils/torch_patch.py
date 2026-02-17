import torch

print("[torch_patch] 1. Patching torch.xpu (Intel XPU)...", flush=True)
if not hasattr(torch, 'xpu') or not hasattr(torch.xpu, 'is_available'):
    class _DummyXPU:
        @staticmethod
        def is_available():
            return False
        @staticmethod
        def device_count():
            return 0
        @staticmethod
        def empty_cache():
            pass
        @staticmethod
        def synchronize():
            pass
        @staticmethod
        def current_device():
            return 0
        @staticmethod
        def memory_allocated(device=None):
            return 0
        @staticmethod
        def max_memory_allocated(device=None):
            return 0
        @staticmethod
        def manual_seed(seed):
            pass
        @staticmethod
        def set_rng_state(state, device=None):
            pass
        @staticmethod
        def get_rng_state(device=None):
            return None
        @staticmethod
        def reset_peak_memory_stats(device=None):
            pass
    torch.xpu = _DummyXPU()

print("[torch_patch] 2. Patching is_flash_attention_available...", flush=True)
if not hasattr(torch.backends.cuda, 'is_flash_attention_available'):
    torch.backends.cuda.is_flash_attention_available = lambda: False

print("[torch_patch] 3. Patching flop_counter (transformers)...", flush=True)
try:
    from torch.utils import flop_counter
    if not hasattr(flop_counter, '_unpack_params'):
        flop_counter._unpack_params = lambda params: params
    
    if not hasattr(flop_counter, '_unpack_flash_attention_nested_shapes'):
        flop_counter._unpack_flash_attention_nested_shapes = lambda args: args
        
except ImportError:
    pass

print("[torch_patch] 4. Patching accelerate (memory management)...", flush=True)
try:
    import accelerate
    from accelerate.utils import memory
    if not hasattr(memory, 'clear_device_cache') and hasattr(memory, 'release_memory'):
        memory.clear_device_cache = memory.release_memory
    elif not hasattr(memory, 'release_memory') and hasattr(memory, 'clear_device_cache'):
        memory.release_memory = memory.clear_device_cache
except ImportError:
    pass
except Exception as e:
    print(f"Warning: Failed to patch accelerate: {e}")

print("[torch_patch] 5. Patching huggingface_hub (model downloading)...", flush=True)
try:
    import huggingface_hub.errors
    import huggingface_hub.utils
    
    if not hasattr(huggingface_hub.errors, 'LocalEntryNotFoundError') and hasattr(huggingface_hub.utils, 'LocalEntryNotFoundError'):
        huggingface_hub.errors.LocalEntryNotFoundError = huggingface_hub.utils.LocalEntryNotFoundError
        
    if not hasattr(huggingface_hub.errors, 'EntryNotFoundError') and hasattr(huggingface_hub.utils, 'EntryNotFoundError'):
        huggingface_hub.errors.EntryNotFoundError = huggingface_hub.utils.EntryNotFoundError

    if not hasattr(huggingface_hub, 'cached_download'):
        try:
            from huggingface_hub import hf_hub_download
            huggingface_hub.cached_download = hf_hub_download
        except ImportError:
            pass
        
except ImportError:
    pass

print(">>> torch_patch.py loaded: Compatibility patches have been applied.", flush=True)
