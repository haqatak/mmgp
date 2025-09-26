import torch
import warnings

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = get_device()
DEVICE_NAME = DEVICE.type

def is_cuda():
    return DEVICE_NAME == "cuda"

def is_mps():
    return DEVICE_NAME == "mps"

def get_device_properties(device):
    if is_cuda():
        return torch.cuda.get_device_properties(device)
    if is_mps():
        # MPS does not have device properties like CUDA
        warnings.warn("torch.backends.mps.get_device_properties is not available. Returning default values.")
        class MockProperties:
            def __init__(self):
                self.total_memory = 0 # MPS memory is unified, so this is not directly applicable
        return MockProperties()
    return None

def get_total_memory(device):
    if is_cuda():
        return torch.cuda.get_device_properties(device).total_memory
    return 0 # Not applicable for MPS (unified memory)

def memory_reserved():
    if is_cuda():
        return torch.cuda.memory_reserved()
    return 0

def memory_allocated():
    if is_cuda():
        return torch.cuda.memory_allocated()
    return 0

def empty_cache():
    if is_cuda():
        torch.cuda.empty_cache()
    elif is_mps():
        torch.mps.empty_cache()

def Stream():
    if is_cuda():
        return torch.cuda.Stream()
    return None # MPS does not support streams

def current_stream(device=None):
    if is_cuda():
        return torch.cuda.current_stream(device)
    return None

def default_stream(device=None):
    if is_cuda():
        return torch.cuda.default_stream(device)
    return None

def synchronize(device=None):
    if is_cuda():
        torch.cuda.synchronize(device)
    elif is_mps():
        torch.mps.synchronize()

def stream(stream):
    if is_cuda() and stream is not None:
        return torch.cuda.stream(stream)

    # Return a dummy context manager if streams are not supported
    class MockStream:
        def __enter__(self):
            pass
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass
    return MockStream()

def set_default_device(device):
    if device == "cuda" and is_cuda():
        torch.set_default_device(device)
    # MPS cannot be set as a default device in the same way
    elif device == "mps" and is_mps():
        pass # torch.set_default_tensor_type('torch.mps.FloatTensor') is one way but has side effects
    else:
        torch.set_default_device("cpu")

def can_pin_memory():
    return is_cuda()