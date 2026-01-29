"""
GPU Detection and Configuration Utilities
Supports: NVIDIA CUDA, AMD DirectML (Windows), Apple Metal (macOS), CPU fallback
"""

import os
import platform
import sys

def get_platform():
    """Get current platform: Windows, Linux, or Darwin (macOS)"""
    return platform.system()

def detect_nvidia_gpu():
    """Check if NVIDIA GPU is available"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            return True, gpu_name
    except ImportError:
        pass
    return False, None

def detect_amd_gpu():
    """Check if AMD GPU with DirectML is available (Windows only)"""
    if get_platform() != "Windows":
        return False, None
    
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        if 'DmlExecutionProvider' in providers:
            return True, "AMD GPU (DirectML)"
    except ImportError:
        pass
    return False, None

def get_onnx_providers():
    """
    Get optimal ONNX Runtime providers based on available hardware.
    Returns: list of provider names in priority order
    """
    try:
        import onnxruntime as ort
        available = ort.get_available_providers()
    except ImportError:
        print("⚠️  onnxruntime not installed. Install with: pip install onnxruntime")
        return ['CPUExecutionProvider']
    
    providers = []
    
    # NVIDIA CUDA (cross-platform)
    if 'CUDAExecutionProvider' in available:
        providers.append('CUDAExecutionProvider')
        print("✅ Using NVIDIA CUDA acceleration")
    
    # AMD DirectML (Windows only)
    elif 'DmlExecutionProvider' in available:
        providers.append('DmlExecutionProvider')
        print("✅ Using AMD DirectML acceleration")
    
    # Apple CoreML/Metal (macOS)
    elif 'CoreMLExecutionProvider' in available:
        providers.append('CoreMLExecutionProvider')
        print("✅ Using Apple CoreML acceleration")
    
    # Fallback to CPU
    if 'CPUExecutionProvider' in available:
        providers.append('CPUExecutionProvider')
    
    if len(providers) == 1 and providers[0] == 'CPUExecutionProvider':
        print("⚠️  Using CPU only. For GPU acceleration:")
        if get_platform() == "Windows":
            print("   - NVIDIA: pip install onnxruntime-gpu")
            print("   - AMD: pip install onnxruntime-directml")
        elif get_platform() == "Linux":
            print("   - NVIDIA: pip install onnxruntime-gpu")
        elif get_platform() == "Darwin":
            print("   - Apple Silicon: CoreML support built-in")
    
    return providers

def get_torch_device():
    """
    Get optimal PyTorch device.
    Returns: 'cuda', 'mps', 'cpu', or CUDA device index (0)
    """
    try:
        import torch
        
        # NVIDIA CUDA
        if torch.cuda.is_available():
            print(f"✅ PyTorch using NVIDIA GPU: {torch.cuda.get_device_name(0)}")
            return 0  # Device index for ultralytics/YOLO
        
        # Apple Metal Performance Shaders (M1/M2/M3)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print("✅ PyTorch using Apple Metal (MPS)")
            return 'mps'
        
        # CPU fallback
        print("⚠️  PyTorch using CPU")
        return 'cpu'
        
    except ImportError:
        print("⚠️  PyTorch not installed")
        return 'cpu'

def configure_tensorflow_gpu():
    """Configure TensorFlow to use GPU if available"""
    try:
        import tensorflow as tf
        
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                # Enable memory growth to avoid allocating all GPU memory
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"✅ TensorFlow using GPU: {len(gpus)} device(s)")
                return True
            except RuntimeError as e:
                print(f"⚠️  TensorFlow GPU configuration error: {e}")
                return False
        else:
            print("⚠️  TensorFlow using CPU")
            return False
    except ImportError:
        print("⚠️  TensorFlow not installed")
        return False

def print_hardware_summary():
    """Print a summary of detected hardware and acceleration capabilities"""
    print("\n" + "="*60)
    print("HARDWARE DETECTION SUMMARY")
    print("="*60)
    
    # Platform
    plat = get_platform()
    print(f"Platform: {plat} ({platform.machine()})")
    print(f"Python: {sys.version.split()[0]}")
    
    # NVIDIA GPU
    has_nvidia, nvidia_name = detect_nvidia_gpu()
    if has_nvidia:
        print(f"NVIDIA GPU: ✅ {nvidia_name}")
    else:
        print(f"NVIDIA GPU: ❌ Not detected")
    
    # AMD GPU
    has_amd, amd_name = detect_amd_gpu()
    if has_amd:
        print(f"AMD GPU: ✅ {amd_name}")
    else:
        print(f"AMD GPU: ❌ Not detected")
    
    # ONNX Runtime
    print(f"\nONNX Runtime Providers:")
    providers = get_onnx_providers()
    for i, p in enumerate(providers, 1):
        print(f"  {i}. {p}")
    
    # PyTorch
    print(f"\nPyTorch Device: {get_torch_device()}")
    
    # TensorFlow
    print(f"TensorFlow GPU: {'✅ Enabled' if configure_tensorflow_gpu() else '❌ Disabled'}")
    
    print("="*60 + "\n")

if __name__ == "__main__":
    # When run directly, print hardware summary
    print_hardware_summary()
