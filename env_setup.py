"""
Environment Setup Verification Script
Checks dependencies, model files, GPU availability, and provides diagnostics
"""

import os
import sys

def check_python_version():
    """Check if Python version is compatible"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"  ✅ Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"  ❌ Python {version.major}.{version.minor}.{version.micro} - Requires Python 3.8+")
        return False

def check_dependencies():
    """Check if required packages are installed"""
    print("\nChecking dependencies...")
    
    required = {
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'onnxruntime': 'onnxruntime',
        'faiss': 'faiss-cpu',
        'deepface': 'deepface',
        'insightface': 'insightface',
        'PIL': 'Pillow'
    }
    
    optional = {
        'torch': 'torch (for NVIDIA GPU support)',
        'tensorflow': 'tensorflow (for DeepFace)',
        'ultralytics': 'ultralytics (for YOLO person detection)'
    }
    
    all_good = True
    
    # Check required
    for module, package in required.items():
        try:
            __import__(module)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} - Install with: pip install {package}")
            all_good = False
    
    # Check optional
    print("\nOptional dependencies:")
    for module, package in optional.items():
        try:
            __import__(module)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ⚠️  {package} - Not installed")
    
    return all_good

def check_model_files():
    """Check if required model files exist"""
    print("\nChecking model files...")
    
    try:
        import config
        
        models_to_check = {
            'RetinaFace': config.RETINAFACE_PATH,
            'ArcFace R100': config.ARCFACE_R100_PATH,
            'YOLOv8n': config.YOLOV8N_PATH
        }
        
        embeddings_to_check = {
            'Class Photos': config.EMBEDDINGS_ARCFACE,
            'Passport R100 (Default)': config.EMBEDDINGS_PASSPORT_R100,
            'Merged Model': config.EMBEDDINGS_MERGED
        }
        
        print("\n  Core Models:")
        for name, path in models_to_check.items():
            if os.path.exists(path):
                size_mb = os.path.getsize(path) / (1024 * 1024)
                print(f"    ✅ {name}: {size_mb:.1f} MB")
            else:
                print(f"    ❌ {name}: NOT FOUND")
                print(f"       Expected at: {path}")
        
        print("\n  Face Embeddings:")
        found_any = False
        for name, path in embeddings_to_check.items():
            if os.path.exists(path):
                size_kb = os.path.getsize(path) / 1024
                print(f"    ✅ {name}: {size_kb:.1f} KB")
                found_any = True
            else:
                print(f"    ⚠️  {name}: Not trained yet")
        
        if not found_any:
            print("\n  ℹ️  No face embeddings found. Train models using:")
            print("     - train_class_photos_r100.py (for class photos)")
            print("     - train_passport_insightface.py (for passport photos)")
        
    except ImportError:
        print("  ⚠️  config.py not found in project directory")

def check_gpu_support():
    """Check GPU acceleration support"""
    print("\nChecking GPU support...")
    
    try:
        import gpu_utils
        gpu_utils.print_hardware_summary()
    except ImportError:
        print("  ⚠️  gpu_utils.py not found - skipping GPU check")

def check_video_directory():
    """Check if video directory exists and has videos"""
    print("\nChecking video directory...")
    
    try:
        import config
        if os.path.exists(config.VIDEO_DIR):
            videos = [f for f in os.listdir(config.VIDEO_DIR) 
                     if f.lower().endswith(config.VIDEO_EXTENSIONS)]
            if videos:
                print(f"  ✅ Found {len(videos)} video(s) in {config.VIDEO_DIR}")
                for v in videos[:5]:  # Show first 5
                    print(f"     - {v}")
                if len(videos) > 5:
                    print(f"     ... and {len(videos) - 5} more")
            else:
                print(f"  ⚠️  No videos found in {config.VIDEO_DIR}")
                print(f"     Add .mp4, .avi, .mov, .mkv, or .wmv files")
        else:
            print(f"  ⚠️  Video directory not found: {config.VIDEO_DIR}")
            print(f"     Creating directory...")
            os.makedirs(config.VIDEO_DIR, exist_ok=True)
    except ImportError:
        print("  ⚠️  config.py not found")

def main():
    """Run all verification checks"""
    print("="*70)
    print("ENVIRONMENT SETUP VERIFICATION")
    print("="*70)
    
    # Run checks
    python_ok = check_python_version()
    deps_ok = check_dependencies()
    
    check_model_files()
    check_gpu_support()
    check_video_directory()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    if python_ok and deps_ok:
        print("✅ Environment is ready!")
        print("\nNext steps:")
        print("  1. Train models if not already done:")
        print("     python train_class_photos_r100.py")
        print("  2. Run face recognition:")
        print("     python test_amd.py --model 4")
    else:
        print("❌ Environment setup incomplete")
        print("\nPlease install missing dependencies:")
        print("  pip install -r requirements.txt")
    
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
