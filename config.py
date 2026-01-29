"""
Centralized Configuration for Face Recognition System
All configurable paths and settings with platform-aware defaults
"""

import os
import platform

# =========================
# PLATFORM DETECTION
# =========================
PLATFORM = platform.system()  # 'Windows', 'Linux', or 'Darwin'

# =========================
# DIRECTORY PATHS
# =========================
# All paths are relative to project root for portability
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Video directory
VIDEO_DIR = os.getenv('VIDEO_DIR', os.path.join(PROJECT_ROOT, 'videos'))

# Models directory
MODELS_DIR = os.getenv('MODELS_DIR', os.path.join(PROJECT_ROOT, 'models'))

# Student faces directory
STUDENTS_DIR = os.getenv('STUDENTS_DIR', os.path.join(PROJECT_ROOT, 'students faces'))

# Passport photos directory
PASSPORT_DIR = os.getenv('PASSPORT_DIR', os.path.join(PROJECT_ROOT, 'student faces paasport'))

# Detected faces output directory
DETECTED_FACES_DIR = os.getenv('DETECTED_FACES_DIR', os.path.join(PROJECT_ROOT, 'detected_faces'))

# =========================
# MODEL PATHS
# =========================
# Face detection models
RETINAFACE_PATH = os.path.join(MODELS_DIR, 'retinaface_r50.onnx')
SCRFD_PATH = os.path.join(MODELS_DIR, 'scrfd_10g_bnkps.onnx')
YOLOV8_FACE_PATH = os.path.join(MODELS_DIR, 'yolov8n-face-lindevs.onnx')

# Face recognition models
ARCFACE_R100_PATH = os.path.join(MODELS_DIR, 'arcface_r100.onnx')
ARCFACE_PATH = os.path.join(MODELS_DIR, 'arcface.onnx')

# Person detection models
YOLOV8N_PATH = os.path.join(PROJECT_ROOT, 'yolov8n.pt')
YOLOV8N_ONNX_PATH = os.path.join(PROJECT_ROOT, 'yolov8n.onnx')

# =========================
# EMBEDDINGS PATHS
# =========================
# Face embeddings databases
EMBEDDINGS_ARCFACE = os.path.join(MODELS_DIR, 'face_embeddings_arcface.pkl')
EMBEDDINGS_PASSPORT = os.path.join(MODELS_DIR, 'face_embeddings_passport.pkl')
EMBEDDINGS_PASSPORT_R100 = os.path.join(MODELS_DIR, 'face_embeddings_passport_r100.pkl')
EMBEDDINGS_MERGED = os.path.join(MODELS_DIR, 'face_embeddings_merged.pkl')

# Model selection mapping
MODEL_PATHS = {
    1: EMBEDDINGS_ARCFACE,           # Class photos
    2: EMBEDDINGS_PASSPORT,           # Passport photos (legacy)
    3: EMBEDDINGS_MERGED,             # Merged model
    4: EMBEDDINGS_PASSPORT_R100       # InsightFace SOTA (default)
}

# Default model
DEFAULT_MODEL = 4
DEFAULT_EMBEDDINGS_PATH = MODEL_PATHS[DEFAULT_MODEL]

# =========================
# RECOGNITION SETTINGS
# =========================
# Face recognition thresholds
RECOGNITION_THRESHOLD = float(os.getenv('RECOGNITION_THRESHOLD', '0.4'))

# Face detection confidence
DETECTION_CONFIDENCE = float(os.getenv('DETECTION_CONFIDENCE', '0.5'))

# DeepFace settings (for training scripts)
DEEPFACE_RECOGNITION_MODEL = 'ArcFace'
DEEPFACE_DETECTION_BACKEND = 'retinaface'

# =========================
# VIDEO PROCESSING SETTINGS
# =========================
# Frame skip for recognition (process every Nth frame)
FRAME_SKIP = int(os.getenv('FRAME_SKIP', '1'))

# Video extensions to process
VIDEO_EXTENSIONS = ('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv')

# =========================
# DISPLAY SETTINGS
# =========================
# OpenCV window settings
WINDOW_NAME = 'Face Recognition System'
ENABLE_OPENCL = True  # Use OpenCL for GPU-accelerated rendering

# =========================
# HELPER FUNCTIONS
# =========================
def get_embeddings_path(model_id=None):
    """
    Get embeddings path for specified model ID.
    
    Args:
        model_id: 1-4 for different models, None for default
    
    Returns:
        Path to embeddings file
    """
    if model_id is None:
        model_id = DEFAULT_MODEL
    return MODEL_PATHS.get(model_id, DEFAULT_EMBEDDINGS_PATH)

def get_model_name(model_id):
    """Get human-readable name for model ID"""
    names = {
        1: "Class Photos (ArcFace)",
        2: "Passport Photos (Legacy)",
        3: "Merged Model",
        4: "InsightFace Passport Model (SOTA)"
    }
    return names.get(model_id, "Unknown Model")

def ensure_directories():
    """Create necessary directories if they don't exist"""
    dirs = [
        VIDEO_DIR,
        MODELS_DIR,
        STUDENTS_DIR,
        PASSPORT_DIR,
        DETECTED_FACES_DIR
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

def print_config():
    """Print current configuration"""
    print("\n" + "="*60)
    print("CONFIGURATION")
    print("="*60)
    print(f"Platform: {PLATFORM}")
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"\nDirectories:")
    print(f"  Videos: {VIDEO_DIR}")
    print(f"  Models: {MODELS_DIR}")
    print(f"  Students: {STUDENTS_DIR}")
    print(f"  Passport: {PASSPORT_DIR}")
    print(f"\nDefault Model: {get_model_name(DEFAULT_MODEL)}")
    print(f"  Path: {DEFAULT_EMBEDDINGS_PATH}")
    print(f"\nSettings:")
    print(f"  Recognition Threshold: {RECOGNITION_THRESHOLD}")
    print(f"  Detection Confidence: {DETECTION_CONFIDENCE}")
    print(f"  Frame Skip: {FRAME_SKIP}")
    print("="*60 + "\n")

if __name__ == "__main__":
    # When run directly, print configuration
    ensure_directories()
    print_config()
