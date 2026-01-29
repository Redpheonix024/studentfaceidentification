# Student Face Identification System

Cross-platform face recognition system with GPU acceleration support for classroom attendance and identification.

## Features

- ✅ **Cross-Platform**: Works on Windows, Linux, and macOS
- ✅ **GPU Acceleration**: Automatic detection and support for:
  - NVIDIA CUDA (Windows/Linux)
  - AMD DirectML (Windows)
  - Apple Metal (macOS)
  - CPU fallback (all platforms)
- ✅ **State-of-the-Art Models**: InsightFace ArcFace R100 for high accuracy
- ✅ **Multiple Training Sources**: Train from class photos or passport photos
- ✅ **Real-time Video Processing**: Multi-threaded architecture for smooth playback
- ✅ **Easy to Use**: Simple GUI and command-line interfaces

## Quick Start

### 1. Installation

```bash
# Clone the repository
cd studentfaceidentification

# Install dependencies
pip install -r requirements.txt

# Verify your setup
python env_setup.py
```

### 2. GPU-Specific Setup

**For NVIDIA GPU:**
```bash
pip install onnxruntime-gpu torch torchvision
```

**For AMD GPU (Windows only):**
```bash
pip uninstall onnxruntime
pip install onnxruntime-directml
```

**For CPU only:**
```bash
# No additional steps needed
```

### 3. Train a Model

Add student photos to `students faces/<StudentName>/` directory, then:

**Option A: Using GUI (Recommended)**
```bash
python launch_training_gui.py
```

**Option B: Command Line**
```bash
# Train from class photos
python train_class_photos_r100.py

# Train from passport photos (higher accuracy)
python train_passport_insightface.py
```

### 4. Run Face Recognition

```bash
# Add videos to the 'videos' directory first
python test_amd.py --model 4
```

**Controls:**
- `p` or `Space`: Pause/Resume
- `n`: Skip to next video
- `q`: Quit

## Project Structure

```
studentfaceidentification/
├── gpu_utils.py              # Cross-platform GPU detection utilities
├── config.py                 # Centralized configuration
├── env_setup.py              # Environment verification script
├── launch_training_gui.py    # Training GUI launcher
├── test_amd.py              # Main face recognition script (cross-platform)
├── test1.py                 # Alternative recognition script (YOLO-based)
├── train_gui.py             # Training GUI
├── train_class_photos_r100.py    # Training script for class photos
├── train_passport_insightface.py # Training script for passport photos
├── models/                  # Model files directory
│   ├── retinaface_r50.onnx
│   ├── arcface_r100.onnx
│   └── face_embeddings_*.pkl
├── students faces/          # Training photos (class)
├── student faces paasport/  # Training photos (passport)
└── videos/                  # Input videos for recognition
```

## Model Selection

The system supports 4 different models:

1. **Class Photos**: Trained from classroom photos
2. **Passport Photos (Legacy)**: Older passport-based model
3. **Merged Model**: Combination of class and passport
4. **InsightFace SOTA (Default)**: State-of-the-art model (recommended)

Select model using `--model` flag:
```bash
python test_amd.py --model 4  # Use InsightFace SOTA
```

## Platform-Specific Notes

### Windows
- Both AMD and NVIDIA GPUs are supported
- Use `launch_training_gui.py` for cross-platform support
- DirectML provides good performance on AMD GPUs

### Linux
- NVIDIA CUDA support available
- Requires libGL for OpenCV: `sudo apt-get install libgl1`
- Install tkinter for GUI: `sudo apt-get install python3-tk`

### macOS
- Apple Silicon (M1/M2/M3) acceleration via Metal
- Install tkinter: included with Python from python.org
- Some OpenCV builds may not support all window properties

## Troubleshooting

### GPU Not Detected

Run diagnostics:
```bash
python gpu_utils.py
```

This will show which GPUs are available and which providers are being used.

### Missing Dependencies

Verify all dependencies:
```bash
python env_setup.py
```

### Model Files Not Found

Download or train models:
- Face detection models should be in `models/` directory
- Face embeddings are created when you run training scripts

### Performance Issues

- **For AMD GPU users**: Make sure you have `onnxruntime-directml` installed
- **For NVIDIA GPU users**: Install `onnxruntime-gpu` and ensure CUDA is set up
- **Low FPS**: Try reducing video resolution or using a lower model number

## Configuration

Environment variables for customization:

```bash
# Custom video directory
export VIDEO_DIR=/path/to/videos

# Custom models directory
export MODELS_DIR=/path/to/models

# Recognition threshold (0.0-1.0, default: 0.4)
export RECOGNITION_THRESHOLD=0.5

# Frame skip (process every N frames, default: 1)
export FRAME_SKIP=2
```

Or edit `config.py` directly.

## Documentation

- [`TRAINING_GUIDE.md`](TRAINING_GUIDE.md) - Detailed training instructions
- [`ATTENDANCE_GUIDE.md`](ATTENDANCE_GUIDE.md) - How to use for attendance tracking
- [`MODEL_VERSIONING.md`](MODEL_VERSIONING.md) - Model version management

## Requirements

- Python 3.8+
- 4GB RAM minimum (8GB+ recommended)
- For GPU: Compatible GPU with latest drivers
- For training: CUDA or DirectML (GPU) or CPU

## License

[Add your license here]

##Support

For issues and questions, please check:
1. Run `python env_setup.py` to diagnose issues
2. Check the troubleshooting section above
3. Review the documentation guides