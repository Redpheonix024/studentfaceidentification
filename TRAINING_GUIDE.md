# Passport + Class Photos Training Guide

## Overview
This guide explains how to train a face recognition model using both passport-size photos and class photos, then merge them into a single unified model.

## Directory Structure

```
student faces paasport/    # Passport-size photos (flat directory)
├── ABHAY.jpeg
├── GOURIPRIYA.jpeg
├── Hamdha Mohammed.jpeg
└── ...

students faces/            # Class photos (subdirectory structure)
├── ABHAY/
│   ├── photo1.jpg
│   ├── photo2.jpg
│   └── ...
├── GOURIPRIYA/
│   ├── photo1.jpg
│   └── ...
└── ...
```

## Training Scripts

### 1. All-in-One Training & Merging
**File:** `train_and_merge_passport_class.py`

This script does everything in one go:
- Trains on passport photos
- Trains on class photos  
- Merges both models into one

**Usage:**
```bash
.\enven\Scripts\python.exe train_and_merge_passport_class.py
```

**Output:**
- `models/face_embeddings_passport_r100.pkl` - Passport photos model
- `models/face_embeddings_class_r100.pkl` - Class photos model
- `models/face_embeddings_merged_r100.pkl` - **Combined model (use this!)**

### 2. Individual Training Scripts (if needed)

#### Train Passport Photos Only
```bash
.\enven\Scripts\python.exe train_passport_insightface.py
```

#### Train Class Photos Only
```bash
.\enven\Scripts\python.exe train_class_photos_r100.py
```

#### Merge Existing Models
```bash
.\enven\Scripts\python.exe merge_models_r100.py
```

## Verification

To verify the merged model was created successfully:

```bash
.\enven\Scripts\python.exe verify_merged_model.py
```

Or simply double-click:
```
verify_model.bat
```

This will show:
- Total number of embeddings
- Number of unique students
- Breakdown of embeddings per student

## Using the Merged Model

### In Your Recognition Scripts

Update your test/recognition scripts to use the merged model:

```python
# Change this line:
MODEL_PATH = "models/face_embeddings_class_r100.pkl"

# To this:
MODEL_PATH = "models/face_embeddings_merged_r100.pkl"
```

### Example Files to Update
- `test_amd.py`
- `test_models.py`
- `generate_attendance.py`
- Any other script that loads face embeddings

## Workflow

### Initial Setup
1. Place passport photos in `student faces paasport/` directory
   - Filename should be: `StudentName.jpg` or `StudentName.jpeg`
   
2. Place class photos in `students faces/` directory
   - Create subdirectory for each student
   - Place their photos inside: `students faces/StudentName/photo1.jpg`

### Training
3. Run the all-in-one training script:
   ```bash
   .\enven\Scripts\python.exe train_and_merge_passport_class.py
   ```

4. Verify the model:
   ```bash
   .\enven\Scripts\python.exe verify_merged_model.py
   ```

### Using the Model
5. Update your recognition scripts to use `face_embeddings_merged_r100.pkl`

6. Run your recognition/attendance scripts as usual

## Model Details

- **Detection Model:** RetinaFace R50
- **Recognition Model:** ArcFace R100
- **Embedding Dimension:** 512-dimensional vectors
- **Format:** Python pickle file containing (embeddings_list, names_list)

## Benefits of Merged Model

1. **More Training Data:** Each student has embeddings from both passport and class photos
2. **Better Recognition:** More diverse angles, lighting, and expressions
3. **Higher Accuracy:** Multiple embeddings per student improve matching reliability
4. **Single Model:** No need to manage multiple model files

## Troubleshooting

### "No face detected" warnings
- The script will use the full image as fallback
- Consider improving image quality or lighting

### Student name mismatches
- Passport photos: Filename determines the student name
- Class photos: Subdirectory name determines the student name
- Make sure names are consistent between both directories

### Model file not found
- Ensure the ONNX models exist in `models/` directory:
  - `retinaface_r50.onnx`
  - `arcface_r100.onnx`

## Quick Reference

| Task | Command |
|------|---------|
| Train & Merge (Recommended) | `.\enven\Scripts\python.exe train_and_merge_passport_class.py` |
| Verify Model | `.\enven\Scripts\python.exe verify_merged_model.py` |
| Train Passport Only | `.\enven\Scripts\python.exe train_passport_insightface.py` |
| Train Class Only | `.\enven\Scripts\python.exe train_class_photos_r100.py` |
| Merge Existing Models | `.\enven\Scripts\python.exe merge_models_r100.py` |

## Example Output

```
======================================================================
  PASSPORT + CLASS PHOTOS TRAINING & MERGING PIPELINE
  Using: RetinaFace R50 + ArcFace R100 (512-dim embeddings)
======================================================================

============================================================
  STEP 1: TRAINING ON PASSPORT PHOTOS
============================================================

Found 8 passport images

[OK] ABHAY: Face detected
[OK] GOURIPRIYA: Face detected
...

[OK] Passport Training Complete:
   - Processed: 8
   - Failed: 0
   - Saved to: models/face_embeddings_passport_r100.pkl

============================================================
  STEP 2: TRAINING ON CLASS PHOTOS
============================================================

Found 73 student directories

[User] ABHAY (15 photos)
   [OK] Generated 15 embeddings
...

[OK] Class Training Complete:
   - Total images: 782
   - Faces detected: 782
   - Failed: 0
   - Unique students: 73
   - Total embeddings: 782
   - Saved to: models/face_embeddings_class_r100.pkl

============================================================
  STEP 3: MERGING MODELS
  Passport + Class → Merged Model
============================================================

[OK] Added 8 passport embeddings
[OK] Added 782 class embeddings

======================================================================
  [SUCCESS] MERGE COMPLETE!
======================================================================
[OK] Total embeddings: 790
[OK] Unique students: 73
[Saved] Saved to: models/face_embeddings_merged_r100.pkl
======================================================================

[BREAKDOWN] Embeddings per student:
   ABHAY: 16
   GOURIPRIYA: 11
   ...
```
