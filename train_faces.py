import os
import pickle
import numpy as np
from deepface import DeepFace

# Disable oneDNN optimizations to reduce log spam
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# =========================
# CONFIG
# =========================
STUDENTS_DIR = "students faces"
EMBEDDINGS_PATH = "models/face_embeddings_arcface.pkl"
RECOGNITION_MODEL = "ArcFace"
# We use 'retinaface' or 'opencv' for alignment during training to ensure good quality
DETECTION_BACKEND = "retinaface" 

def train_model():
    print("==========================================")
    print("   TRAINING FACE RECOGNITION MODEL")
    print("==========================================")

    if not os.path.exists(STUDENTS_DIR):
        print(f"❌ Error: Directory '{STUDENTS_DIR}' not found.")
        return

    known_embeddings = []
    known_names = []
    
    students = [d for d in os.listdir(STUDENTS_DIR) if os.path.isdir(os.path.join(STUDENTS_DIR, d))]
    
    if not students:
        print(f"⚠️ No student folders found in '{STUDENTS_DIR}'.")
        return

    print(f"Found {len(students)} students: {', '.join(students)}\n")

    for student in students:
        spath = os.path.join(STUDENTS_DIR, student)
        images = [f for f in os.listdir(spath) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not images:
            print(f"⚠️ {student}: No images found.")
            continue
            
        print(f"Processing {student} ({len(images)} images)...")
        
        processed_count = 0
        for img_file in images:
            img_path = os.path.join(spath, img_file)
            try:
                # DeepFace.represent returns a list of dicts
                # enable enforce_detection to ensure we are learning from actual faces
                embedding_objs = DeepFace.represent(
                    img_path=img_path,
                    model_name=RECOGNITION_MODEL,
                    enforce_detection=True,
                    detector_backend=DETECTION_BACKEND
                )
                
                # Take the first face found in the image
                if embedding_objs:
                    embedding = embedding_objs[0]["embedding"]
                    known_embeddings.append(embedding)
                    known_names.append(student)
                    processed_count += 1
                    
            except Exception as e:
                # If face not detected or file error
                print(f"  ❌ Skipped {img_file}: {e}")

        if processed_count > 0:
            print(f"  ✅ {student}: Added {processed_count} embeddings.\n")
        else:
            print(f"  ⚠️ {student}: No valid faces extracted.\n")

    if known_embeddings:
        # Create models dir if not exists
        os.makedirs(os.path.dirname(EMBEDDINGS_PATH), exist_ok=True)
        
        with open(EMBEDDINGS_PATH, 'wb') as f:
            pickle.dump((known_embeddings, known_names), f)
        print("==========================================")
        print(f"🎉 Model successfully saved to: {EMBEDDINGS_PATH}")
        print(f"Total Embeddings: {len(known_embeddings)}")
        print(f"Unique Classes: {len(set(known_names))}")
        print("You can now run 'test_amd.py' to use this model.")
        print("==========================================")
    else:
        print("❌ No embeddings generated. Model not saved.")

if __name__ == "__main__":
    train_model()
