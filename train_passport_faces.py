import os
import pickle
import numpy as np
from deepface import DeepFace

# Disable oneDNN optimizations to reduce log spam
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# =========================
# CONFIG
# =========================
STUDENTS_DIR = "student faces paasport"
EMBEDDINGS_PATH = "models/face_embeddings_passport.pkl"
RECOGNITION_MODEL = "ArcFace"
# We use 'retinaface' or 'opencv' for alignment during training to ensure good quality
DETECTION_BACKEND = "retinaface" 

def train_model():
    print("==========================================")
    print("   TRAINING PASSPORT MODEL")
    print("==========================================")

    if not os.path.exists(STUDENTS_DIR):
        print(f"❌ Error: Directory '{STUDENTS_DIR}' not found.")
        return

    known_embeddings = []
    known_names = []
    
    # Get all image files in the directory
    files = [f for f in os.listdir(STUDENTS_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if not files:
        print(f"⚠️ No images found in '{STUDENTS_DIR}'.")
        return

    print(f"Found {len(files)} images to process.\n")

    for img_file in files:
        # Extract name from filename (remove extension)
        student_name = os.path.splitext(img_file)[0]
        img_path = os.path.join(STUDENTS_DIR, img_file)
        
        print(f"Processing {student_name}...")
        
        try:
            # DeepFace.represent returns a list of dicts
            embedding_objs = DeepFace.represent(
                img_path=img_path,
                model_name=RECOGNITION_MODEL,
                enforce_detection=True,
                detector_backend=DETECTION_BACKEND
            )
            
            # Take all faces found (though usually passport photo has one)
            count = 0
            if embedding_objs:
                for obj in embedding_objs:
                    embedding = obj["embedding"]
                    known_embeddings.append(embedding)
                    known_names.append(student_name)
                    count += 1
            
            if count > 0:
                print(f"  ✅ Added {count} embedding(s).")
            else:
                 print(f"  ⚠️ No faces found.")

        except Exception as e:
            # If face not detected or file error
            print(f"  ❌ Skipped {img_file}: {e}")
            
    print("\n")
    if known_embeddings:
        # Create models dir if not exists
        os.makedirs(os.path.dirname(EMBEDDINGS_PATH), exist_ok=True)
        
        with open(EMBEDDINGS_PATH, 'wb') as f:
            pickle.dump((known_embeddings, known_names), f)
        print("==========================================")
        print(f"🎉 Model successfully saved to: {EMBEDDINGS_PATH}")
        print(f"Total Embeddings: {len(known_embeddings)}")
        print(f"Unique Classes: {len(set(known_names))}")
        print("You can now run 'test_amd.py' and select 'Passport' model.")
        print("==========================================")
    else:
        print("❌ No embeddings generated. Model not saved.")

if __name__ == "__main__":
    train_model()
