import os
import pickle
import numpy as np
from deepface import DeepFace

# Disable oneDNN optimizations
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# =========================
# CONFIG
# =========================
DIR_CLASSROOM = "students faces"
DIR_PASSPORT = "student faces paasport"
EMBEDDINGS_PATH = "models/face_embeddings_merged.pkl"
RECOGNITION_MODEL = "ArcFace"
DETECTION_BACKEND = "retinaface"

def train_model():
    print("==========================================")
    print("   TRAINING MERGED MODEL (Class + Passport)")
    print("==========================================")

    known_embeddings = []
    known_names = []
    
    # ---------------------------------------------------------
    # 1. Process Classroom Photos (Folder Structure)
    # ---------------------------------------------------------
    if os.path.exists(DIR_CLASSROOM):
        students = [d for d in os.listdir(DIR_CLASSROOM) if os.path.isdir(os.path.join(DIR_CLASSROOM, d))]
        print(f"📂 Processing Classroom Directory: {len(students)} students found.")
        
        for student in students:
            spath = os.path.join(DIR_CLASSROOM, student)
            images = [f for f in os.listdir(spath) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            if not images: continue
            
            # print(f"  > {student}: {len(images)} images")
            
            count = 0
            for img_file in images:
                try:
                    embedding_objs = DeepFace.represent(
                        img_path=os.path.join(spath, img_file),
                        model_name=RECOGNITION_MODEL,
                        enforce_detection=True,
                        detector_backend=DETECTION_BACKEND
                    )
                    if embedding_objs:
                        known_embeddings.append(embedding_objs[0]["embedding"])
                        known_names.append(student)
                        count += 1
                except:
                    pass
            print(f"    ✔ {student}: Added {count} embeddings from class photos.")
    else:
        print(f"⚠️ Directory '{DIR_CLASSROOM}' not found.")

    print("-" * 30)

    # ---------------------------------------------------------
    # 2. Process Passport Photos (Flat Structure)
    # ---------------------------------------------------------
    if os.path.exists(DIR_PASSPORT):
        files = [f for f in os.listdir(DIR_PASSPORT) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"📂 Processing Passport Directory: {len(files)} files found.")
        
        for img_file in files:
            student_name = os.path.splitext(img_file)[0]
            # print(f"  > {student_name}...")
            
            try:
                embedding_objs = DeepFace.represent(
                    img_path=os.path.join(DIR_PASSPORT, img_file),
                    model_name=RECOGNITION_MODEL,
                    enforce_detection=True,
                    detector_backend=DETECTION_BACKEND
                )
                
                count = 0
                if embedding_objs:
                    for obj in embedding_objs:
                        known_embeddings.append(obj["embedding"])
                        known_names.append(student_name)
                        count += 1
                
                if count > 0:
                    print(f"    ✔ {student_name}: Added {count} embeddings from passport.")
            except:
                pass
    else:
        print(f"⚠️ Directory '{DIR_PASSPORT}' not found.")

    # ---------------------------------------------------------
    # 3. Save Model
    # ---------------------------------------------------------
    print("\n")
    if known_embeddings:
        os.makedirs(os.path.dirname(EMBEDDINGS_PATH), exist_ok=True)
        with open(EMBEDDINGS_PATH, 'wb') as f:
            pickle.dump((known_embeddings, known_names), f)
            
        print("==========================================")
        print(f"🎉 Merged Model saved to: {EMBEDDINGS_PATH}")
        print(f"Total Embeddings: {len(known_embeddings)}")
        print(f"Unique Student Names: {len(set(known_names))}")
        print("==========================================")
    else:
        print("❌ No embeddings generated. Model not saved.")

if __name__ == "__main__":
    train_model()
