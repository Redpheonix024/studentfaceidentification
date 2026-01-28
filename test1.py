import os
import cv2
import time
import queue
import threading
import pickle

import numpy as np
from ultralytics import YOLO
from deepface import DeepFace
# GPU Support
import torch
import tensorflow as tf


os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# =========================
# GPU SETUP
# =========================
DEVICE = 'cpu'
if torch.cuda.is_available():
    DEVICE = 0
    print(f"✅ Torch using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("❌ Torch using CPU")

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ TensorFlow using GPU: {len(gpus)} device(s)")
    except RuntimeError as e:
        print(e)
else:
    print("❌ TensorFlow using CPU")


# =========================
# PATHS
# =========================
VIDEO_PATH = "videos/20260107_124259.mp4"
FACE_MODEL_PATH = "models/yolov8n-face-lindevs.onnx"
STUDENTS_DIR = "students faces"
EMBEDDINGS_PATH = "models/face_embeddings_arcface.pkl"
# Models
RECOGNITION_MODEL = "ArcFace"
DETECTION_BACKEND = "retinaface"



# =========================
# LOAD MODELS
# =========================
# face_net removed, using DeepFace internal detector
person_model = YOLO("yolov8n.pt")


# =========================
# TRAIN OR LOAD MODEL
# =========================
known_embeddings = []
known_names = []

use_saved = False
if os.path.exists(EMBEDDINGS_PATH):
    choice = input(f"Found saved model at {EMBEDDINGS_PATH}. Use it? (y/n): ").strip().lower()
    if choice == 'y':
        use_saved = True

if use_saved:
    print(f"Loading embeddings from {EMBEDDINGS_PATH}...")
    try:
        with open(EMBEDDINGS_PATH, 'rb') as f:
            known_embeddings, known_names = pickle.load(f)
        print(f"Loaded {len(known_names)} students.")
    except Exception as e:
        print(f"Error loading model: {e}. Switching to training mode.")
        use_saved = False

if not use_saved:
    print("Training students...")
    for student in os.listdir(STUDENTS_DIR):
        spath = os.path.join(STUDENTS_DIR, student)
        if not os.path.isdir(spath):
            continue

        embs = []
        for img in os.listdir(spath):
            try:
                # Enforce detection = True ensures we crop precisely to the face
                rep = DeepFace.represent(
                    img_path=os.path.join(spath, img),
                    model_name=RECOGNITION_MODEL,
                    enforce_detection=True,
                    detector_backend=DETECTION_BACKEND
                )[0]["embedding"]
                embs.append(rep)
            except Exception as e:
                print(f"⚠️ Skipped {img} for {student}: {e}")



        if embs:
            known_embeddings.append(np.mean(embs, axis=0))
            known_names.append(student)
            print(f"✔ {student} ({len(embs)} images)")
            
    # Save the trained model
    print(f"Saving model to {EMBEDDINGS_PATH}...")
    with open(EMBEDDINGS_PATH, 'wb') as f:
        pickle.dump((known_embeddings, known_names), f)
    print("Model saved.")

# =========================
# HELPERS
# =========================
def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def center(box):
    x1, y1, x2, y2 = box
    return (int((x1+x2)/2), int((y1+y2)/2))

# =========================
# THREAD QUEUES
# =========================
frame_queue = queue.Queue(maxsize=2)
result_queue = queue.Queue(maxsize=2)
running = True

# =========================
# THREAD 1: VIDEO READER
# =========================
def video_reader():
    cap = cv2.VideoCapture(VIDEO_PATH)
    while running:
        ret, frame = cap.read()
        if not ret:
            break
        if not frame_queue.full():
            frame_queue.put(frame)
    cap.release()

# =========================
# THREAD 2: DETECTION + RECOGNITION
# =========================
def detector():
    while running:
        if frame_queue.empty():
            continue

        frame = frame_queue.get()
        h, w = frame.shape[:2]

        persons = []
        faces = []

        # ---- PERSON DETECTION ----
        # Explicitly using the determined DEVICE
        results = person_model(frame, conf=0.3, classes=[0], device=DEVICE, verbose=False)

        for r in results:
            for b in r.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                persons.append([x1, y1, x2, y2, "Unknown"])

        # ---- FACE DETECTION & RECOGNITION via DeepFace ----
        # Iterate over detected persons to narrow down search area
        for p in persons:
             # p = [x1, y1, x2, y2, "Unknown"]
             x1, y1, x2, y2 = p[:4]
             
             # Crop person
             person_img = frame[y1:y2, x1:x2]
             if person_img.size == 0:
                 continue
                 
             try:
                # Detect and Recognize in one go or detect first
                # Using DeepFace.extract_faces first to get bounding box within crop
                face_objs = DeepFace.extract_faces(
                    img_path=person_img,
                    detector_backend=DETECTION_BACKEND,
                    enforce_detection=True,
                    align=True
                )
                
                for face_obj in face_objs:
                    # facial_area keys: x, y, w, h
                    area = face_obj["facial_area"]
                    fx = x1 + area['x']
                    fy = y1 + area['y']
                    fw = area['w']
                    fh = area['h']
                    
                    # Get embedding for this face
                    # We can pass the face array directly to represent
                    # face_obj['face'] is normalized 0-1, represent expects path or numpy array (0-255 usually if BGR)
                    # Safe to use the crop from original frame again
                    face_crop = frame[fy:fy+fh, fx:fx+fw]
                    
                    if face_crop.size == 0:
                        continue

                    emb_results = DeepFace.represent(
                        img_path=face_crop,
                        model_name=RECOGNITION_MODEL,
                        enforce_detection=False,
                        detector_backend="skip" # Already detected
                    )
                    
                    if not emb_results:
                        continue
                        
                    emb = emb_results[0]["embedding"]
                    name = "Unknown"
                    best_score = 0
                    
                    for kemb, kname in zip(known_embeddings, known_names):
                        s = cosine(emb, kemb)
                        if s > best_score:
                            best_score = s
                            name = kname
                    
                    if best_score < 0.4: # ArcFace threshold is usually different, 0.4 is safe conservative for cosine
                        name = "Unknown"
                        
                    faces.append((fx, fy, fw, fh, name))

             except:
                 # No face found in this person crop
                 pass

        # ---- ASSIGN FACE → PERSON ----
        # (Reusing existing logic, though we iterated persons to find faces, 
        # so we could have assigned directly. But keeping separate lists matches original structure)

        for fx, fy, bw, bh, name in faces:
            fc = center([fx, fy, fx+bw, fy+bh])
            for p in persons:
                pc = center(p[:4])
                if abs(fc[0]-pc[0]) < 80 and abs(fc[1]-pc[1]) < 80:
                    p[4] = name

        if not result_queue.full():
            result_queue.put((frame, persons))

# =========================
# START THREADS
# =========================
threading.Thread(target=video_reader, daemon=True).start()
threading.Thread(target=detector, daemon=True).start()

# =========================
# MAIN DISPLAY LOOP
# =========================
while True:
    if not result_queue.empty():
        frame, persons = result_queue.get()

        for x1, y1, x2, y2, name in persons:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, name, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.imshow("Multithreaded Classroom Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        running = False
        break

cv2.destroyAllWindows()
