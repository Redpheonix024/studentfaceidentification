import os
import sys
import cv2
import time
import queue
import threading
import numpy as np
import pickle
import faiss
import argparse

# =========================
# ONNX / GPU SETUP
# =========================
try:
    import onnxruntime as ort
    providers = ort.get_available_providers()
    if 'DmlExecutionProvider' in providers:
        print("✅ Using DirectML (AMD GPU)")
        PROVIDERS = ['DmlExecutionProvider']
    else:
        print("⚠️ DirectML not found. Falling back to CPU.")
        PROVIDERS = ['CPUExecutionProvider']
except ImportError:
    print("❌ onnxruntime not found.")
    PROVIDERS = ['CPUExecutionProvider']
    ort = None

if cv2.ocl.haveOpenCL():
    cv2.ocl.setUseOpenCL(True)

# =========================
# CONFIG
# =========================
VIDEO_DIR = "videos"
MODELS_DIR = "models"
RETINAFACE_PATH = "models/retinaface_r50.onnx"
ARCFACE_PATH = "models/arcface_r100.onnx"
# Default to new passport model
EMBEDDINGS_PATH = "models/face_embeddings_passport_r100.pkl"

# =========================
# MODEL CLASSES
# =========================
class SCRFD:
    def __init__(self, model_file, providers):
        self.session = ort.InferenceSession(model_file, providers=providers)
        self.conf_thresh = 0.5
        self.nms_thresh = 0.4
        self.input_shape = (640, 640)
        
    def detect(self, img):
        h, w = img.shape[:2]
        scale = 640 / max(h, w)
        img_resized = cv2.resize(img, (0,0), fx=scale, fy=scale)
        pad_h = 640 - img_resized.shape[0]
        pad_w = 640 - img_resized.shape[1]
        img_input = cv2.copyMakeBorder(img_resized, 0, pad_h, 0, pad_w, cv2.BORDER_CONSTANT, value=0)
        
        blob = cv2.dnn.blobFromImage(img_input, 1.0/128, (640, 640), (127.5, 127.5, 127.5), swapRB=True)
        outputs = self.session.run(None, {self.session.get_inputs()[0].name: blob})
        
        scores_list = []
        bboxes_list = []
        
        fmc = 3
        strides = [8, 16, 32]
        
        for i, stride in enumerate(strides):
            score = outputs[i]
            bbox = outputs[i + fmc]
            
            if len(score.shape) == 3: score = score[0]
            if len(bbox.shape) == 3: bbox = bbox[0]
            
            if len(score.shape) == 1: score = score.reshape(-1, 1)
            if len(bbox.shape) == 1: bbox = bbox.reshape(-1, 4)
            
            h_fmap = 640 // stride
            w_fmap = 640 // stride
            
            anchor_centers = np.stack(np.meshgrid(np.arange(w_fmap), np.arange(h_fmap)), axis=-1)
            anchor_centers = anchor_centers.reshape(-1, 2) * stride
            
            if score.shape[0] != anchor_centers.shape[0]:
                n_anchors = anchor_centers.shape[0]
                ratio = score.shape[0] // n_anchors
                if ratio > 1:
                     anchor_centers = np.repeat(anchor_centers, ratio, axis=0)
                     
                if score.shape[0] != anchor_centers.shape[0]:
                    min_len = min(score.shape[0], anchor_centers.shape[0])
                    score = score[:min_len]
                    bbox = bbox[:min_len]
                    anchor_centers = anchor_centers[:min_len]
                
            bbox = bbox * stride
            
            valid = (score[:, 0] > self.conf_thresh)
            if not np.any(valid): continue
            
            score = score[valid]
            bbox = bbox[valid]
            anchor_centers = anchor_centers[valid]
            
            x1 = anchor_centers[:, 0] - bbox[:, 0]
            y1 = anchor_centers[:, 1] - bbox[:, 1]
            x2 = anchor_centers[:, 0] + bbox[:, 2]
            y2 = anchor_centers[:, 1] + bbox[:, 3]
            
            boxes = np.stack([x1, y1, x2, y2], axis=-1)
            scores_list.append(score)
            bboxes_list.append(boxes)
                
        if not scores_list: return []
        
        scores = np.vstack(scores_list)
        bboxes = np.vstack(bboxes_list)
        
        keep = cv2.dnn.NMSBoxes(bboxes.tolist(), scores[:, 0].tolist(), self.conf_thresh, self.nms_thresh)
        
        final_faces = []
        if len(keep) > 0:
            indices = keep.flatten()
            for k in indices:
                x1, y1, x2, y2 = bboxes[k]
                x1 /= scale; y1 /= scale; x2 /= scale; y2 /= scale
                
                x1 = max(0, int(x1)); y1 = max(0, int(y1))
                x2 = min(w, int(x2)); y2 = min(h, int(y2))
                
                width_box = x2 - x1
                height_box = y2 - y1
                
                final_faces.append([x1, y1, width_box, height_box, "Unknown"])
                
        return final_faces

class ArcFaceONNX:
    def __init__(self, model_file, providers):
        self.session = ort.InferenceSession(model_file, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        
    def get_embedding(self, img):
        if img.shape[0] != 112 or img.shape[1] != 112:
            img = cv2.resize(img, (112, 112))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, 0)
        img = (img - 127.5) / 128.0
        img = img.astype(np.float32)
        emb = self.session.run(None, {self.input_name: img})[0]
        norm = np.linalg.norm(emb)
        if norm > 0: emb /= norm
        return emb.flatten()

# =========================
# LOADING DB
# =========================
known_embeddings = []
known_names = []

# Parse command-line arguments for model selection
parser = argparse.ArgumentParser(description='Face Recognition with AMD GPU support')
parser.add_argument('--model', type=int, default=4, choices=[1, 2, 3, 4],
                    help='Model selection: 1=Class Photos, 2=Passport Legacy, 3=Merged, 4=InsightFace SOTA (default: 4)')
args = parser.parse_args()

print("\n-----------------------------")
print("Available Models:")
print("1. Detection w/ Class Photos")
print("2. Detection w/ Passport Photos (Legacy)")
print("3. Detection w/ Merged Model")
print("4. Detection w/ InsightFace Passport Model (SOTA)")
print(f"Selected: Model {args.model}")

if args.model == 2: 
    EMBEDDINGS_PATH = "models/face_embeddings_passport.pkl"
elif args.model == 3: 
    EMBEDDINGS_PATH = "models/face_embeddings_merged.pkl"
elif args.model == 1: 
    EMBEDDINGS_PATH = "models/face_embeddings_arcface.pkl"
else: 
    EMBEDDINGS_PATH = "models/face_embeddings_passport_r100.pkl"

print(f">> Using {EMBEDDINGS_PATH}\n")

if os.path.exists(EMBEDDINGS_PATH):
    try:
        with open(EMBEDDINGS_PATH, 'rb') as f:
            known_embeddings, known_names = pickle.load(f)
        if known_embeddings:
            dim = len(known_embeddings[0])
            db_emb = np.array(known_embeddings, dtype=np.float32)
            faiss.normalize_L2(db_emb)
            index = faiss.IndexFlatIP(dim)
            index.add(db_emb)
            print(f"✅ Loaded {len(known_embeddings)} items.")
        else: index = None
    except Exception as e:
        print(f"❌ Error: {e}")
        index = None
else:
    print("❌ Model file not found.")
    index = None

# =========================
# THREADS & SHARED STATE
# =========================
# Main Thread: Video Reading + Drawing + Imshow
# Background Thread 1: Detection (Updates LATEST_FACES)
# Background Thread 2: Recognition (Updates LATEST_FACES names)

raw_frame_queue = queue.Queue(maxsize=30) # Buffer video frames
detect_input_queue = queue.Queue(maxsize=1) 
recog_input_queue = queue.Queue(maxsize=1)

LATEST_FACES = [] # [(x,y,w,h,name), ...]
LATEST_FACES_LOCK = threading.Lock()

running = True
paused = False
skip_video = False

def detector_thread():
    global running, LATEST_FACES
    
    if not os.path.exists(RETINAFACE_PATH):
        print(f"❌ {RETINAFACE_PATH} missing.")
        model = None
    else:
        model = SCRFD(RETINAFACE_PATH, PROVIDERS)
        print("✅ RetinaFace Loaded")
        
    while running:
        frame = detect_input_queue.get()
        if frame is None: break # Sentinel
        
        faces = []
        if model:
            try:
                faces = model.detect(frame)
            except: pass
            
        # Update shared state with Unknown names
        with LATEST_FACES_LOCK:
            # We want to preserve names if boxes match closely? 
            # For simplicity in this robust version, we overwrite. 
            # Recognition thread will fill names roughly same time?
            # Better: pass to recognizer directly.
            pass
            
        if faces and recog_input_queue.empty():
            recog_input_queue.put((frame, faces))
        elif faces:
            # If recognizer busy, just update display with Unknowns for responsiveness
            with LATEST_FACES_LOCK:
                LATEST_FACES = faces 

def recognizer_thread():
    global running, LATEST_FACES
    
    if not os.path.exists(ARCFACE_PATH):
        print(f"❌ {ARCFACE_PATH} missing.")
        model = None
    else:
        model = ArcFaceONNX(ARCFACE_PATH, PROVIDERS)
        print("✅ ArcFace Loaded")

    while running:
        item = recog_input_queue.get()
        if item is None: break
        frame, faces = item
        
        res = []
        if model and index:
            for x, y, w, h, _ in faces:
                try:
                    crop = frame[y:y+h, x:x+w]
                    if crop.size == 0: continue
                    emb = model.get_embedding(crop)
                    
                    qn = np.array([emb], dtype=np.float32)
                    faiss.normalize_L2(qn)
                    D, I = index.search(qn, 1)
                    
                    name = "Unknown"
                    if D[0][0] > 0.4:
                        name = known_names[I[0][0]]
                    res.append([x, y, w, h, name])
                except: 
                    res.append([x, y, w, h, "Unknown"])
        else:
             res = faces
             
        # Update Display State
        with LATEST_FACES_LOCK:
            LATEST_FACES = res

def video_player():
    global running, paused, skip_video
    
    threading.Thread(target=detector_thread, daemon=True).start()
    threading.Thread(target=recognizer_thread, daemon=True).start()
    
    # Create window with enhanced display settings
    window_name = "InsightFace AMD (Fluid)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    try:
        # Force window to stay on top and be visible
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
    except:
        pass  # Some OpenCV builds don't support this
    
    print(f"\n{'='*50}")
    print(f"🎬 VIDEO PLAYER STARTED")
    print(f"{'='*50}")
    print(f"Controls: 'q' quit | 'p'/Space pause | 'n' next")
    print(f"{'='*50}\n")
    
    while running:
        # scan for files
        files = [f for f in os.listdir(VIDEO_DIR) 
                 if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv'))]
        files.sort() # Play in order
        
        if not files:
            print("No videos found. Waiting...")
            time.sleep(1)
            continue
            
        print(f"Playlist: {files}")
            
        for v in files:
            if not running: break
            video_path = os.path.join(VIDEO_DIR, v)
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                print(f"Failed to open {v}")
                continue
                
            print(f"Playing {v}")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"Debug: FPS={fps}")
            if fps <= 0: fps = 30.0
            target_delay = 1.0 / fps
            
            frame_count = 0
            while running:
                start_time = time.time()
                
                if skip_video:
                    skip_video = False
                    print("Skipping video...")
                    break
                    
                if paused:
                    k = cv2.waitKey(100) & 0xFF
                    if k == ord('p') or k == ord(' '): paused = not paused
                    elif k == ord('n'): skip_video = True
                    elif k == ord('q'): running = False
                    continue
                
                # print("Debug: Reading frame...")
                ret, frame = cap.read()
                if not ret: 
                    print(f"Finished {v} after {frame_count} frames.")
                    break
                frame_count += 1
                
                # Send to Detector (Non-blocking drop)
                if detect_input_queue.empty():
                    detect_input_queue.put(frame)
                
                # Render
                with LATEST_FACES_LOCK:
                    current_faces = list(LATEST_FACES)
                    
                for x,y,w,h,n in current_faces:
                     color = (0, 255, 0) if n != "Unknown" else (0, 0, 255)
                     cv2.rectangle(frame, (x,y), (x+w, y+h), color, 2)
                     cv2.putText(frame, n, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Add frame info overlay
                info_text = f"Frame: {frame_count} | Faces: {len(current_faces)} | FPS: {fps:.1f}"
                cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                     
                try:
                    cv2.imshow(window_name, frame)
                except Exception as e:
                    print(f"⚠️  Display error: {e}")
                    print("   Continuing without display...")
                    # Continue processing even if display fails
                
                # FPS Control
                elapsed = time.time() - start_time
                wait_ms = max(1, int((target_delay - elapsed) * 1000))
                
                k = cv2.waitKey(wait_ms) & 0xFF
                if k == ord('q'): 
                    running = False
                    break
                elif k == ord('p') or k == ord(' '): 
                    paused = not paused
                    print(f"Paused: {paused}")
                elif k == ord('n'): 
                    print("Next video")
                    skip_video = True
                    
            cap.release()
            
    cv2.destroyAllWindows()

if __name__ == "__main__":
    print("Controls: 'q' quit, 'p' pause, 'n' next")
    video_player()
