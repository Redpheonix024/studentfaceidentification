import os
import cv2
import numpy as np
import pickle
import time
import math
from collections import defaultdict
from deepface import DeepFace

# Check Environment
try:
    import onnxruntime as ort
except ImportError:
    print("❌ onnxruntime not found. Please run: pip install onnxruntime-directml")
    exit(1)

# =========================
# CONFIG
# =========================
VIDEO_DIR = "videos"
MODELS_DIR = "models"
EMBEDDINGS_PATH = "models/face_embeddings_arcface.pkl"
YOLO_ONNX_PATH = "yolov8n.onnx"

CONF_THRES = 0.5    # Detection confidence
RECOG_THRES = 0.4   # Recognition similarity threshold
UPDATE_WEIGHT = 0.1 # How much the new video data affects the old model (0.1 = 10% new, 90% old)

# =========================
# MODEL LOADING
# =========================
def load_models():
    # Load YOLO (ONNX) - CPU Mode
    print("ℹ️ Using CPU for Detection")
    providers = ['CPUExecutionProvider']

    try:
        yolo_sess = ort.InferenceSession(YOLO_ONNX_PATH, providers=providers)
        yolo_input = yolo_sess.get_inputs()[0].name
        yolo_outputs = [x.name for x in yolo_sess.get_outputs()]
    except Exception as e:
        print(f"❌ Failed to load YOLO: {e}")
        exit(1)
        
    return yolo_sess, yolo_input, yolo_outputs

def load_known_faces():
    if os.path.exists(EMBEDDINGS_PATH):
        try:
            with open(EMBEDDINGS_PATH, 'rb') as f:
                known_embeddings, known_names = pickle.load(f)
            return list(known_embeddings), list(known_names)
        except Exception as e:
            print(f"❌ Error loading embeddings: {e}")
            return [], []
    else:
        print("⚠️ No existing model found. Starting fresh.")
        return [], []

# =========================
# HELPERS
# =========================
def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def preprocess_yolo(frame):
    img = cv2.resize(frame, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32) / 255.0
    return img

def postprocess_yolo(outputs, orig_shape):
    # YOLOv8 Person (Class 0)
    preds = np.squeeze(outputs[0]).T # [8400, 84]
    scores = preds[:, 4] # Class 0 confidence
    keep = scores > CONF_THRES
    preds = preds[keep]
    scores = scores[keep]
    
    if len(preds) == 0: return []
    
    boxes = preds[:, :4] # cx, cy, w, h
    h, w = orig_shape
    scale_w = w / 640
    scale_h = h / 640
    
    final_boxes = []
    rects = []
    
    for i, box in enumerate(boxes):
        cx, cy, bw, bh = box
        x1 = (cx - bw/2) * scale_w
        y1 = (cy - bh/2) * scale_h
        bw = bw * scale_w
        bh = bh * scale_h
        rects.append([int(x1), int(y1), int(bw), int(bh)])
        
    indices = cv2.dnn.NMSBoxes(rects, scores.tolist(), CONF_THRES, 0.5)
    
    if len(indices) > 0:
        for i in indices.flatten():
            # Returns x1, y1, x2, y2
            r = rects[i]
            final_boxes.append([r[0], r[1], r[0]+r[2], r[1]+r[3]])
            
    return final_boxes

# =========================
# TRACKER
# =========================
class SimpleTracker:
    def __init__(self):
        self.tracks = {} 
        self.next_id = 0
        
    def update(self, detected_boxes, frame_idx, frame):
        # detected_boxes: [[x1, y1, x2, y2], ...]
        
        updated_track_ids = []
        
        # 1. Match existing
        for det_box in detected_boxes:
            dx1, dy1, dx2, dy2 = det_box
            dcx, dcy = (dx1+dx2)/2, (dy1+dy2)/2
            
            best_id = -1
            min_dist = 100 # pixels
            
            for tid, track in self.tracks.items():
                if tid in updated_track_ids: continue 
                if frame_idx - track['last_seen'] > 10: continue 
                
                tx1, ty1, tx2, ty2 = track['bbox']
                tcx, tcy = (tx1+tx2)/2, (ty1+ty2)/2
                
                dist = math.sqrt((dcx-tcx)**2 + (dcy-tcy)**2)
                if dist < min_dist:
                    min_dist = dist
                    best_id = tid
            
            if best_id != -1:
                # Update Track
                self.tracks[best_id]['bbox'] = det_box
                self.tracks[best_id]['last_seen'] = frame_idx
                self.tracks[best_id]['hits'] += 1
                updated_track_ids.append(best_id)
                self.extract_embedding(best_id, frame, det_box)
            else:
                # New Track
                new_id = self.next_id
                self.next_id += 1
                self.tracks[new_id] = {
                    'bbox': det_box,
                    'embeddings': [],
                    'last_seen': frame_idx,
                    'hits': 1
                }
                updated_track_ids.append(new_id)
                self.extract_embedding(new_id, frame, det_box)
                
    def extract_embedding(self, track_id, frame, box):
        x1, y1, x2, y2 = box
        h, w = frame.shape[:2]
        # Clamp
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x2-x1 < 20 or y2-y1 < 20: return 
        
        face_crop = frame[y1:y2, x1:x2]
        
        try:
            # Use DeepFace directly (ArcFace model)
            embedding_objs = DeepFace.represent(
                img_path = face_crop,
                model_name = "ArcFace",
                enforce_detection = False,
                detector_backend = "skip"
            )
            emb = embedding_objs[0]["embedding"]
            norm_emb = emb / np.linalg.norm(emb)
            self.tracks[track_id]['embeddings'].append(norm_emb)
        except:
            pass

# =========================
# MAIN LOOP
# =========================
def main():
    print("🚀 Starting Video Reinforcement Learning (DeepFace Mode)...")
    
    # Init Models
    yolo_sess, yolo_in, yolo_out = load_models()
    known_emb, known_names = load_known_faces()
    
    if not known_names:
        print("⚠️ Warning: No existing students known. This script calculates averages but needs initial labels to reinforce specific people.")
        print("   If you want to discover NEW people, we can group them as 'Person_1', 'Person_2' etc.")
    
    # Scan Videos
    videos = [f for f in os.listdir(VIDEO_DIR) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
    if not videos:
        print("❌ No videos found in 'videos/' directory.")
        return

    tracker = SimpleTracker()
    
    for video_file in videos:
        path = os.path.join(VIDEO_DIR, video_file)
        print(f"\n🎥 Processing: {video_file}")
        
        cap = cv2.VideoCapture(path)
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame_idx += 1
            
            if frame_idx % 2 != 0: continue # Process every 2nd frame for speed
            
            # Detect
            blob = preprocess_yolo(frame)
            outputs = yolo_sess.run(yolo_out, {yolo_in: blob})
            boxes = postprocess_yolo(outputs, frame.shape[:2])
            
            # Track & Extract
            tracker.update(boxes, frame_idx, frame)
            
            # Visualization (Blind run is faster, but let's show progress)
            if frame_idx % 30 == 0:
                print(f"   Processing frame {frame_idx} (Active Tracks: {len(tracker.tracks)})...", end='\r')
                
        cap.release()
        
    print("\n\n🧠 Consolidating Knowledge...")
    
    # Process Tracks
    # 1. Identify each track
    # 2. Update global model
    
    updates_count = defaultdict(int)
    
    for tid, track in tracker.tracks.items():
        if len(track['embeddings']) < 5: continue # Ignore short noise tracks
        
        # Average embedding for this track
        track_embs = np.array(track['embeddings'])
        avg_track_emb = np.mean(track_embs, axis=0)
        avg_track_emb = avg_track_emb / np.linalg.norm(avg_track_emb)
        
        # Who is this?
        best_name = "Unknown"
        best_score = 0
        best_idx = -1
        
        for i, (kemb, kname) in enumerate(zip(known_emb, known_names)):
            score = cosine(avg_track_emb, kemb)
            if score > best_score:
                best_score = score
                best_name = kname
                best_idx = i
                
        if best_score > RECOG_THRES:
            # We are confident this track is 'best_name'
            # REINFORCEMENT: Update the known embedding
            print(f"   Reinforcing '{best_name}' with Track #{tid} ({len(track['embeddings'])} frames, Score: {best_score:.2f})")
            
            # Weighted average update
            old_emb = known_emb[best_idx]
            new_emb = (old_emb * (1.0 - UPDATE_WEIGHT)) + (avg_track_emb * UPDATE_WEIGHT)
            new_emb = new_emb / np.linalg.norm(new_emb) # Re-normalize
            
            known_emb[best_idx] = new_emb
            updates_count[best_name] += 1
        else:
            print(f"   ❓ Track #{tid} is Unknown (Best: {best_name}, Score: {best_score:.2f}). Ignored.")
            
    # Save Result
    print("\n==========================================")
    if updates_count:
        print("📊 Summary of Reinforcement:")
        for name, count in updates_count.items():
            print(f"   - {name}: Updated with {count} video tracks.")
            
        with open(EMBEDDINGS_PATH, 'wb') as f:
            pickle.dump((known_emb, known_names), f)
        print(f"✅ Model saved to {EMBEDDINGS_PATH}")
    else:
        print("⚠️ No updates made (No confident matches found to reinforce).")

if __name__ == "__main__":
    main()
