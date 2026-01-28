#!/usr/bin/env python3
"""
Model Testing Script - MULTITHREADED Video Playback
Smooth video playback with background face detection and recognition
"""

import os
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
    elif 'CUDAExecutionProvider' in providers:
        print("✅ Using CUDA (NVIDIA GPU)")
        PROVIDERS = ['CUDAExecutionProvider']
    else:
        print("⚠️  Using CPU")
        PROVIDERS = ['CPUExecutionProvider']
except ImportError:
    print("❌ onnxruntime not found")
    PROVIDERS = ['CPUExecutionProvider']
    ort = None

if cv2.ocl.haveOpenCL():
    cv2.ocl.setUseOpenCL(True)
    print("✅ OpenCL enabled for GPU rendering")

# =========================
# CONFIG
# =========================
VIDEO_DIR = "videos"
RETINAFACE_PATH = "models/retinaface_r50.onnx"
ARCFACE_PATH = "models/arcface_r100.onnx"

# =========================
# MODEL CLASSES
# =========================
class SCRFD:
    def __init__(self, model_file, providers):
        self.session = ort.InferenceSession(model_file, providers=providers)
        self.conf_thresh = 0.5
        self.nms_thresh = 0.4
        
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
                
                final_faces.append([x1, y1, width_box, height_box, scores[k][0]])
                
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
# THREADING
# =========================
detect_queue = queue.Queue(maxsize=1)
recog_queue = queue.Queue(maxsize=1)
LATEST_FACES = []
LATEST_FACES_LOCK = threading.Lock()
running = True

def detector_thread(detector):
    global running, LATEST_FACES
    while running:
        try:
            frame = detect_queue.get(timeout=0.1)
            if frame is None: break
            
            faces = detector.detect(frame)
            
            if faces and recog_queue.empty():
                recog_queue.put((frame.copy(), faces))
            elif faces:
                with LATEST_FACES_LOCK:
                    LATEST_FACES = [(x, y, w, h, "Unknown", conf) for x, y, w, h, conf in faces]
        except queue.Empty:
            continue

def recognizer_thread(recognizer, index, known_names, confidence_threshold):
    global running, LATEST_FACES
    while running:
        try:
            item = recog_queue.get(timeout=0.1)
            if item is None: break
            
            frame, faces = item
            result = []
            
            for x, y, w, h, conf in faces:
                name = "Unknown"
                
                if recognizer and index:
                    try:
                        crop = frame[y:y+h, x:x+w]
                        if crop.size > 0:
                            emb = recognizer.get_embedding(crop)
                            qn = np.array([emb], dtype=np.float32)
                            faiss.normalize_L2(qn)
                            D, I = index.search(qn, 1)
                            
                            if D[0][0] > confidence_threshold:
                                name = known_names[I[0][0]]
                    except:
                        pass
                
                result.append((x, y, w, h, name, conf))
            
            with LATEST_FACES_LOCK:
                LATEST_FACES = result
        except queue.Empty:
            continue

# =========================
# MAIN
# =========================
def main():
    global running
    
    parser = argparse.ArgumentParser(description='Test models - Multithreaded')
    parser.add_argument('--model', type=int, default=5, choices=[1, 2, 3, 4, 5],
                        help='Model: 1=Class, 2=Passport Legacy, 3=Merged, 4=Passport R100, 5=Merged R100 (default)')
    parser.add_argument('--no-recognition', action='store_true')
    parser.add_argument('--confidence', type=float, default=0.4)
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("  MULTITHREADED MODEL TESTER")
    print("="*60)
    
    embeddings_options = {
        1: "models/face_embeddings_arcface.pkl",
        2: "models/face_embeddings_passport.pkl",
        3: "models/face_embeddings_merged.pkl",
        4: "models/face_embeddings_passport_r100.pkl",
        5: "models/face_embeddings_merged_r100.pkl"
    }
    
    EMBEDDINGS_PATH = embeddings_options.get(args.model, embeddings_options[5])
    print(f"\n📊 Model: {args.model}")
    print(f"📁 Embeddings: {EMBEDDINGS_PATH}")
    
    # Load embeddings
    known_embeddings = []
    known_names = []
    index = None
    
    if not args.no_recognition and os.path.exists(EMBEDDINGS_PATH):
        with open(EMBEDDINGS_PATH, 'rb') as f:
            known_embeddings, known_names = pickle.load(f)
        if known_embeddings:
            dim = len(known_embeddings[0])
            db_emb = np.array(known_embeddings, dtype=np.float32)
            faiss.normalize_L2(db_emb)
            index = faiss.IndexFlatIP(dim)
            index.add(db_emb)
            print(f"✅ Loaded {len(known_embeddings)} embeddings")
    
    # Load models
    print("\n📦 Loading models...")
    detector = SCRFD(RETINAFACE_PATH, PROVIDERS)
    print("✅ RetinaFace loaded")
    
    recognizer = None
    if not args.no_recognition and index:
        recognizer = ArcFaceONNX(ARCFACE_PATH, PROVIDERS)
        print("✅ ArcFace loaded")
    
    # Start threads
    threading.Thread(target=detector_thread, args=(detector,), daemon=True).start()
    threading.Thread(target=recognizer_thread, args=(recognizer, index, known_names, args.confidence), daemon=True).start()
    
    # Find videos
    video_files = sorted([f for f in os.listdir(VIDEO_DIR) if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))])
    print(f"\n🎬 Found {len(video_files)} videos")
    
    window_name = "Multithreaded Model Tester"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    try:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
    except:
        pass
    
    print("\n" + "="*60)
    print("🎮 Controls: 'q'=Quit | 'p'/Space=Pause | 'n'=Next")
    print("="*60 + "\n")
    
    paused = False
    skip_video = False
    
    for video_file in video_files:
        video_path = os.path.join(VIDEO_DIR, video_file)
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            continue
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: fps = 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\n▶️  {video_file} | FPS: {fps:.1f} | Frames: {total_frames}")
        
        frame_count = 0
        target_delay = 1.0 / fps
        
        while running:
            if skip_video:
                skip_video = False
                break
            
            if paused:
                key = cv2.waitKey(100) & 0xFF
                if key == ord('p') or key == ord(' '): paused = False
                elif key == ord('n'): skip_video = True
                elif key == ord('q'): running = False
                continue
            
            start_time = time.time()
            
            ret, frame = cap.read()
            if not ret:
                print(f"✅ Finished: {frame_count} frames")
                break
            
            frame_count += 1
            
            # Send to detector (non-blocking)
            if detect_queue.empty():
                detect_queue.put(frame.copy())
            
            # Render with latest faces
            with LATEST_FACES_LOCK:
                current_faces = list(LATEST_FACES)
            
            for x, y, w, h, name, conf in current_faces:
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                label = f"{name} ({conf:.2f})"
                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x, y-label_h-10), (x+label_w, y), color, -1)
                cv2.putText(frame, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Info overlay
            info = f"{video_file} | Frame: {frame_count}/{total_frames} | Faces: {len(current_faces)}"
            cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
            
            try:
                cv2.imshow(window_name, frame)
            except:
                pass
            
            # FPS control
            elapsed = time.time() - start_time
            wait_ms = max(1, int((target_delay - elapsed) * 1000))
            
            key = cv2.waitKey(wait_ms) & 0xFF
            if key == ord('q'):
                running = False
                break
            elif key == ord('p') or key == ord(' '):
                paused = True
            elif key == ord('n'):
                skip_video = True
        
        cap.release()
        if not running:
            break
    
    running = False
    cv2.destroyAllWindows()
    print("\n✅ Done!\n")

if __name__ == "__main__":
    main()
