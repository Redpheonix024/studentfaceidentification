#!/usr/bin/env python3
"""
Model Testing Script - Video Playback with Face Detection
Tests face detection and recognition models on video files
No training - just testing/demonstration
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
from face_storage import FaceTracker

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
    print("❌ onnxruntime not found. Install: pip install onnxruntime-directml")
    PROVIDERS = ['CPUExecutionProvider']
    ort = None

# Enable OpenCL for GPU-accelerated rendering
if cv2.ocl.haveOpenCL():
    cv2.ocl.setUseOpenCL(True)
    print("✅ OpenCL enabled for GPU rendering")

# =========================
# CONFIG
# =========================
VIDEO_DIR = "videos"
MODELS_DIR = "models"
RETINAFACE_PATH = "models/retinaface_r50.onnx"
ARCFACE_PATH = "models/arcface_r100.onnx"

# =========================
# MODEL CLASSES
# =========================
class SCRFD:
    """Face Detection Model - SCRFD/RetinaFace"""
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
                
                final_faces.append([x1, y1, width_box, height_box, scores[k][0]])
                
        return final_faces

class ArcFaceONNX:
    """Face Recognition Model - ArcFace"""
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
# THREADING SUPPORT
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
            
            # Send to recognizer if not busy
            if faces and recog_queue.empty():
                recog_queue.put((frame.copy(), faces))
            elif faces:
                # Update display with unknown faces
                with LATEST_FACES_LOCK:
                    LATEST_FACES = [(x, y, w, h, "Unknown", conf) for x, y, w, h, conf in faces]
        except queue.Empty:
            continue
        except Exception as e:
            print(f"Detector error: {e}")

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
        except Exception as e:
            print(f"Recognizer error: {e}")

# =========================
# MAIN TEST FUNCTION
# =========================
def main():
    parser = argparse.ArgumentParser(description='Test face detection and recognition models')
    parser.add_argument('--model', type=str, default=None,
                        help='Model name or number to use (e.g., "merged_r100" or "1")')
    parser.add_argument('--no-recognition', action='store_true',
                        help='Skip face recognition, only do detection')
    parser.add_argument('--confidence', type=float, default=0.4,
                        help='Recognition confidence threshold (default: 0.4)')
    parser.add_argument('--save-csv', action='store_true',
                        help='Save consistently detected faces to CSV file')
    parser.add_argument('--csv-path', type=str, default='detected_faces.csv',
                        help='Path to CSV file for saving faces (default: detected_faces.csv)')
    parser.add_argument('--consistency-frames', type=int, default=5,
                        help='Number of consecutive frames needed for consistent detection (default: 5)')
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("  FACE DETECTION & RECOGNITION MODEL TESTER")
    print("="*60)
    
    # Dynamically discover all model files in models directory
    models_dir = "models"
    if not os.path.exists(models_dir):
        print(f"\n[ERROR] Models directory not found: {models_dir}")
        return
    
    # Find all .pkl files
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
    
    if not model_files:
        print(f"\n[ERROR] No .pkl model files found in {models_dir}")
        return
    
    # Load info about each model
    available_models = []
    for i, filename in enumerate(sorted(model_files), start=1):
        model_path = os.path.join(models_dir, filename)
        
        # Extract display name from filename
        display_name = filename.replace('face_embeddings_', '').replace('.pkl', '').replace('_', ' ').title()
        
        # Get file size
        size_bytes = os.path.getsize(model_path)
        size_kb = size_bytes / 1024
        
        # Try to load and get stats
        num_embeddings = 0
        num_students = 0
        student_names = []
        
        try:
            with open(model_path, 'rb') as f:
                embeddings, names = pickle.load(f)
            num_embeddings = len(embeddings)
            num_students = len(set(names))
            student_names = sorted(set(names))
        except Exception as e:
            # If can't load, just show file info
            pass
        
        available_models.append({
            'number': i,
            'filename': filename,
            'path': model_path,
            'display_name': display_name,
            'size_kb': size_kb,
            'num_embeddings': num_embeddings,
            'num_students': num_students,
            'student_names': student_names
        })
    
    # If no model specified, ask user interactively
    selected_model = None
    
    if args.model is None:
        print("\n" + "="*60)
        print("  SELECT A MODEL")
        print("="*60)
        print(f"\nFound {len(available_models)} models in {models_dir}/\n")
        
        # Display all available models
        for model in available_models:
            print(f"  [{model['number']}] {model['display_name']}")
            print(f"      File: {model['filename']}")
            if model['num_students'] > 0:
                print(f"      Stats: {model['num_students']} students, {model['num_embeddings']} embeddings, {model['size_kb']:.1f} KB")
                if model['num_students'] <= 10:
                    print(f"      Students: {', '.join(model['student_names'])}")
            else:
                print(f"      Size: {model['size_kb']:.1f} KB")
            print()
        
        # Get user selection
        while True:
            try:
                choice = input("Select model number (or 'q' to quit): ").strip()
                if choice.lower() == 'q':
                    print("\nExiting...")
                    return
                
                model_num = int(choice)
                if 1 <= model_num <= len(available_models):
                    selected_model = available_models[model_num - 1]
                    break
                else:
                    print(f"[ERROR] Invalid selection. Choose 1-{len(available_models)}")
            except ValueError:
                print("[ERROR] Please enter a number or 'q' to quit")
            except KeyboardInterrupt:
                print("\n\nExiting...")
                return
    else:
        # User specified model via command line
        # Try to match by number or by name
        try:
            model_num = int(args.model)
            if 1 <= model_num <= len(available_models):
                selected_model = available_models[model_num - 1]
        except ValueError:
            # Not a number, try to match by filename
            search_term = args.model.lower()
            for model in available_models:
                if search_term in model['filename'].lower() or search_term in model['display_name'].lower():
                    selected_model = model
                    break
        
        if not selected_model:
            print(f"\n[ERROR] Model not found: {args.model}")
            print(f"Available models: {', '.join([m['filename'] for m in available_models])}")
            return
    
    # Display selected model info
    EMBEDDINGS_PATH = selected_model['path']
    
    print(f"\n[SELECTED MODEL]")
    print(f"  Number: {selected_model['number']}")
    print(f"  Name: {selected_model['display_name']}")
    print(f"  File: {selected_model['filename']}")
    print(f"  Path: {EMBEDDINGS_PATH}")
    
    # Load face embeddings database
    known_embeddings = []
    known_names = []
    index = None
    
    if not args.no_recognition:
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
                    print(f"\n[OK] Loaded {len(known_embeddings)} face embeddings")
                    print(f"     Unique students: {len(set(known_names))}")
                    if len(set(known_names)) <= 30:
                        print(f"     Students: {', '.join(sorted(set(known_names)))}")
                else:
                    print("\n[WARNING] No embeddings found in file")
            except Exception as e:
                print(f"\n[ERROR] Error loading embeddings: {e}")
        else:
            print(f"\n[WARNING] Embeddings file not found: {EMBEDDINGS_PATH}")
            print("   Running in detection-only mode")
    
    # Load models
    print("\n📦 Loading models...")
    
    if not os.path.exists(RETINAFACE_PATH):
        print(f"❌ Detection model not found: {RETINAFACE_PATH}")
        return
    detector = SCRFD(RETINAFACE_PATH, PROVIDERS)
    print("✅ Face detector loaded (RetinaFace)")
    
    recognizer = None
    if not args.no_recognition and index is not None:
        if not os.path.exists(ARCFACE_PATH):
            print(f"⚠️  Recognition model not found: {ARCFACE_PATH}")
            print("   Running in detection-only mode")
        else:
            recognizer = ArcFaceONNX(ARCFACE_PATH, PROVIDERS)
            print("✅ Face recognizer loaded (ArcFace)")
    
    # Initialize face tracker for CSV storage if enabled
    face_tracker = None
    if args.save_csv:
        face_tracker = FaceTracker(
            csv_path=args.csv_path,
            consistency_frames=args.consistency_frames,
            iou_threshold=0.5,
            timeout_seconds=2.0
        )
        print(f"✅ Face tracker enabled - saving to {args.csv_path}")
        print(f"   Consistency threshold: {args.consistency_frames} frames")

    
    # Find videos
    if not os.path.exists(VIDEO_DIR):
        print(f"\n❌ Video directory not found: {VIDEO_DIR}")
        return
        
    video_files = [f for f in os.listdir(VIDEO_DIR) 
                   if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv'))]
    
    if not video_files:
        print(f"\n❌ No video files found in {VIDEO_DIR}")
        return
    
    video_files.sort()
    print(f"\n🎬 Found {len(video_files)} videos: {video_files}")
    
    # Create display window
    window_name = "Model Tester - Face Detection & Recognition"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    try:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
    except:
        pass
    
    print("\n" + "="*60)
    print("🎮 CONTROLS:")
    print("   'q' = Quit")
    print("   'p' or Space = Pause/Resume")
    print("   'n' = Next video")
    print("="*60 + "\n")
    
    # Process videos
    paused = False
    skip_video = False
    
    for video_file in video_files:
        video_path = os.path.join(VIDEO_DIR, video_file)
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"⚠️  Failed to open: {video_file}")
            continue
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0: fps = 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\n▶️  Playing: {video_file}")
        print(f"   FPS: {fps:.1f} | Total frames: {total_frames}")
        
        frame_count = 0
        total_faces_detected = 0
        total_faces_recognized = 0
        target_delay = 1.0 / fps
        
        while True:
            if skip_video:
                skip_video = False
                break
            
            if paused:
                key = cv2.waitKey(100) & 0xFF
                if key == ord('p') or key == ord(' '):
                    paused = False
                    print("▶️  Resumed")
                elif key == ord('n'):
                    skip_video = True
                elif key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    print("\n👋 Exiting...")
                    return
                continue
            
            start_time = time.time()
            
            ret, frame = cap.read()
            if not ret:
                print(f"✅ Finished: {video_file}")
                print(f"   Processed {frame_count} frames")
                print(f"   Detected {total_faces_detected} faces total")
                if recognizer:
                    print(f"   Recognized {total_faces_recognized} faces")
                break
            
            frame_count += 1
            
            # Detect faces
            faces = detector.detect(frame)
            total_faces_detected += len(faces)
            
            # Recognize faces and prepare for tracking
            tracked_faces = []
            for i, face_data in enumerate(faces):
                x, y, w, h, conf = face_data
                
                name = "Unknown"
                color = (0, 0, 255)  # Red for unknown
                
                if recognizer and index is not None:
                    try:
                        crop = frame[y:y+h, x:x+w]
                        if crop.size > 0:
                            emb = recognizer.get_embedding(crop)
                            qn = np.array([emb], dtype=np.float32)
                            faiss.normalize_L2(qn)
                            D, I = index.search(qn, 1)
                            
                            if D[0][0] > args.confidence:
                                name = known_names[I[0][0]]
                                color = (0, 255, 0)  # Green for recognized
                                total_faces_recognized += 1
                    except:
                        pass
                
                # Store for tracking (x, y, w, h, name, confidence)
                tracked_faces.append((x, y, w, h, name, conf))
                
                # Draw bounding box
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                
                # Draw label with background
                label = f"{name} ({conf:.2f})"
                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, (x, y-label_h-10), (x+label_w, y), color, -1)
                cv2.putText(frame, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Update face tracker if enabled
            if face_tracker and tracked_faces:
                face_tracker.update(tracked_faces)
            
            # Draw info overlay
            info_lines = [
                f"Video: {video_file}",
                f"Frame: {frame_count}/{total_frames}",
                f"Faces: {len(faces)}",
                f"FPS: {fps:.1f}"
            ]
            
            y_offset = 30
            for line in info_lines:
                cv2.putText(frame, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
                y_offset += 30
            
            # Display frame
            try:
                cv2.imshow(window_name, frame)
            except Exception as e:
                print(f"⚠️  Display error: {e}")
            
            # Control FPS
            elapsed = time.time() - start_time
            wait_ms = max(1, int((target_delay - elapsed) * 1000))
            
            key = cv2.waitKey(wait_ms) & 0xFF
            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                print("\n👋 Exiting...")
                return
            elif key == ord('p') or key == ord(' '):
                paused = True
                print("⏸️  Paused")
            elif key == ord('n'):
                skip_video = True
        
        cap.release()
    
    cv2.destroyAllWindows()
    print("\n" + "="*60)
    print("✅ All videos processed!")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
