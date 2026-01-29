#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train Face Recognition Model on Class Photos
Uses RetinaFace (detection) + ArcFace R100 (recognition)
Processes subdirectory structure: students faces/<StudentName>/<photo>.jpg
"""

import os
import sys
import cv2
import numpy as np
import pickle
import onnxruntime as ort

# Fix Windows console encoding
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from datetime import datetime

# =========================
# CONFIG
# =========================
DATASET_DIR = "students faces"
MODELS_DIR = "models"
RETINAFACE_PATH = "models/retinaface_r50.onnx"
ARCFACE_PATH = "models/arcface_r100.onnx"

# Generate timestamped filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_PATH = f"models/face_embeddings_class_r100_{timestamp}.pkl"
OUTPUT_PATH_LATEST = "models/face_embeddings_class_r100.pkl"  # Also save as latest

# =========================
# MODEL CLASSES
# =========================
class SCRFD:
    """Face Detection Model - SCRFD/RetinaFace"""
    def __init__(self, model_file):
        providers = ort.get_available_providers()
        print(f"Available providers: {providers}")
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
                
        if not scores_list: return None
        
        scores = np.vstack(scores_list)
        bboxes = np.vstack(bboxes_list)
        
        keep = cv2.dnn.NMSBoxes(bboxes.tolist(), scores[:, 0].tolist(), self.conf_thresh, self.nms_thresh)
        
        if len(keep) > 0:
            best_area = 0
            best_box = None
            indices = keep.flatten()
            for k in indices:
                x1, y1, x2, y2 = bboxes[k]
                x1 /= scale; y1 /= scale; x2 /= scale; y2 /= scale
                x1 = max(0, int(x1)); y1 = max(0, int(y1))
                x2 = min(w, int(x2)); y2 = min(h, int(y2))
                area = (x2-x1)*(y2-y1)
                if area > best_area:
                    best_area = area
                    best_box = (x1, y1, x2, y2)
            return best_box
        return None

class ArcFaceONNX:
    """Face Recognition Model - ArcFace R100"""
    def __init__(self, model_file):
        providers = ort.get_available_providers()
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
# TRAINING
# =========================
def train():
    print("\n" + "="*60)
    print("  TRAINING ON CLASS PHOTOS DATASET")
    print("  Using ArcFace R100 (512-dim embeddings)")
    print("="*60 + "\n")
    
    if not os.path.exists(RETINAFACE_PATH) or not os.path.exists(ARCFACE_PATH):
        print("❌ Models missing.")
        return
    
    print("📦 Loading models...")
    detector = SCRFD(RETINAFACE_PATH)
    print("✅ RetinaFace loaded")
    recognizer = ArcFaceONNX(ARCFACE_PATH)
    print("✅ ArcFace R100 loaded\n")
    
    known_embeddings = []
    known_names = []
    
    if not os.path.exists(DATASET_DIR):
        print(f"❌ Directory not found: {DATASET_DIR}")
        return
    
    # Process subdirectories
    student_dirs = [d for d in os.listdir(DATASET_DIR) 
                   if os.path.isdir(os.path.join(DATASET_DIR, d))]
    
    print(f"📁 Found {len(student_dirs)} student directories\n")
    
    total_images = 0
    total_faces = 0
    total_failed = 0
    
    for student_name in sorted(student_dirs):
        student_path = os.path.join(DATASET_DIR, student_name)
        
        # Find all image files in this student's directory
        image_files = [f for f in os.listdir(student_path) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not image_files:
            print(f"⚠️  {student_name}: No images found")
            continue
        
        total_images += len(image_files)
        student_faces = 0
        
        print(f"👤 {student_name} ({len(image_files)} photos)")
        
        for img_file in image_files:
            img_path = os.path.join(student_path, img_file)
            img = cv2.imread(img_path)
            
            if img is None:
                print(f"   ⚠️  Failed to load: {img_file}")
                total_failed += 1
                continue
            
            # Detect face
            box = detector.detect(img)
            
            if box:
                x1, y1, x2, y2 = box
                face_img = img[y1:y2, x1:x2]
                
                if face_img.size == 0:
                    print(f"   ⚠️  Empty crop: {img_file}")
                    total_failed += 1
                    continue
                
                try:
                    emb = recognizer.get_embedding(face_img)
                    known_embeddings.append(emb)
                    known_names.append(student_name)
                    student_faces += 1
                    total_faces += 1
                except Exception as e:
                    print(f"   ❌ Error processing {img_file}: {e}")
                    total_failed += 1
            else:
                # No face detected - try using full image as fallback
                try:
                    emb = recognizer.get_embedding(img)
                    known_embeddings.append(emb)
                    known_names.append(student_name)
                    student_faces += 1
                    total_faces += 1
                    print(f"   ⚠️  No face in {img_file}, used full image")
                except Exception as e:
                    print(f"   ❌ Failed {img_file}: {e}")
                    total_failed += 1
        
        print(f"   ✅ Generated {student_faces} embeddings\n")
    
    # Save embeddings
    if known_embeddings:
        with open(OUTPUT_PATH, 'wb') as f:
            pickle.dump((known_embeddings, known_names), f)
        
        # Also save as "latest"
        with open(OUTPUT_PATH_LATEST, 'wb') as f:
            pickle.dump((known_embeddings, known_names), f)
        
        print("\n" + "="*60)
        print("  TRAINING COMPLETE")
        print("="*60)
        print(f"✅ Total images processed: {total_images}")
        print(f"✅ Faces detected: {total_faces}")
        print(f"⚠️  Failed: {total_failed}")
        print(f"✅ Unique students: {len(set(known_names))}")
        print(f"✅ Total embeddings: {len(known_embeddings)}")
        print(f"📁 Saved:")
        print(f"   - {OUTPUT_PATH} (timestamped)")  
        print(f"   - {OUTPUT_PATH_LATEST} (latest)")
        print("="*60 + "\n")
    else:
        print("❌ No embeddings generated.")

if __name__ == "__main__":
    train()
