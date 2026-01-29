#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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

# =========================
# CONFIG
# =========================
PASSPORT_DIR = "student faces paasport"
MODELS_DIR = "models"
RETINAFACE_PATH = "models/retinaface_r50.onnx"
ARCFACE_PATH = "models/arcface_r100.onnx"
OUTPUT_PATH = "models/face_embeddings_passport_r100.pkl"

# =========================
# MODEL CLASSES
# =========================
class SCRFD:
    def __init__(self, model_file):
        providers = ort.get_available_providers()
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
        
        # Logic for det_10g.onnx (buffalo_l):
        # Outputs are Grouped: [Score8, Score16, Score32, BBox8, BBox16, BBox32, KPS8, KPS16, KPS32]
        fmc = 3 # number of strides
        strides = [8, 16, 32]
        
        for i, stride in enumerate(strides):
            score = outputs[i]
            bbox = outputs[i + fmc]
            # kps = outputs[i + fmc*2]
            
            # Remove batch dim if present (outputs were (N, 1) in inspection, no batch dim?)
            # Inspection: (12800, 1). So no batch dim.
            if len(score.shape) == 3: score = score[0]
            if len(bbox.shape) == 3: bbox = bbox[0]
            
            h_fmap = 640 // stride
            w_fmap = 640 // stride
            
            anchor_centers = np.stack(np.meshgrid(np.arange(w_fmap), np.arange(h_fmap)), axis=-1)
            anchor_centers = anchor_centers.reshape(-1, 2) * stride
            
            # Validate shapes
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
    print("🚀 Starting Training (Passport Photos)...")
    
    if not os.path.exists(RETINAFACE_PATH) or not os.path.exists(ARCFACE_PATH):
        print("❌ Models missing.")
        return

    detector = SCRFD(RETINAFACE_PATH)
    recognizer = ArcFaceONNX(ARCFACE_PATH)
    
    known_embeddings = []
    known_names = []
    
    if not os.path.exists(PASSPORT_DIR):
        print(f"❌ Directory not found: {PASSPORT_DIR}")
        return

    files = [f for f in os.listdir(PASSPORT_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Found {len(files)} images.")

    for f in files:
        name = os.path.splitext(f)[0]
        path = os.path.join(PASSPORT_DIR, f)
        img = cv2.imread(path)
        if img is None: continue
        
        box = detector.detect(img)
        
        face_img = img
        if box:
            x1, y1, x2, y2 = box
            face_img = img[y1:y2, x1:x2]
            print(f"  ✅ Detected Face in {name}")
        else:
            print(f"  ⚠️ No face detected in {name}, using full image.")
        
        if face_img.size == 0: continue

        try:
            emb = recognizer.get_embedding(face_img)
            known_embeddings.append(emb)
            known_names.append(name)
        except Exception as e:
            print(f"  ❌ Error processing {name}: {e}")

    if known_embeddings:
        with open(OUTPUT_PATH, 'wb') as f:
            pickle.dump((known_embeddings, known_names), f)
        print(f"🎉 Saved {len(known_embeddings)} items to {OUTPUT_PATH}")
    else:
        print("❌ No embeddings generated.")

if __name__ == "__main__":
    train()
