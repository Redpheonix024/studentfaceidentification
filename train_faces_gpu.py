import os
import cv2
import numpy as np
import pickle
import time
import shutil

# PATTERN: Monkey-patch np.object for old libraries (like tf2onnx 1.8.4) running on new numpy
try:
    if not hasattr(np, 'object'):
        np.object = object
except:
    pass

# Check/Import ONNX Runtime
try:
    import onnxruntime as ort
except ImportError:
    print("❌ onnxruntime not found. Please run: pip install onnxruntime-directml")
    exit(1)

# =========================
# CONFIG
# =========================
STUDENTS_DIR = "students faces"
MODELS_DIR = "models"
EMBEDDINGS_PATH = "models/face_embeddings_arcface.pkl"
ARCFACE_ONNX_PATH = "models/arcface.onnx"
YOLO_FACE_PATH = "models/yolov8n-face-lindevs.onnx" # From test1.py

# ArcFace Standard Input
INPUT_SIZE = (112, 112)

# =========================
# HELPER: EXPORT ARCFACE
# =========================
def export_arcface_onnx():
    print("⏳ Exporting DeepFace ArcFace model to ONNX (One-time setup)...")
    try:
        from deepface import DeepFace
        import tensorflow as tf
        import tf2onnx
    except ImportError:
        print("❌ Error: 'tf2onnx' or 'tensorflow' or 'deepface' missing.")
        print("To export the model, run: pip install tf2onnx tensorflow tf-keras")
        return False

    try:
        # Build Keras Model
        obj = DeepFace.build_model("ArcFace")
        if hasattr(obj, 'model'):
            print("ℹ️ Unwrapping DeepFace model object...")
            model = obj.model
        else:
            model = obj
        
        # Define Input Signature (Batch, 112, 112, 3)
        spec = (tf.TensorSpec((None, 112, 112, 3), tf.float32, name="input_1"),)
        
        # Convert via SavedModel (More robust for older tf2onnx)
        output_path = ARCFACE_ONNX_PATH
        temp_dir = "temp_arcface_savedmodel"
        
        print(f"ℹ️ Saving Keras model to {temp_dir}...")
        model.save(temp_dir)
        
        print("ℹ️ Converting SavedModel to ONNX...")
        # Use subprocess to call tf2onnx module directly to avoid some in-process issues
        # or use internal API. Internal API is better for error catching.
        import tf2onnx
        model_proto, _ = tf2onnx.convert.from_saved_model(temp_dir, opset=11)
        
        with open(output_path, "wb") as f:
            f.write(model_proto.SerializeToString())
            
        # Cleanup
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            
        print(f"✅ Exported to {ARCFACE_ONNX_PATH}")
        return True
    except Exception as e:
        print(f"❌ Export failed: {e}")
        return False

# =========================
# HELPER: PREPROCESSING
# =========================
def preprocess_arcface(img):
    # Resize
    img = cv2.resize(img, INPUT_SIZE)
    # Normalize (Standard ArcFace: -1 to 1 or 0 to 1? DeepFace uses (x-127.5)/128)
    img = img.astype(np.float32)
    img = (img - 127.5) / 128.0
    # CHW or HWC? ONNX usually NCHW
    # DeepFace Keras models are NHWC. 
    # BUT tf2onnx usually preserves format unless specified. 
    # Keras is NHWC. Let's assume NHWC.
    img = np.expand_dims(img, axis=0) # Add batch dimension -> (1, 112, 112, 3)
    return img

def preprocess_yolo(frame):
    # Resize to 640x640, RGB, NCHW, 0-1
    img = cv2.resize(frame, (640, 640))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1) # HWC -> CHW
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32) / 255.0
    return img

def postprocess_yolo_face(outputs, orig_shape, conf_thres=0.5):
    # YOLO Face output is usually [1, 5+Points, 8400]
    preds = np.squeeze(outputs[0]).T
    
    # Filter by confidence
    scores = preds[:, 4]
    keep = scores > conf_thres
    preds = preds[keep]
    
    if len(preds) == 0: return []
    
    # NMS
    boxes = preds[:, :4]
    scores = preds[:, 4]
    
    # Scale coords
    h, w = orig_shape
    scale_w = w / 640
    scale_h = h / 640
    
    # Convert cx,cy,w,h -> x1,y1,x2,y2
    # Adjust for scaling
    final_boxes = []
    
    # Simple NMS using OpenCV
    rects = []
    for i, box in enumerate(boxes):
        cx, cy, bw, bh = box
        x1 = (cx - bw/2) * scale_w
        y1 = (cy - bh/2) * scale_h
        # w, h
        bw = bw * scale_w
        bh = bh * scale_h
        rects.append([int(x1), int(y1), int(bw), int(bh)])
        
    indices = cv2.dnn.NMSBoxes(rects, scores.tolist(), conf_thres, 0.4)
    
    if len(indices) > 0:
        for i in indices.flatten():
            final_boxes.append(rects[i]) # x, y, w, h
            
    return final_boxes

# =========================
# MAIN
# =========================
def train_gpu():
    print("==========================================")
    print("   TRAINING ON AMD GPU (DirectML)")
    print("==========================================")
    
    # 0. Check Environment
    providers = ort.get_available_providers()
    if 'DmlExecutionProvider' not in providers:
        print("⚠️ DirectML not detected. Falling back to CPU.")
        cur_providers = ['CPUExecutionProvider']
    else:
        print("✅ Using DirectML (AMD GPU)")
        cur_providers = ['DmlExecutionProvider']

    # 1. Load/Check ArcFace Model
    if not os.path.exists(ARCFACE_ONNX_PATH):
        if not export_arcface_onnx():
            return
            
    try:
        arcface_sess = ort.InferenceSession(ARCFACE_ONNX_PATH, providers=cur_providers)
        af_input_name = arcface_sess.get_inputs()[0].name
        print(f"Loaded ArcFace: {ARCFACE_ONNX_PATH}")
    except Exception as e:
        print(f"❌ Failed to load ArcFace ONNX: {e}")
        return

    # 2. Load YOLO Face Detector (Optional if we just crop center, but detection is better)
    yolo_sess = None
    if os.path.exists(YOLO_FACE_PATH):
        try:
            yolo_sess = ort.InferenceSession(YOLO_FACE_PATH, providers=cur_providers)
            yolo_input_name = yolo_sess.get_inputs()[0].name
            print(f"Loaded Detector: {YOLO_FACE_PATH}")
        except:
            print("⚠️ Failed to load YOLO detector. Will use center crop.")
    else:
        print("⚠️ YOLO Face model not found. Will use center crop.")

    # 3. Process Students
    known_embeddings = []
    known_names = []
    
    if not os.path.exists(STUDENTS_DIR):
        print("No student directory.")
        return

    students = [d for d in os.listdir(STUDENTS_DIR) if os.path.isdir(os.path.join(STUDENTS_DIR, d))]
    print(f"Found {len(students)} students.")

    for student in students:
        spath = os.path.join(STUDENTS_DIR, student)
        images = [f for f in os.listdir(spath) if f.lower().endswith(('.jpg', '.png'))]
        
        student_embs = []
        if not images: continue
        
        print(f"Processing {student} ({len(images)})...")
        
        for img_file in images:
            img_path = os.path.join(spath, img_file)
            img = cv2.imread(img_path)
            if img is None: continue
            
            # A. Detect Face (Get best box)
            face_img = None
            if yolo_sess:
                try:
                    blob = preprocess_yolo(img)
                    outputs = yolo_sess.run(None, {yolo_input_name: blob})
                    boxes = postprocess_yolo_face(outputs, img.shape[:2])
                    if boxes:
                        # Take largest face
                        x, y, w, h = max(boxes, key=lambda b: b[2]*b[3])
                        # Clamp
                        h_img, w_img = img.shape[:2]
                        x = max(0, x); y = max(0, y)
                        w = min(w, w_img-x); h = min(h, h_img-y)
                        face_img = img[y:y+h, x:x+w]
                except Exception as e:
                    pass
            
            # Fallback to whole image if detection failed or no detector
            if face_img is None or face_img.size == 0:
                face_img = img
                
            # B. Preprocess for ArcFace
            try:
                blob = preprocess_arcface(face_img)
                # C. Inference
                emb = arcface_sess.run(None, {af_input_name: blob})[0]
                # Normalize embedding
                norm_emb = emb / np.linalg.norm(emb)
                student_embs.append(norm_emb.flatten())
            except Exception as e:
                print(f"Error on {img_file}: {e}")

        if student_embs:
            avg_emb = np.mean(student_embs, axis=0)
            avg_emb = avg_emb / np.linalg.norm(avg_emb) # Normalize average
            known_embeddings.append(avg_emb)
            known_names.append(student)
            print(f"  ✅ {student} Done.")
        else:
            print(f"  ⚠️ {student} Skipped (No faces).")

    # 4. Save
    if known_embeddings:
        with open(EMBEDDINGS_PATH, 'wb') as f:
            pickle.dump((known_embeddings, known_names), f)
        print(f"🎉 Model saved to {EMBEDDINGS_PATH}")
    else:
        print("❌ No data to save.")

if __name__ == "__main__":
    train_gpu()
