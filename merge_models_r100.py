#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create Merged Model by combining existing embeddings
Merges: Passport R100 + Class R100
"""

import sys
import pickle
from datetime import datetime

# Fix Windows console encoding
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# =========================
# CONFIG
# =========================
PASSPORT_MODEL = "models/face_embeddings_passport_r100.pkl"
CLASS_MODEL = "models/face_embeddings_class_r100.pkl"
OUTPUT_MODEL = "models/face_embeddings_merged_r100.pkl"

# Generate timestamped backup filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
BACKUP_MODEL = f"models/face_embeddings_merged_r100_{timestamp}.pkl"

def merge_models():
    print("\n" + "="*60)
    print("  MERGING MODELS")
    print("  Passport R100 + Class R100 → Merged R100")
    print("="*60 + "\n")
    
    all_embeddings = []
    all_names = []
    
    # Load passport model
    print(f"📁 Loading: {PASSPORT_MODEL}")
    try:
        with open(PASSPORT_MODEL, 'rb') as f:
            passport_emb, passport_names = pickle.load(f)
        print(f"   ✅ Loaded {len(passport_emb)} passport embeddings")
        all_embeddings.extend(passport_emb)
        all_names.extend(passport_names)
    except Exception as e:
        print(f"   ⚠️  Warning: {e}")
    
    # Load class model
    print(f"📁 Loading: {CLASS_MODEL}")
    try:
        with open(CLASS_MODEL, 'rb') as f:
            class_emb, class_names = pickle.load(f)
        print(f"   ✅ Loaded {len(class_emb)} class embeddings")
        all_embeddings.extend(class_emb)
        all_names.extend(class_names)
    except Exception as e:
        print(f"   ⚠️  Warning: {e}")
    
    # Save merged model
    if all_embeddings:
        # Save timestamped backup
        with open(BACKUP_MODEL, 'wb') as f:
            pickle.dump((all_embeddings, all_names), f)
        
        # Save to main filename (for scripts to use)
        with open(OUTPUT_MODEL, 'wb') as f:
            pickle.dump((all_embeddings, all_names), f)
        
        print("\n" + "="*60)
        print("  MERGE COMPLETE")
        print("="*60)
        print(f"✅ Total embeddings: {len(all_embeddings)}")
        print(f"✅ Unique students: {len(set(all_names))}")
        print(f"📁 Saved to: {OUTPUT_MODEL}")
        print(f"📁 Backup: {BACKUP_MODEL}")
        print("="*60 + "\n")
        
        # Show student breakdown
        from collections import Counter
        student_counts = Counter(all_names)
        print("📊 Embeddings per student:")
        for name, count in sorted(student_counts.items()):
            print(f"   {name}: {count}")
        print()
    else:
        print("❌ No embeddings to merge")

if __name__ == "__main__":
    merge_models()
