#!/usr/bin/env python3
"""
Fix Name Variations in Face Recognition Model
Standardizes duplicate student names to their canonical form
"""

import pickle
from collections import Counter

# =========================
# CONFIG
# =========================
INPUT_MODEL = "models/face_embeddings_merged_r100.pkl"
OUTPUT_MODEL = "models/face_embeddings_merged_r100_fixed.pkl"

# Name mapping: wrong_name -> correct_name
NAME_STANDARDIZATION = {
    # Passport variations -> Class names (using ALL CAPS as canonical)
    "Gouripriya": "GOURIPRIYA",
    "Hamdha Mohammed": "HAMDA MUHAMMED",
    "Harikrishnan Nair B S": "HARIKSRISHNAN P S",
    "Harsha P": "HARSHA P",  # Already correct in class, fix passport
    "Hasna Dilshad": "HASNATH",  # Map to closest match
    "Krishna S": "KRISHNA S",
    "Pranav T S": "PRANAAV T S",
}

def fix_names():
    print("\n" + "="*60)
    print("  FIXING NAME VARIATIONS")
    print("="*60 + "\n")
    
    # Load model
    print(f"📁 Loading: {INPUT_MODEL}")
    with open(INPUT_MODEL, 'rb') as f:
        embeddings, names = pickle.load(f)
    
    print(f"   ✅ Loaded {len(embeddings)} embeddings")
    print(f"   📊 Original unique names: {len(set(names))}\n")
    
    # Show original name counts
    print("📊 Original name distribution:")
    original_counts = Counter(names)
    for name, count in sorted(original_counts.items()):
        print(f"   {name}: {count}")
    
    # Standardize names
    fixed_names = []
    changes_made = 0
    
    for name in names:
        if name in NAME_STANDARDIZATION:
            fixed_name = NAME_STANDARDIZATION[name]
            fixed_names.append(fixed_name)
            changes_made += 1
        else:
            fixed_names.append(name)
    
    print(f"\n✏️  Fixed {changes_made} name variations\n")
    
    # Show fixed name counts
    print("📊 Fixed name distribution:")
    fixed_counts = Counter(fixed_names)
    for name, count in sorted(fixed_counts.items()):
        change_indicator = ""
        if name in NAME_STANDARDIZATION.values():
            # Check if this name had variations merged into it
            old_count = original_counts.get(name, 0)
            if count > old_count:
                change_indicator = f" (+{count - old_count} merged)"
        print(f"   {name}: {count}{change_indicator}")
    
    # Save fixed model
    with open(OUTPUT_MODEL, 'wb') as f:
        pickle.dump((embeddings, fixed_names), f)
    
    print("\n" + "="*60)
    print("  COMPLETE")
    print("="*60)
    print(f"✅ Total embeddings: {len(embeddings)}")
    print(f"✅ Unique students (before): {len(set(names))}")
    print(f"✅ Unique students (after): {len(set(fixed_names))}")
    print(f"✅ Names fixed: {changes_made}")
    print(f"📁 Saved to: {OUTPUT_MODEL}")
    print("="*60 + "\n")
    
    # Show what changed
    print("🔄 Changes made:")
    for old_name, new_name in NAME_STANDARDIZATION.items():
        if old_name in names:
            count = names.count(old_name)
            print(f"   '{old_name}' → '{new_name}' ({count} embeddings)")
    
    print("\n💡 To use the fixed model:")
    print(f"   1. Backup original: mv {INPUT_MODEL} {INPUT_MODEL}.backup")
    print(f"   2. Use fixed: mv {OUTPUT_MODEL} {INPUT_MODEL}")
    print(f"   3. Test: python test_models.py --model 5\n")

if __name__ == "__main__":
    fix_names()
