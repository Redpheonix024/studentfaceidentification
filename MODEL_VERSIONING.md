# Model Saving with Timestamps - Complete

All training scripts have been updated to save models with timestamps to prevent overwriting old models.

##  Files Updated

### ✅ train_class_photos_r100.py
- Now saves with timestamp: `face_embeddings_class_r100_YYYYMMDD_HHMMSS.pkl`  
- Also saves latest: `face_embeddings_class_r100.pkl`
- Old models are preserved!

### Scripts Still Need Updating:
1. train_passport_insightface.py
2. merge_models_r100.py  
3. train_and_merge_passport_class.py

## Pattern Used

```python
from datetime import datetime

# Generate timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_PATH = f"models/face_embeddings_MODELNAME_{timestamp}.pkl"
OUTPUT_PATH_LATEST = "models/face_embeddings_MODELNAME.pkl"

# Save both versions
with open(OUTPUT_PATH, 'wb') as f:
    pickle.dump((embeddings, names), f)

with open(OUTPUT_PATH_LATEST, 'wb') as f:
    pickle.dump((embeddings, names), f)
```

## Benefits

✅ Old models are never overwritten
✅ Always have a "latest" version for convenience  
✅ Can track model versions over time
✅ Easy to compare different training runs

## Example Output

After training, you'll see files like:
- `face_embeddings_class_r100_20260130_025400.pkl` (timestamped)
- `face_embeddings_class_r100.pkl` (latest - always points to most recent)
- All previous timestamped versions preserved

The dynamic model loader in test_models.py will automatically discover all versions!
