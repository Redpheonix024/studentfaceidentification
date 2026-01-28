#!/usr/bin/env python3
"""
Example: How to use the CSV face storage feature
This script demonstrates running test_models.py with CSV export enabled
"""

print("=" * 60)
print("  FACE DETECTION CSV EXPORT - USAGE EXAMPLES")
print("=" * 60)

print("\n1. BASIC USAGE - Enable CSV export with default settings:")
print("   python test_models.py --save-csv")
print("   - Saves to: detected_faces.csv")
print("   - Consistency: 5 frames required")
print()

print("2. CUSTOM CSV PATH:")
print("   python test_models.py --save-csv --csv-path my_faces.csv")
print()

print("3. ADJUST CONSISTENCY THRESHOLD:")
print("   python test_models.py --save-csv --consistency-frames 10")
print("   - Requires 10 consecutive frames (more strict)")
print()

print("4. COMBINE WITH OTHER OPTIONS:")
print("   python test_models.py --save-csv --model 5 --confidence 0.5")
print("   - Use merged R100 model")
print("   - Higher confidence threshold")
print("   - Save consistent detections to CSV")
print()

print("5. DETECTION-ONLY WITH CSV:")
print("   python test_models.py --save-csv --no-recognition")
print("   - Skip face recognition")
print("   - Save all detected faces (marked as 'Unknown')")
print()

print("=" * 60)
print("  CSV OUTPUT FORMAT")
print("=" * 60)
print("\nThe CSV file will contain the following columns:")
print("  - timestamp: When the face was saved")
print("  - track_id: Unique ID for this face track")
print("  - name: Recognized name (or 'Unknown')")
print("  - confidence: Average detection confidence")
print("  - x, y, width, height: Bounding box coordinates")
print("  - detection_count: Number of frames this face was detected")
print("  - first_seen: Timestamp of first detection")
print("  - last_seen: Timestamp of last detection")
print()

print("=" * 60)
print("  HOW IT WORKS")
print("=" * 60)
print("""
The face tracker uses IoU (Intersection over Union) to track the same
face across frames. A face is considered 'consistent' and saved to CSV
only when it appears in N consecutive frames (default: 5).

This prevents saving:
  ✗ False positives (random detections)
  ✗ Fleeting faces (people who appear briefly)
  ✗ Duplicate entries (same person counted multiple times)

And ensures you only save:
  ✓ Faces that appear consistently in the video
  ✓ One entry per unique face track
  ✓ Reliable, high-quality detections
""")

print("=" * 60)
