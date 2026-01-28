# Class Attendance System - User Guide

## Overview
This system processes videos of class sessions to automatically generate attendance lists using face detection and recognition.

## Workflow

### Step 1: Process Videos with Face Detection
```bash
python test_models.py --save-csv --model 5
```

**What it does:**
- Detects and recognizes faces in classroom videos
- Tracks faces across frames to ensure consistency
- Saves only faces detected in 5+ consecutive frames
- Outputs to `detected_faces.csv`

**Parameters:**
- `--save-csv`: Enable CSV storage of detected faces
- `--model 5`: Use the best recognition model (Merged R100)
- `--csv-path`: Custom CSV path (optional)
- `--consistency-frames`: Number of frames required (default: 5)

### Step 2: Generate Attendance List
```bash
python generate_attendance.py
```

**What it does:**
- Reads `detected_faces.csv`
- Filters out "Unknown" faces
- Aggregates data per student
- Generates attendance summary

**Parameters:**
- `--input`: Input CSV file (default: detected_faces.csv)
- `--output`: Output CSV file (default: attendance.csv)
- `--min-detections`: Minimum detections to mark present (default: 1)
- `--simple-list`: Also create a text file with just names

## Output Files

### attendance.csv
Detailed attendance report with columns:
- **student_name**: Name of the student
- **attendance_status**: Always "PRESENT" (filtered list)
- **total_detections**: Total frame detections across all appearances
- **unique_appearances**: Number of separate tracking instances
- **avg_confidence**: Average recognition confidence (0.0-1.0)
- **first_seen**: Timestamp of first detection
- **last_seen**: Timestamp of last detection

### attendance_list.txt (if --simple-list used)
Simple text file with one student name per line, sorted alphabetically.

## Example Usage

### Basic Attendance Generation
```bash
# 1. Process class video
python test_models.py --save-csv --model 5

# 2. Generate attendance
python generate_attendance.py
```

### Custom Configuration
```bash
# Stricter consistency (10 frames)
python test_models.py --save-csv --consistency-frames 10

# Require minimum 20 total detections
python generate_attendance.py --min-detections 20

# Generate both detailed CSV and simple list
python generate_attendance.py --simple-list
```

### Multiple Videos
```bash
# Process multiple videos to same CSV
python test_models.py --save-csv --csv-path class_2026-01-27.csv

# Generate attendance from specific CSV
python generate_attendance.py --input class_2026-01-27.csv --output attendance_2026-01-27.csv
```

## Tips for Best Results

1. **Video Quality**: Use videos with clear front-facing shots of students
2. **Lighting**: Ensure good lighting conditions for better face detection
3. **Training**: Train your model with clear passport-style photos of all students
4. **Consistency Threshold**: Adjust `--consistency-frames` based on video quality
   - Lower (3-5): More lenient, may include brief appearances
   - Higher (10-15): Stricter, only very consistent detections

5. **Review Unknown Faces**: If many Unknown faces appear:
   - Check if students are in your training database
   - Review video quality and camera angles
   - Consider retraining with more student photos

## Attendance Statistics

The system provides:
- **Total Students Present**: Count of unique recognized students
- **Detection Count**: How many times each student was detected
- **Unique Appearances**: How many separate tracking instances (indicates movement)
- **Average Confidence**: Recognition reliability (higher is better)
- **Time Range**: When student was first and last seen in video

## Troubleshooting

### No students detected (all Unknown)
- **Cause**: Students not in training database
- **Solution**: Run `train_faces.py` or `train_faces_gpu.py` with student photos

### Too many false detections
- **Cause**: Consistency threshold too low
- **Solution**: Increase `--consistency-frames` to 10 or higher

### Missing students who were present
- **Cause**: Poor video quality or recognition confidence too high
- **Solution**: 
  - Lower `--confidence` threshold in test_models.py
  - Improve video quality
  - Retrain model with more varied photos

### Duplicate entries for same student
- **Cause**: Face tracking lost between appearances
- **Solution**: This is normal behavior. Check `unique_appearances` in attendance.csv

## Quick Reference

```bash
# Full pipeline
python test_models.py --save-csv --model 5
python generate_attendance.py --simple-list

# View results
# - detected_faces.csv (raw detections)
# - attendance.csv (attendance report)
# - attendance_list.txt (simple name list)
```

## Sample Output

```
============================================================
 CLASS ATTENDANCE PROCESSOR  
============================================================

📂 Reading from: detected_faces.csv
✅ Processed 288 total records
   - 101 unknown faces (filtered out)
   - 187 recognized faces

💾 Writing attendance to: attendance.csv

============================================================
  ATTENDANCE SUMMARY
============================================================

📊 Total Students Present: 13

Student Names:
   1. ABHINAV C S              (10 detections, 3 appearances, 77.1% avg confidence)
   2. ANESSHA                  (5 detections, 1 appearances, 73.4% avg confidence)
   3. ARDRA SURESH            (5 detections, 1 appearances, 76.7% avg confidence)
   4. ARJUN P NAIR            (5 detections, 1 appearances, 73.6% avg confidence)
   5. DHEERAJ BABU            (305 detections, 23 appearances, 80.8% avg confidence)
   ... [more students]

============================================================

✅ Attendance saved to: attendance.csv
📝 Simple attendance list saved to: attendance_list.txt
   Total students: 13
```

---

**Created**: 2026-01-27  
**Version**: 1.0
