#!/usr/bin/env python3
"""
Class Attendance Processor
Generates student attendance list from face detection CSV data
Filters out unknown faces and provides attendance summary
"""

import csv
import argparse
from collections import defaultdict
from datetime import datetime


def process_attendance(input_csv, output_csv="attendance.csv", min_detections=1):
    """
    Process detected faces CSV and generate attendance list
    
    Args:
        input_csv: Path to detected_faces.csv
        output_csv: Path to output attendance CSV
        min_detections: Minimum number of detections required to mark present
    """
    
    # Store student attendance data
    student_data = defaultdict(lambda: {
        'total_detections': 0,
        'total_tracks': 0,
        'avg_confidence': 0.0,
        'confidences': [],
        'first_seen': None,
        'last_seen': None
    })
    
    print(f"\n{'='*60}")
    print("  CLASS ATTENDANCE PROCESSOR")
    print(f"{'='*60}\n")
    print(f"📂 Reading from: {input_csv}")
    
    # Read and process the CSV file
    total_records = 0
    unknown_count = 0
    
    try:
        with open(input_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                total_records += 1
                name = row['name'].strip()
                
                # Skip unknown faces
                if name.upper() == 'UNKNOWN':
                    unknown_count += 1
                    continue
                
                # Extract data
                confidence = float(row['confidence'])
                detection_count = int(row['detection_count'])
                timestamp = row['timestamp']
                
                # Update student data
                student_data[name]['total_detections'] += detection_count
                student_data[name]['total_tracks'] += 1
                student_data[name]['confidences'].append(confidence)
                
                # Track first and last seen
                if student_data[name]['first_seen'] is None:
                    student_data[name]['first_seen'] = timestamp
                student_data[name]['last_seen'] = timestamp
        
        print(f"✅ Processed {total_records} total records")
        print(f"   - {unknown_count} unknown faces (filtered out)")
        print(f"   - {total_records - unknown_count} recognized faces")
        
    except FileNotFoundError:
        print(f"❌ Error: File not found: {input_csv}")
        return
    except Exception as e:
        print(f"❌ Error reading CSV: {e}")
        return
    
    # Calculate average confidences
    for name, data in student_data.items():
        if data['confidences']:
            data['avg_confidence'] = sum(data['confidences']) / len(data['confidences'])
    
    # Filter students based on minimum detections
    attending_students = {
        name: data for name, data in student_data.items()
        if data['total_detections'] >= min_detections
    }
    
    if not attending_students:
        print("\n⚠️  No students found in the video (all faces were unknown)")
        return
    
    # Sort students by name
    sorted_students = sorted(attending_students.items())
    
    # Write attendance CSV
    print(f"\n💾 Writing attendance to: {output_csv}")
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        fieldnames = [
            'student_name',
            'attendance_status',
            'total_detections',
            'unique_appearances',
            'avg_confidence',
            'first_seen',
            'last_seen'
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for name, data in sorted_students:
            writer.writerow({
                'student_name': name,
                'attendance_status': 'PRESENT',
                'total_detections': data['total_detections'],
                'unique_appearances': data['total_tracks'],
                'avg_confidence': f"{data['avg_confidence']:.3f}",
                'first_seen': data['first_seen'],
                'last_seen': data['last_seen']
            })
    
    # Print summary
    print(f"\n{'='*60}")
    print("  ATTENDANCE SUMMARY")
    print(f"{'='*60}\n")
    print(f"📊 Total Students Present: {len(sorted_students)}")
    print(f"\nStudent Names:")
    for i, (name, data) in enumerate(sorted_students, 1):
        detections = data['total_detections']
        tracks = data['total_tracks']
        conf = data['avg_confidence']
        print(f"  {i:2d}. {name:25s} ({detections} detections, {tracks} appearances, {conf:.1%} avg confidence)")
    
    print(f"\n{'='*60}\n")
    print(f"✅ Attendance saved to: {output_csv}")
    
    return sorted_students


def generate_simple_list(input_csv, output_txt="attendance_list.txt"):
    """
    Generate a simple text file with just student names (one per line)
    
    Args:
        input_csv: Path to detected_faces.csv
        output_txt: Path to output text file
    """
    
    # Collect unique student names (excluding Unknown)
    student_names = set()
    
    try:
        with open(input_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                name = row['name'].strip()
                if name.upper() != 'UNKNOWN':
                    student_names.add(name)
        
        # Sort and write to file
        sorted_names = sorted(student_names)
        
        with open(output_txt, 'w', encoding='utf-8') as f:
            for name in sorted_names:
                f.write(f"{name}\n")
        
        print(f"📝 Simple attendance list saved to: {output_txt}")
        print(f"   Total students: {len(sorted_names)}")
        
    except Exception as e:
        print(f"❌ Error generating simple list: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Process face detection CSV and generate attendance list'
    )
    parser.add_argument('--input', type=str, default='detected_faces.csv',
                        help='Input CSV file with detected faces (default: detected_faces.csv)')
    parser.add_argument('--output', type=str, default='attendance.csv',
                        help='Output CSV file for attendance (default: attendance.csv)')
    parser.add_argument('--min-detections', type=int, default=1,
                        help='Minimum detections required to mark present (default: 1)')
    parser.add_argument('--simple-list', action='store_true',
                        help='Also generate a simple text file with student names')
    
    args = parser.parse_args()
    
    # Process attendance
    result = process_attendance(
        input_csv=args.input,
        output_csv=args.output,
        min_detections=args.min_detections
    )
    
    # Generate simple list if requested
    if args.simple_list and result:
        generate_simple_list(
            input_csv=args.input,
            output_txt=args.output.replace('.csv', '_list.txt')
        )


if __name__ == "__main__":
    main()
