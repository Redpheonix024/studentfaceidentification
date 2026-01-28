#!/usr/bin/env python3
"""
Face Storage Module - CSV Export for Consistent Face Detections
Tracks faces across frames and exports to CSV when detected consistently
"""

import csv
import os
import time
from collections import defaultdict, deque
from datetime import datetime
import threading


class FaceTracker:
    """Tracks faces across frames and stores them when detected consistently"""
    
    def __init__(self, csv_path="detected_faces.csv", consistency_frames=5, 
                 iou_threshold=0.5, timeout_seconds=2.0):
        """
        Initialize the face tracker
        
        Args:
            csv_path: Path to CSV file for storing face detections
            consistency_frames: Number of consecutive frames needed to consider a face consistent
            iou_threshold: IoU threshold for matching faces across frames (0.0-1.0)
            timeout_seconds: Time after which a tracked face is considered gone
        """
        self.csv_path = csv_path
        self.consistency_frames = consistency_frames
        self.iou_threshold = iou_threshold
        self.timeout_seconds = timeout_seconds
        
        # Thread-safe data structures
        self.lock = threading.Lock()
        
        # Track face detections: {track_id: deque of (timestamp, x, y, w, h, name, confidence)}
        self.tracks = {}
        self.next_track_id = 0
        
        # Store which faces have been saved to CSV
        self.saved_faces = set()
        
        # Initialize CSV file
        self._initialize_csv()
    
    def _initialize_csv(self):
        """Create CSV file with headers if it doesn't exist"""
        file_exists = os.path.exists(self.csv_path)
        
        with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    'timestamp', 'track_id', 'name', 'confidence', 
                    'x', 'y', 'width', 'height', 
                    'detection_count', 'first_seen', 'last_seen'
                ])
                print(f"✅ Created CSV file: {self.csv_path}")
    
    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union (IoU) between two bounding boxes"""
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        
        # Calculate intersection area
        x_left = max(x1, x2)
        y_top = max(y1, y2)
        x_right = min(x1 + w1, x2 + w2)
        y_bottom = min(y1 + h1, y2 + h2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union area
        box1_area = w1 * h1
        box2_area = w2 * h2
        union_area = box1_area + box2_area - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0.0
    
    def _find_matching_track(self, box, current_time):
        """Find existing track that matches the given bounding box"""
        best_match_id = None
        best_iou = 0.0
        
        for track_id, detections in self.tracks.items():
            if not detections:
                continue
            
            # Get the most recent detection
            last_time, last_box = detections[-1][0], detections[-1][1:5]
            
            # Check if track is still active (not timed out)
            if current_time - last_time > self.timeout_seconds:
                continue
            
            # Calculate IoU
            iou = self._calculate_iou(box, last_box)
            
            if iou > best_iou and iou >= self.iou_threshold:
                best_iou = iou
                best_match_id = track_id
        
        return best_match_id
    
    def _cleanup_old_tracks(self, current_time):
        """Remove tracks that have timed out"""
        tracks_to_remove = []
        
        for track_id, detections in self.tracks.items():
            if detections and current_time - detections[-1][0] > self.timeout_seconds * 2:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
    
    def _save_to_csv(self, track_id, detections):
        """Save a consistent face detection to CSV"""
        if track_id in self.saved_faces:
            return  # Already saved
        
        # Calculate average position and confidence
        avg_x = sum(d[1] for d in detections) / len(detections)
        avg_y = sum(d[2] for d in detections) / len(detections)
        avg_w = sum(d[3] for d in detections) / len(detections)
        avg_h = sum(d[4] for d in detections) / len(detections)
        avg_conf = sum(d[6] for d in detections) / len(detections)
        
        # Get most common name (or last name if all different)
        names = [d[5] for d in detections]
        name = max(set(names), key=names.count)
        
        # Timestamps
        first_seen = detections[0][0]
        last_seen = detections[-1][0]
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Write to CSV
        with open(self.csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                track_id,
                name,
                f"{avg_conf:.3f}",
                f"{int(avg_x)}",
                f"{int(avg_y)}",
                f"{int(avg_w)}",
                f"{int(avg_h)}",
                len(detections),
                f"{first_seen:.2f}",
                f"{last_seen:.2f}"
            ])
        
        self.saved_faces.add(track_id)
        print(f"💾 Saved to CSV: Track {track_id} - {name} (detected in {len(detections)} frames)")
    
    def update(self, faces, current_time=None):
        """
        Update tracker with new face detections from current frame
        
        Args:
            faces: List of face tuples (x, y, w, h, name, confidence)
            current_time: Current timestamp (uses time.time() if not provided)
        """
        if current_time is None:
            current_time = time.time()
        
        with self.lock:
            # Cleanup old tracks
            self._cleanup_old_tracks(current_time)
            
            # Process each detected face
            for face in faces:
                x, y, w, h, name, conf = face
                box = (x, y, w, h)
                
                # Find matching track or create new one
                track_id = self._find_matching_track(box, current_time)
                
                if track_id is None:
                    # Create new track
                    track_id = self.next_track_id
                    self.next_track_id += 1
                    self.tracks[track_id] = deque(maxlen=self.consistency_frames * 2)
                
                # Add detection to track
                detection = (current_time, x, y, w, h, name, conf)
                self.tracks[track_id].append(detection)
                
                # Check if track has enough consistent detections
                if len(self.tracks[track_id]) >= self.consistency_frames:
                    self._save_to_csv(track_id, list(self.tracks[track_id]))
    
    def get_stats(self):
        """Get statistics about tracked faces"""
        with self.lock:
            return {
                'active_tracks': len(self.tracks),
                'saved_faces': len(self.saved_faces),
                'total_track_ids': self.next_track_id
            }
    
    def reset(self):
        """Reset all tracking data (does not delete CSV file)"""
        with self.lock:
            self.tracks.clear()
            self.saved_faces.clear()
            self.next_track_id = 0
            print("🔄 Face tracker reset")


# Example usage function
def example_usage():
    """Example of how to use the FaceTracker"""
    
    # Initialize tracker
    tracker = FaceTracker(
        csv_path="detected_faces.csv",
        consistency_frames=5,  # Need 5 consecutive frames
        iou_threshold=0.5,      # 50% overlap required
        timeout_seconds=2.0     # 2 seconds timeout
    )
    
    # Simulate face detections from video frames
    # In real usage, this would come from your face detection model
    
    frame_count = 0
    
    # Simulate some frames with detectable faces
    for i in range(20):
        frame_count += 1
        current_time = time.time()
        
        # Simulate detected faces: (x, y, w, h, name, confidence)
        faces = []
        
        # Person 1 appears in frames 0-10 (should be saved after 5 frames)
        if i < 10:
            faces.append((100, 100, 80, 80, "John Doe", 0.95))
        
        # Person 2 appears in frames 5-15 (should be saved after frame 10)
        if 5 <= i < 15:
            faces.append((300, 150, 75, 75, "Jane Smith", 0.92))
        
        # Person 3 appears only in 2 frames (should NOT be saved - not consistent)
        if i in [8, 9]:
            faces.append((500, 200, 70, 70, "Unknown", 0.75))
        
        # Update tracker
        tracker.update(faces, current_time)
        
        # Print stats every 5 frames
        if frame_count % 5 == 0:
            stats = tracker.get_stats()
            print(f"Frame {frame_count}: {stats}")
        
        # Small delay to simulate video playback
        time.sleep(0.05)
    
    print("\n✅ Example completed. Check 'detected_faces.csv' for results.")


if __name__ == "__main__":
    example_usage()
