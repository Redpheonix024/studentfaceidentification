import cv2
import os

VIDEO_DIR = "videos"

print("=== Video Playback Test ===")

# Test 1: Check if videos exist
files = [f for f in os.listdir(VIDEO_DIR) 
         if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv'))]
print(f"\n1. Found {len(files)} video files: {files}")

if not files:
    print("ERROR: No videos found!")
    exit(1)

# Test 2: Try to open first video
video_path = os.path.join(VIDEO_DIR, files[0])
print(f"\n2. Testing video: {video_path}")
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("ERROR: Cannot open video file!")
    exit(1)

print("✅ Video opened successfully")

# Test 3: Read first frame
ret, frame = cap.read()
if not ret:
    print("ERROR: Cannot read frame!")
    exit(1)

print(f"✅ Frame read successfully: {frame.shape}")

# Test 4: Create window and display
print("\n3. Testing window display...")
print("   A window should appear. Press ANY KEY to continue.")

cv2.namedWindow("Test Window", cv2.WINDOW_NORMAL)
cv2.imshow("Test Window", frame)

print("\n   Waiting for keypress (timeout 10 seconds)...")
key = cv2.waitKey(10000)  # 10 second timeout

if key == -1:
    print("   ⚠️  TIMEOUT: No key pressed (window may not be visible)")
else:
    print(f"   ✅ Key pressed: {chr(key) if key < 256 else key}")

cv2.destroyAllWindows()
cap.release()

print("\n=== Test Complete ===")
print("If you saw a window with the video frame, everything is working!")
print("If not, there's a display/window issue.")
