
import os
import cv2
import sys
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from backend.app.services.face import FaceInfoService
from backend.app.config import settings

def debug_face_detection():
    print("Initializing FaceInfoService...")
    try:
        service = FaceInfoService()
    except Exception as e:
        print(f"Failed to initialize FaceInfoService: {e}")
        return

    uploads_dir = os.path.join(os.path.dirname(__file__), '..', 'uploads')
    if not os.path.exists(uploads_dir):
        print(f"Uploads dir not found: {uploads_dir}")
        return

    print(f"Scanning {uploads_dir}...")
    files = [f for f in os.listdir(uploads_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not files:
        print("No image files found in uploads.")
        return

    # Sort by time to get latest
    files.sort(key=lambda x: os.path.getmtime(os.path.join(uploads_dir, x)), reverse=True)
    
    print(f"Found {len(files)} images. Testing the latest 5...")
    
    for fname in files[:5]:
        path = os.path.join(uploads_dir, fname)
        print(f"\nProcessing: {fname}")
        
        img = cv2.imread(path)
        if img is None:
            print(f"  FAILED to load image (cv2.imread returned None)")
            continue
            
        print(f"  Image shape: {img.shape}")
        
        # Test original
        faces = service.analyze(img)
        if faces:
             print(f"  SUCCESS: Face detected in ORIGINAL! bbox: {faces.bbox}")
             continue
             
        print(f"  FAILURE: No face detected in ORIGINAL.")
        
        # Test rotations
        for angle in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]:
            rotated = cv2.rotate(img, angle)
            faces = service.analyze(rotated)
            if faces:
                print(f"  SUCCESS: Face detected after ROTATION (angle code {angle})! bbox: {faces.bbox}")
                break
        else:
            print(f"  FAILURE: No face detected even after rotations.")

if __name__ == "__main__":
    debug_face_detection()
