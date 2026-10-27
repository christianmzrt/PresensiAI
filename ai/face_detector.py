import os
import cv2
import dlib
import numpy as np
from config import BASE_DIR

class FaceDetector:
    def __init__(self):
        # Look for cascade file in models directory, then cv2 data
        cascade_local = os.path.join(BASE_DIR, "models", "haarcascade_frontalface_default.xml")
        if os.path.exists(cascade_local):
            self.face_cascade = cv2.CascadeClassifier(cascade_local)
        else:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            
        # Also initialize dlib detector as high-precision fallback
        self.dlib_detector = dlib.get_frontal_face_detector()

    def detect_faces(self, image):
        """
        Detect faces in image (BGR or RGB).
        Returns list of (x, y, w, h) bounding boxes.
        """
        if image is None or image.size == 0:
            return []
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Equalize histogram for better detection across lighting conditions
        gray = cv2.equalizeHist(gray)

        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(80, 80),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        if len(faces) > 0:
            return list(faces)

        # Fallback to dlib detector
        rects = self.dlib_detector(gray, 0)
        return [(r.left(), r.top(), r.width(), r.height()) for r in rects]

    def extract_face(self, image, target_size=(160, 160), margin=15):
        """
        Detects the largest face in image, crops with margin, and resizes to target_size.
        Returns: (cropped_face, (x, y, w, h)) or (None, None)
        """
        faces = self.detect_faces(image)
        if len(faces) == 0:
            return None, None

        # Pick the largest face by area
        largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
        x, y, w, h = largest_face

        # Add margin
        img_h, img_w = image.shape[:2]
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(img_w, x + w + margin)
        y2 = min(img_h, y + h + margin)

        cropped = image[y1:y2, x1:x2]
        if cropped.size == 0:
            return None, None

        resized = cv2.resize(cropped, target_size, interpolation=cv2.INTER_AREA)
        return resized, (x, y, w, h)
