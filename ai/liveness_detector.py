import os
import cv2
import dlib
import numpy as np
from config import EAR_THRESHOLD, EAR_CONSEC_FRAMES, REQUIRED_BLINKS, BASE_DIR

# Landmark indices for dlib 68-point model (0-indexed)
# Eye points
RIGHT_EYE_IDXS = list(range(36, 42))  # [36, 37, 38, 39, 40, 41]
LEFT_EYE_IDXS = list(range(42, 48))   # [42, 43, 44, 45, 46, 47]

def euclidean_dist(pt1, pt2):
    return float(np.linalg.norm(pt1 - pt2))

def eye_aspect_ratio(eye_pts):
    """
    Compute the Eye Aspect Ratio (EAR) from 6 landmark points.
    eye_pts: numpy array of shape (6, 2)
    """
    # Vertical distances
    A = euclidean_dist(eye_pts[1], eye_pts[5])
    B = euclidean_dist(eye_pts[2], eye_pts[4])
    # Horizontal distance
    C = euclidean_dist(eye_pts[0], eye_pts[3])

    if C == 0:
        return 0.0

    ear = (A + B) / (2.0 * C)
    return float(ear)

class LivenessDetector:
    def __init__(self, predictor_path=None):
        if predictor_path is None:
            predictor_path = os.path.join(BASE_DIR, "models", "shape_predictor_68_face_landmarks.dat")
        
        self.predictor_path = predictor_path
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = None
        
        if os.path.exists(self.predictor_path):
            self.predictor = dlib.shape_predictor(self.predictor_path)
        else:
            print(f"Warning: Predictor model not found at {self.predictor_path}.")

        # Session state for blink tracking
        self.reset()

    def reset(self):
        self.blink_counter = 0
        self.consecutive_closed_frames = 0
        self.liveness_passed = False
        self.last_ear = 0.0

    def is_ready(self):
        return self.predictor is not None

    def process_frame(self, frame_bgr):
        """
        Process a single video frame.
        Returns a dict:
        {
            'face_detected': bool,
            'ear': float,
            'is_blinking': bool,
            'blink_count': int,
            'blinks_needed': int,
            'liveness_passed': bool,
            'message': str,
            'face_box': (x, y, w, h) or None
        }
        """
        if self.predictor is None:
            return {
                'face_detected': False,
                'ear': 0.0,
                'is_blinking': False,
                'blink_count': self.blink_counter,
                'blinks_needed': REQUIRED_BLINKS,
                'liveness_passed': False,
                'message': 'Model facial landmarks belum siap',
                'face_box': None
            }

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        rects = self.detector(gray, 0)

        if len(rects) == 0:
            return {
                'face_detected': False,
                'ear': 0.0,
                'is_blinking': False,
                'blink_count': self.blink_counter,
                'blinks_needed': REQUIRED_BLINKS,
                'liveness_passed': self.liveness_passed,
                'message': 'Wajah tidak terdeteksi. Posisikan wajah Anda di depan kamera.',
                'face_box': None
            }

        # Take largest detected face
        rect = max(rects, key=lambda r: r.width() * r.height())
        face_box = (rect.left(), rect.top(), rect.width(), rect.height())

        shape = self.predictor(gray, rect)
        landmarks = np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)])

        left_eye = landmarks[LEFT_EYE_IDXS]
        right_eye = landmarks[RIGHT_EYE_IDXS]

        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0
        self.last_ear = ear

        is_closed = ear < EAR_THRESHOLD

        if is_closed:
            self.consecutive_closed_frames += 1
        else:
            # If eyes were closed for enough frames, register a blink!
            if self.consecutive_closed_frames >= EAR_CONSEC_FRAMES:
                self.blink_counter += 1
                if self.blink_counter >= REQUIRED_BLINKS:
                    self.liveness_passed = True
            self.consecutive_closed_frames = 0

        if self.liveness_passed:
            msg = f"Liveness Terverifikasi! ({self.blink_counter}/{REQUIRED_BLINKS} kedipan)"
        else:
            msg = f"Silakan kedipkan mata ({self.blink_counter}/{REQUIRED_BLINKS})"

        return {
            'face_detected': True,
            'ear': round(ear, 3),
            'is_blinking': is_closed,
            'blink_count': self.blink_counter,
            'blinks_needed': REQUIRED_BLINKS,
            'liveness_passed': self.liveness_passed,
            'message': msg,
            'face_box': face_box
        }
