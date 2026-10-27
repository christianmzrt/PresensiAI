import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
FACES_DIR = os.path.join(DATA_DIR, "faces")
DB_PATH = os.path.join(DATA_DIR, "presensi.db")

# Ensure required directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FACES_DIR, exist_ok=True)

# Face Recognition Configuration
# Euclidean distance threshold: smaller = stricter, larger = looser. 0.85-0.95 is typical for FaceNet
RECOGNITION_THRESHOLD = 0.90

# Liveness Detection (EAR - Eye Aspect Ratio)
# When eyes are closed, EAR drops below this threshold
EAR_THRESHOLD = 0.22
# Number of consecutive frames eye must be closed to register a blink
EAR_CONSEC_FRAMES = 2
# Number of blinks required to pass liveness
REQUIRED_BLINKS = 2

# Camera / Image specifications
IMAGE_SIZE = (160, 160)
CONFIDENCE_FACE_DETECTION = 0.65
