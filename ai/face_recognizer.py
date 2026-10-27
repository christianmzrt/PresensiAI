import torch
import numpy as np
import cv2
from facenet_pytorch import InceptionResnetV1

class FaceRecognizer:
    def __init__(self, device=None):
        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        print(f"Loading FaceNet InceptionResnetV1 on device: {self.device}...")
        # Pretrained on VGGFace2 dataset, outputs 512-dimensional embeddings
        self.model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

    def preprocess(self, face_img):
        """
        Takes a face image (BGR or RGB 160x160 numpy array),
        converts to normalized PyTorch tensor (1, 3, 160, 160).
        """
        if face_img.shape[:2] != (160, 160):
            face_img = cv2.resize(face_img, (160, 160))
            
        # Convert BGR to RGB if needed
        if len(face_img.shape) == 3 and face_img.shape[2] == 3:
            rgb_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        else:
            rgb_img = face_img
            
        # Standard fixed image standardization for FaceNet: (img - 127.5) / 128.0
        normalized = (rgb_img.astype(np.float32) - 127.5) / 128.0
        # Transpose to (3, H, W)
        tensor = torch.from_numpy(normalized.transpose((2, 0, 1))).float()
        # Add batch dimension -> (1, 3, 160, 160)
        return tensor.unsqueeze(0).to(self.device)

    def get_embedding(self, face_img):
        """
        Extracts 512-d embedding vector as a 1D numpy array.
        """
        with torch.no_grad():
            tensor = self.preprocess(face_img)
            embedding = self.model(tensor)
            # Embedding is L2-normalized by InceptionResnetV1
            emb_np = embedding.cpu().squeeze(0).numpy()
            # Ensure unit norm
            norm = np.linalg.norm(emb_np)
            if norm > 0:
                emb_np = emb_np / norm
            return emb_np

    @staticmethod
    def euclidean_distance(emb1, emb2):
        return float(np.linalg.norm(emb1 - emb2))

    @staticmethod
    def cosine_similarity(emb1, emb2):
        dot = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(dot / (norm1 * norm2))

    def identify(self, query_embedding, db_records, threshold=0.90):
        """
        Compares query_embedding against db_records.
        db_records: list of dicts with keys: {'mahasiswa_id', 'embedding', 'nim', 'nama'}
        Returns:
            match_found (bool),
            matched_student (dict or None),
            best_distance (float),
            confidence_pct (float)
        """
        if not db_records:
            return False, None, float('inf'), 0.0

        best_distance = float('inf')
        matched_record = None

        # Group distances per mahasiswa to find the best match
        for record in db_records:
            db_emb = record['embedding']
            dist = self.euclidean_distance(query_embedding, db_emb)
            if dist < best_distance:
                best_distance = dist
                matched_record = record

        if best_distance <= threshold and matched_record is not None:
            # Calculate confidence score between 0% and 100%
            # If distance is 0 -> 100%, if distance == threshold -> 50%
            confidence = max(0.0, min(100.0, (1.0 - (best_distance / (threshold * 1.5))) * 100))
            return True, matched_record, best_distance, round(confidence, 2)
        else:
            confidence = max(0.0, min(100.0, (1.0 - (best_distance / 2.0)) * 100))
            return False, None, best_distance, round(confidence, 2)
