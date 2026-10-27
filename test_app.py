import unittest
import numpy as np
import cv2
import os
import io

import config
import database
from ai.face_detector import FaceDetector
from ai.face_recognizer import FaceRecognizer
from ai.liveness_detector import eye_aspect_ratio, LivenessDetector
from app import app

class TestPresensiAI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n--- Menjalankan Pengujian Sistem Presensi AI ---")
        database.init_db()

    def test_1_database_operations(self):
        print("Testing Database CRUD...")
        # 1. Register
        mhs_id, err = database.register_mahasiswa("TEST001", "Mahasiswa Uji Coba")
        self.assertIsNone(err)
        self.assertIsNotNone(mhs_id)

        # 2. Save embedding
        dummy_emb = np.random.randn(512).astype(np.float32)
        dummy_emb = dummy_emb / np.linalg.norm(dummy_emb)
        database.save_embedding(mhs_id, dummy_emb)

        # 3. Retrieve embeddings
        records = database.get_all_embeddings()
        self.assertTrue(len(records) > 0)
        found = any(r["nim"] == "TEST001" for r in records)
        self.assertTrue(found)

        # 4. Record attendance
        success, msg = database.record_attendance(mhs_id, "Kecerdasan Tiruan", confidence=95.0)
        self.assertTrue(success)

        # 5. History check
        history = database.get_attendance_history()
        self.assertTrue(len(history) > 0)

        # 6. Cleanup test student
        database.delete_mahasiswa(mhs_id)
        records_after = database.get_all_embeddings()
        self.assertFalse(any(r["nim"] == "TEST001" for r in records_after))
        print("[OK] Database CRUD Passed!")

    def test_2_face_detector_and_recognizer(self):
        print("Testing AI Modules (FaceDetector & FaceRecognizer)...")
        fr = FaceRecognizer()
        
        # Test 512-d embedding extraction
        dummy_face = np.full((160, 160, 3), 120, dtype=np.uint8)
        emb = fr.get_embedding(dummy_face)
        self.assertEqual(emb.shape, (512,))
        # Embedding should be normalized
        self.assertAlmostEqual(np.linalg.norm(emb), 1.0, places=4)

        # Test identify matching
        db_records = [
            {"mahasiswa_id": 99, "embedding": emb, "nim": "TEST_MATCH", "nama": "Match Person"}
        ]
        is_match, matched, dist, conf = fr.identify(emb, db_records, threshold=0.90)
        self.assertTrue(is_match)
        self.assertEqual(matched["nim"], "TEST_MATCH")
        self.assertAlmostEqual(dist, 0.0, places=4)
        print("[OK] AI FaceNet Recognizer Passed!")

    def test_3_liveness_ear_calculation(self):
        print("Testing Liveness EAR Formula...")
        # Synthetic open eye landmark points (6 points)
        # p1=(0,0), p2=(2,2), p3=(4,2), p4=(6,0), p5=(4,-2), p6=(2,-2)
        open_eye = np.array([
            [0.0, 0.0],
            [2.0, 2.0],
            [4.0, 2.0],
            [6.0, 0.0],
            [4.0, -2.0],
            [2.0, -2.0]
        ])
        ear_open = eye_aspect_ratio(open_eye)
        # Vertical = 4+4=8, Horizontal = 6 -> EAR = 8 / (2*6) = 0.666
        self.assertGreater(ear_open, 0.25)

        # Closed eye points (vertical distance near 0)
        closed_eye = np.array([
            [0.0, 0.0],
            [2.0, 0.1],
            [4.0, 0.1],
            [6.0, 0.0],
            [4.0, -0.1],
            [2.0, -0.1]
        ])
        ear_closed = eye_aspect_ratio(closed_eye)
        self.assertLess(ear_closed, 0.10)
        print("[OK] Liveness EAR Calculation Passed!")

    def test_4_flask_routes(self):
        print("Testing Flask Web Routes...")
        client = app.test_client()

        # Check Home
        res = client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn(b"PresensiAI", res.data)

        # Check Register Page
        res = client.get("/register")
        self.assertEqual(res.status_code, 200)

        # Check Attendance Page
        res = client.get("/attendance")
        self.assertEqual(res.status_code, 200)

        # Check History Page
        res = client.get("/history")
        self.assertEqual(res.status_code, 200)

        # Check Students Page
        res = client.get("/students")
        self.assertEqual(res.status_code, 200)

        # Test Liveness Reset API
        res = client.post("/api/liveness/reset", json={"session_id": "test_session"})
        self.assertEqual(res.status_code, 200)
        print("[OK] Flask Web Routes Passed!")

    def test_5_search_algorithms(self):
        print("Testing Classical Search Algorithms (BFS, DFS, Greedy, A*)...")
        from ai.search_algorithms import FaceGraph, search_bfs, search_dfs, search_greedy, search_astar, compare_all_search
        
        # Synthetic database of 4 dummy student embeddings
        np.random.seed(42)
        records = []
        for i in range(4):
            emb = np.random.randn(512).astype(np.float32)
            emb = emb / np.linalg.norm(emb)
            records.append({
                "mahasiswa_id": i + 1,
                "nim": f"220856100{i+1}",
                "nama": f"Mahasiswa {i+1}",
                "embedding": emb
            })

        # Query matching student 3
        target_emb = records[2]["embedding"] + np.random.normal(0, 0.01, size=512).astype(np.float32)
        target_emb = target_emb / np.linalg.norm(target_emb)

        graph = FaceGraph(records)
        start_node = list(graph.nodes.keys())[0]

        res_bfs = search_bfs(graph, start_node, target_emb, threshold=0.90)
        self.assertTrue(res_bfs["found"])

        res_dfs = search_dfs(graph, start_node, target_emb, threshold=0.90)
        self.assertTrue(res_dfs["found"])

        res_greedy = search_greedy(graph, start_node, target_emb, threshold=0.90)
        self.assertTrue(res_greedy["found"])

        res_astar = search_astar(graph, start_node, target_emb, threshold=0.90)
        self.assertTrue(res_astar["found"])

        comp = compare_all_search(records, target_emb, threshold=0.90)
        self.assertEqual(len(comp["results"]), 4)
        print("[OK] Classical AI Search Algorithms (BFS, DFS, Greedy, A*) Passed!")

if __name__ == "__main__":
    unittest.main()
