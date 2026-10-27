import os
import base64
import cv2
import numpy as np
import datetime
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash

import config
import database
from ai.face_detector import FaceDetector
from ai.face_recognizer import FaceRecognizer
from ai.liveness_detector import LivenessDetector

app = Flask(__name__)
app.secret_key = "presensi-ai-anti-spoofing-face-recognition-key"

# Initialize Database
database.init_db()

# Initialize AI Modules (Singleton in memory)
print("Initializing AI models for Presensi Web...")
face_detector = FaceDetector()
face_recognizer = FaceRecognizer()
liveness_detector = LivenessDetector()

# In-memory session store for tracking real-time liveness state per client session
# key: session_id or client_token -> LivenessDetector instance
active_liveness_sessions = {}

def get_liveness_session(session_id):
    if session_id not in active_liveness_sessions:
        # Create a fresh detector instance with its own blink counters
        active_liveness_sessions[session_id] = LivenessDetector()
    return active_liveness_sessions[session_id]

def decode_base64_image(base64_str):
    """
    Decodes base64 data URL string into an OpenCV BGR numpy array.
    """
    try:
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]
        img_bytes = base64.b64decode(base64_str)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"Error decoding image: {e}")
        return None

# ================= ROUTES =================

@app.route("/")
def index():
    mahasiswa_list = database.get_all_mahasiswa()
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    today_attendance = database.get_attendance_history(today)
    return render_template(
        "index.html", 
        total_mhs=len(mahasiswa_list), 
        today_attendance_count=len(today_attendance),
        today_date=today
    )

@app.route("/register")
def register_page():
    return render_template("register.html")

@app.route("/api/register", methods=["POST"])
def api_register():
    data = request.get_json()
    if not data:
        return jsonify({"success": False, "message": "Data tidak valid"}), 400

    nim = data.get("nim", "").strip()
    nama = data.get("nama", "").strip()
    images = data.get("images", []) # List of base64 images (5-10 samples)

    if not nim or not nama:
        return jsonify({"success": False, "message": "NIM dan Nama wajib diisi"}), 400

    if not images or len(images) < 3:
        return jsonify({"success": False, "message": "Minimal 3 sampel foto wajah diperlukan untuk ekstraksi fitur yang akurat"}), 400

    # Check or create mahasiswa in DB
    mahasiswa_id, err = database.register_mahasiswa(nim, nama)
    if err:
        return jsonify({"success": False, "message": err}), 400

    valid_samples = 0
    saved_paths = []

    for idx, img_b64 in enumerate(images):
        frame = decode_base64_image(img_b64)
        if frame is None:
            continue

        cropped_face, face_box = face_detector.extract_face(frame, target_size=config.IMAGE_SIZE)
        if cropped_face is None:
            continue

        # Extract 512-d embedding using FaceNet CNN
        embedding = face_recognizer.get_embedding(cropped_face)

        # Save sample image to disk
        filename = f"{nim}_{idx}_{int(datetime.datetime.now().timestamp())}.jpg"
        save_path = os.path.join(config.FACES_DIR, filename)
        cv2.imwrite(save_path, cropped_face)
        saved_paths.append(save_path)

        # Save embedding to SQLite
        database.save_embedding(mahasiswa_id, embedding, image_path=save_path)
        valid_samples += 1

    if valid_samples == 0:
        # Rollback mahasiswa creation if no valid faces detected
        database.delete_mahasiswa(mahasiswa_id)
        return jsonify({"success": False, "message": "Tidak ada wajah terdeteksi pada foto yang diambil. Silakan coba lagi."}), 400

    return jsonify({
        "success": True,
        "message": f"Mahasiswa {nama} berhasil didaftarkan dengan {valid_samples} sampel wajah!"
    })

@app.route("/attendance")
def attendance_page():
    return render_template("attendance.html")

@app.route("/api/liveness/reset", methods=["POST"])
def api_liveness_reset():
    data = request.get_json() or {}
    session_id = data.get("session_id", "default")
    detector = get_liveness_session(session_id)
    detector.reset()
    return jsonify({"success": True, "message": "Liveness session reset"})

@app.route("/api/liveness/check", methods=["POST"])
def api_liveness_check():
    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"success": False, "message": "Frame tidak ditemukan"}), 400

    session_id = data.get("session_id", "default")
    detector = get_liveness_session(session_id)

    frame = decode_base64_image(data["image"])
    if frame is None:
        return jsonify({"success": False, "message": "Gagal membaca frame kamera"}), 400

    result = detector.process_frame(frame)
    return jsonify({
        "success": True,
        "result": result
    })

@app.route("/api/recognize", methods=["POST"])
def api_recognize():
    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"success": False, "message": "Frame tidak ditemukan"}), 400

    session_id = data.get("session_id", "default")
    detector = get_liveness_session(session_id)

    # Optional: verify liveness passed in this session
    bypass_liveness = data.get("bypass_liveness", False)
    if not bypass_liveness and not detector.liveness_passed:
        return jsonify({
            "success": False, 
            "message": "Anti-Spoofing: Verifikasi liveness (kedipan mata) belum terpenuhi!"
        }), 403

    frame = decode_base64_image(data["image"])
    if frame is None:
        return jsonify({"success": False, "message": "Gagal membaca frame kamera"}), 400

    cropped_face, face_box = face_detector.extract_face(frame, target_size=config.IMAGE_SIZE)
    if cropped_face is None:
        return jsonify({"success": False, "message": "Wajah tidak terdeteksi untuk pengenalan"}), 400

    # Extract 512-d embedding from query face
    query_embedding = face_recognizer.get_embedding(cropped_face)

    # Fetch all stored embeddings from DB
    db_records = database.get_all_embeddings()
    if not db_records:
        return jsonify({
            "success": False, 
            "message": "Belum ada mahasiswa yang terdaftar dalam sistem. Silakan registrasi terlebih dahulu."
        }), 404

    # Identify with FaceNet CNN embedding comparison
    is_match, matched_student, distance, confidence = face_recognizer.identify(
        query_embedding, db_records, threshold=config.RECOGNITION_THRESHOLD
    )

    if not is_match or matched_student is None:
        return jsonify({
            "success": False,
            "message": f"Wajah tidak dikenali dalam basis data (Jarak terdekat: {distance:.3f} > {config.RECOGNITION_THRESHOLD})"
        }), 404

    # Record attendance in database
    mahasiswa_id = matched_student["mahasiswa_id"]
    mata_kuliah = data.get("mata_kuliah", "General / Enterprise")
    
    ts = int(datetime.datetime.now().timestamp())
    att_filename = f"att_{matched_student['nim']}_{ts}.jpg"
    att_path = os.path.join(config.FACES_DIR, att_filename)
    cv2.imwrite(att_path, cropped_face)

    recorded, rec_msg = database.record_attendance(
        mahasiswa_id=mahasiswa_id,
        mata_kuliah=mata_kuliah,
        confidence=confidence,
        photo_path=att_path
    )

    # Reset sesi banh
    detector.reset()

    return jsonify({
        "success": True,
        "already_recorded": not recorded,
        "message": rec_msg,
        "student": {
            "id": matched_student["mahasiswa_id"],
            "nim": matched_student["nim"],
            "nama": matched_student["nama"]
        },
        "metrics": {
            "euclidean_distance": round(distance, 4),
            "confidence_pct": confidence,
            "threshold": config.RECOGNITION_THRESHOLD
        }
    })

@app.route("/history")
def history_page():
    date_filter = request.args.get("tanggal", "")
    records = database.get_attendance_history(date_filter if date_filter else None)
    return render_template("history.html", records=records, current_filter=date_filter)

@app.route("/students")
def students_page():
    students = database.get_all_mahasiswa()
    return render_template("students.html", students=students)

@app.route("/api/students/<int:student_id>/delete", methods=["POST"])
def api_delete_student(student_id):
    database.delete_mahasiswa(student_id)
    return jsonify({"success": True, "message": "Mahasiswa dan data embedding berhasil dihapus."})

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 PresensiAI - Biometric Smart Attendance System")
    print("FaceNet CNN (512-d) & Real-Time Liveness Anti-Spoofing")
    print("Server running at: http://127.0.0.1:5000")
    print("="*60 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=True)
