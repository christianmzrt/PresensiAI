import sqlite3
import numpy as np
import io
import datetime
from config import DB_PATH

def adapt_array(arr):
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)

# Register sqlite converters for numpy arrays
sqlite3.register_adapter(np.ndarray, adapt_array)
sqlite3.register_converter("array", convert_array)

def get_connection():
    conn = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_connection()
    cursor = conn.cursor()
    
    # Table mahasiswa
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS mahasiswa (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nim TEXT UNIQUE NOT NULL,
            nama TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Table face_embeddings
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS face_embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            mahasiswa_id INTEGER NOT NULL,
            embedding array NOT NULL,
            image_path TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(mahasiswa_id) REFERENCES mahasiswa(id) ON DELETE CASCADE
        )
    """)
    
    # Table kehadiran
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS kehadiran (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            mahasiswa_id INTEGER NOT NULL,
            mata_kuliah TEXT NOT NULL DEFAULT 'General / Enterprise',
            tanggal TEXT NOT NULL,
            waktu TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'Hadir',
            confidence REAL,
            photo_path TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(mahasiswa_id) REFERENCES mahasiswa(id) ON DELETE CASCADE
        )
    """)
    
    conn.commit()
    conn.close()

def register_mahasiswa(nim, nama):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("INSERT INTO mahasiswa (nim, nama) VALUES (?, ?)", (nim.strip(), nama.strip()))
        mahasiswa_id = cursor.lastrowid
        conn.commit()
        return mahasiswa_id, None
    except sqlite3.IntegrityError:
        return None, f"Mahasiswa dengan NIM {nim} sudah terdaftar."
    finally:
        conn.close()

def save_embedding(mahasiswa_id, embedding_arr, image_path=None):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO face_embeddings (mahasiswa_id, embedding, image_path) VALUES (?, ?, ?)",
        (mahasiswa_id, embedding_arr, image_path)
    )
    conn.commit()
    conn.close()

def get_all_embeddings():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT fe.mahasiswa_id, fe.embedding, m.nim, m.nama
        FROM face_embeddings fe
        JOIN mahasiswa m ON fe.mahasiswa_id = m.id
    """)
    rows = cursor.fetchall()
    conn.close()
    
    result = []
    for r in rows:
        result.append({
            "mahasiswa_id": r["mahasiswa_id"],
            "embedding": r["embedding"],
            "nim": r["nim"],
            "nama": r["nama"]
        })
    return result

def get_all_mahasiswa():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT m.id, m.nim, m.nama, m.created_at, COUNT(fe.id) as sample_count
        FROM mahasiswa m
        LEFT JOIN face_embeddings fe ON m.id = fe.mahasiswa_id
        GROUP BY m.id
        ORDER BY m.nama ASC
    """)
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

def delete_mahasiswa(mahasiswa_id):
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("PRAGMA foreign_keys = ON")
    cursor.execute("DELETE FROM mahasiswa WHERE id = ?", (mahasiswa_id,))
    conn.commit()
    conn.close()

def record_attendance(mahasiswa_id, mata_kuliah="General / Enterprise", confidence=0.0, photo_path=None):
    now = datetime.datetime.now()
    tanggal = now.strftime("%Y-%m-%d")
    waktu = now.strftime("%H:%M:%S")
    
    conn = get_connection()
    cursor = conn.cursor()
    
    # Check if already attended today for the same course
    cursor.execute("""
        SELECT id FROM kehadiran 
        WHERE mahasiswa_id = ? AND mata_kuliah = ? AND tanggal = ?
    """, (mahasiswa_id, mata_kuliah, tanggal))
    
    existing = cursor.fetchone()
    if existing:
        conn.close()
        return False, "Mahasiswa sudah melakukan presensi hari ini untuk mata kuliah ini."
    
    cursor.execute("""
        INSERT INTO kehadiran (mahasiswa_id, mata_kuliah, tanggal, waktu, status, confidence, photo_path)
        VALUES (?, ?, ?, ?, 'Hadir', ?, ?)
    """, (mahasiswa_id, mata_kuliah, tanggal, waktu, float(confidence), photo_path))
    
    conn.commit()
    conn.close()
    return True, f"Presensi berhasil dicatat pada {waktu}"

def get_attendance_history(tanggal=None):
    conn = get_connection()
    cursor = conn.cursor()
    if tanggal:
        cursor.execute("""
            SELECT k.id, m.nim, m.nama, k.mata_kuliah, k.tanggal, k.waktu, k.status, k.confidence, k.photo_path
            FROM kehadiran k
            JOIN mahasiswa m ON k.mahasiswa_id = m.id
            WHERE k.tanggal = ?
            ORDER BY k.id DESC
        """, (tanggal,))
    else:
        cursor.execute("""
            SELECT k.id, m.nim, m.nama, k.mata_kuliah, k.tanggal, k.waktu, k.status, k.confidence, k.photo_path
            FROM kehadiran k
            JOIN mahasiswa m ON k.mahasiswa_id = m.id
            ORDER BY k.id DESC
        """)
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]
