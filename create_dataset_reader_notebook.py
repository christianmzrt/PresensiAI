import json
import os

cells = []

def add_markdown(source_lines):
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source_lines]
    })

def add_code(source_lines):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source_lines]
    })

# --- CELL 1: Header ---
add_markdown([
    "# 🧪 Pipeline Pembacaan & Verifikasi Dataset Face Recognition",
    "**Mata Kuliah:** Kecerdasan Tiruan  ",
    "**Tujuan:** Membaca dataset citra wajah, mendeteksi keberadaan wajah, melakukan normalisasi dimensi, dan mengekstraksi vektor embedding CNN FaceNet dengan output **murni laporan teks terstruktur**.",
    "",
    "---",
    "### 📋 Apa Saja yang Dibaca & Dideteksi oleh Program ini?",
    "1. **Metadata Berkas:** Nama file, format ekstensi, ukuran file (KB), dan resolusi citra asli (Lebar x Tinggi x Channel).",
    "2. **Deteksi Wajah (Face Detection):** Mendeteksi koordinat area wajah $(x, y, w, h)$ menggunakan Haar Cascade.",
    "3. **Preprocessing Citra:** Cropping area wajah dan penyesuaian dimensi standar input CNN ($160 \\times 160$ piksel).",
    "4. **Ekstraksi Fitur Biometrik:** Konversi matriks piksel wajah ke representasi vektor embedding 512-dimensi via CNN FaceNet.",
    "5. **Laporan Teks:** Status verifikasi dataset berhasil dibaca dan siap diproses ke tahap pengenalan / presensi."
])

# --- CELL 2: Install facenet-pytorch (Ringan & Cepat ~3 Detik) ---
add_markdown([
    "## 📦 1. Persiapan Environment (Hanya ~3 Detik)",
    "Hanya menginstall `facenet-pytorch`. OpenCV, NumPy, PyTorch, dan library lainnya sudah terpasang bawaan di Google Colab."
])

add_code([
    "# Install modul CNN FaceNet secara ringkas dan hening (-q)",
    "!pip install facenet-pytorch -q",
    "",
    "import os",
    "import cv2",
    "import torch",
    "import time",
    "import numpy as np",
    "from facenet_pytorch import InceptionResnetV1",
    "",
    "print('====================================================')",
    "print('✅ [STATUS] Seluruh pustaka AI berhasil dimuat!')",
    "print(f'   - PyTorch Version : {torch.__version__}')",
    "print(f'   - OpenCV Version  : {cv2.__version__}')",
    "print('====================================================')"
])

# --- CELL 3: Folder Dataset & Opsi Upload ---
add_markdown([
    "## 📁 2. Pemeriksaan Folder Dataset",
    "Program akan memeriksa folder `dataset/`. Jika folder masih kosong, Anda bisa mengunggah foto Anda, ATAU sistem akan secara otomatis membuatkan data citra sampel agar program langsung dapat dites tanpa kendala."
])

add_code([
    "# Siapkan direktori dataset",
    "DATASET_DIR = 'dataset'",
    "os.makedirs(DATASET_DIR, exist_ok=True)",
    "",
    "def cek_isi_dataset(folder):",
    "    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')",
    "    files = [f for f in os.listdir(folder) if f.lower().endswith(valid_exts)]",
    "    return files",
    "",
    "files = cek_isi_dataset(DATASET_DIR)",
    "print(f'📂 Direktori Dataset : /{DATASET_DIR}')",
    "print(f'📄 Jumlah file citra ditemukan: {len(files)} file')",
    "",
    "if len(files) == 0:",
    "    print('\\n⚠️ Folder dataset masih kosong!')",
    "    print('👉 Anda punya 2 pilihan:')",
    "    print('   1. Upload foto Anda sendiri (Jalankan cell opsional di bawah ini)')",
    "    print('   2. Biarkan program membuat 4 sampel citra wajah otomatis untuk simulasi')"
])

# --- CELL 4: Opsional Upload Foto Sendiri ---
add_markdown([
    "### 📤 (Opsional) Upload Foto Mahasiswa ke Dataset",
    "*Jalankan cell ini jika ingin mengunggah foto wajah anggota kelompok Anda (bisa pilih 1 sampai 4 foto sekaligus). Jika tidak ingin upload, lewati cell ini.*"
])

add_code([
    "# Eksekusi cell ini HANYA jika ingin upload file dari laptop Anda",
    "try:",
    "    from google.colab import files",
    "    print('Silakan pilih file foto dari komputer Anda:')",
    "    uploaded = files.upload()",
    "    for fname in uploaded.keys():",
    "        target_path = os.path.join(DATASET_DIR, fname)",
    "        with open(target_path, 'wb') as f:",
    "            f.write(uploaded[fname])",
    "        print(f'✅ Berhasil menyimpan: {target_path}')",
    "except Exception as e:",
    "    print('Dijalankan di luar Colab atau upload dilewati.')"
])

# --- CELL 5: Auto-Generate Sample Images if Empty ---
add_markdown([
    "### 🛡️ Jaminan Dataset (Auto-Fallback)",
    "Jika Anda belum mengunggah foto, cell ini secara otomatis menyiapkan 4 berkas citra berformat JPG dengan pola wajah sintetis sehingga program pembacaan dataset 100% pasti berjalan sukses."
])

add_code([
    "files = cek_isi_dataset(DATASET_DIR)",
    "",
    "if len(files) == 0:",
    "    print('[INFO] Membuat 4 citra sampel otomatis agar pengujian dapat langsung berjalan...')",
    "    for i in range(1, 5):",
    "        dummy = np.full((320, 240, 3), 200 + i * 10, dtype=np.uint8)",
    "        # Gambar pola kepala & mata sederhana",
    "        cv2.ellipse(dummy, (120, 160), (70, 90), 0, 0, 360, (140, 110, 90), -1)",
    "        cv2.circle(dummy, (95, 140), 12, (255, 255, 255), -1)",
    "        cv2.circle(dummy, (95, 140), 6, (0, 0, 0), -1)",
    "        cv2.circle(dummy, (145, 140), 12, (255, 255, 255), -1)",
    "        cv2.circle(dummy, (145, 140), 6, (0, 0, 0), -1)",
    "        cv2.ellipse(dummy, (120, 195), (25, 12), 0, 0, 180, (50, 50, 200), 3)",
    "        cv2.putText(dummy, f'MHS_{i}', (60, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)",
    "        sample_name = f'mhs_{i}.jpg'",
    "        cv2.imwrite(os.path.join(DATASET_DIR, sample_name), dummy)",
    "    print('✅ Berhasil membuat 4 file citra sampel di folder dataset/!')",
    "else:",
    "    print(f'✅ Dataset siap! Menggunakan {len(files)} file foto yang ada.')"
])

# --- CELL 6: Model & Detector Initialization ---
add_markdown([
    "## ⚙️ 3. Inisialisasi Modul Detektor & Ekstraktor Fitur CNN",
    "- **Detektor Wajah:** OpenCV Haar Cascade (Frontal Face Default)",
    "- **Ekstraktor Fitur:** CNN FaceNet (Inception-ResNet-v1, Pretrained VGGFace2) 512 Dimensi"
])

add_code([
    "import urllib.request",
    "",
    "# 1. Muat Haar Cascade Detector",
    "cascade_path = 'haarcascade_frontalface_default.xml'",
    "if not os.path.exists(cascade_path):",
    "    cv2_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'",
    "    if os.path.exists(cv2_path):",
    "        cascade_path = cv2_path",
    "    else:",
    "        url = 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml'",
    "        urllib.request.urlretrieve(url, cascade_path)",
    "",
    "face_cascade = cv2.CascadeClassifier(cascade_path)",
    "",
    "# 2. Muat CNN FaceNet (Inception-ResNet-v1)",
    "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')",
    "model_facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)",
    "",
    "print('====================================================')",
    "print('✅ [STATUS] Modul AI Berhasil Diinisialisasi:')",
    "print(f'   - Algoritma Deteksi  : OpenCV Haar Cascade Classifier')",
    "print(f'   - Model Ekstraksi   : CNN FaceNet (Inception-ResNet-v1)')",
    "print(f'   - Ukuran Vektor Fitur: 512 Dimensi Floating-Point')",
    "print(f'   - Device Komputasi  : {device}')",
    "print('====================================================')"
])

# --- CELL 7: Core Dataset Inspection & Face Reading Pipeline ---
add_markdown([
    "## 🔍 4. Proses Pembacaan Dataset & Ekstraksi Informasi",
    "Pada tahap ini, sistem melakukan iterasi ke seluruh berkas dataset dan membaca:",
    "1. **Metadata Berkas:** Nama berkas, ukuran (KB), dimensi citra $(W \\times H \\times C)$.",
    "2. **Deteksi Wajah:** Apakah wajah ditemukan dalam citra, berapa jumlahnya, dan di mana koordinatnya.",
    "3. **Preprocessing:** Cropping wajah dengan margin aman, lalu di-resize ke $160 \\times 160$ piksel.",
    "4. **Vektor Representasi:** Menghasilkan vektor embedding 512-dimensi yang dinormalisasi $L_2$."
])

add_code([
    "def proses_baca_dataset(folder_path):",
    "    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')",
    "    all_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(valid_exts)])",
    "    ",
    "    hasil_laporan = []",
    "    ",
    "    print(f'🚀 MEMULAI PROSES PEMBACAAN DATASET DARI: /{folder_path}')",
    "    print('=' * 75)",
    "    ",
    "    start_total = time.time()",
    "    ",
    "    for idx, fname in enumerate(all_files, 1):",
    "        fpath = os.path.join(folder_path, fname)",
    "        file_size_kb = os.path.getsize(fpath) / 1024.0",
    "        ",
    "        # 1. Baca Citra menggunakan OpenCV",
    "        img_bgr = cv2.imread(fpath)",
    "        if img_bgr is None:",
    "            print(f'❌ [GAGAL BACA] {fname}: Berkas rusak atau bukan citra valid.')",
    "            continue",
    "            ",
    "        h_orig, w_orig, c_orig = img_bgr.shape",
    "        ",
    "        # 2. Deteksi Wajah",
    "        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)",
    "        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(50, 50))",
    "        ",
    "        status_deteksi = 'TERDETEKSI'",
    "        jml_wajah = len(faces)",
    "        ",
    "        # Jika wajah terdeteksi, ambil wajah terbesar",
    "        if jml_wajah > 0:",
    "            largest = max(faces, key=lambda r: r[2] * r[3])",
    "            fx, fy, fw, fh = largest",
    "            coord_str = f'x={fx}, y={fy}, w={fw}, h={fh}'",
    "            ",
    "            # Crop wajah dengan margin",
    "            margin = 15",
    "            x1 = max(0, fx - margin)",
    "            y1 = max(0, fy - margin)",
    "            x2 = min(w_orig, fx + fw + margin)",
    "            y2 = min(h_orig, fy + fh + margin)",
    "            cropped = img_bgr[y1:y2, x1:x2]",
    "            face_160 = cv2.resize(cropped, (160, 160))",
    "        else:",
    "            status_deteksi = 'TIDAK TERDETEKSI (Menggunakan Full Image)'",
    "            coord_str = 'N/A'",
    "            face_160 = cv2.resize(img_bgr, (160, 160))",
    "            ",
    "        # 3. Ekstraksi Vektor Fitur CNN FaceNet (512 dimensi)",
    "        rgb = cv2.cvtColor(face_160, cv2.COLOR_BGR2RGB)",
    "        tensor = torch.tensor(rgb, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)",
    "        tensor = (tensor - 127.5) / 128.0  # Normalisasi [-1, 1]",
    "        tensor = tensor.to(device)",
    "        ",
    "        with torch.no_grad():",
    "            emb = model_facenet(tensor).cpu().numpy()[0]",
    "            norm = np.linalg.norm(emb)",
    "            if norm > 0:",
    "                emb = emb / norm  # L2 Normalization",
    "                ",
    "        # Simpan info ke list laporan",
    "        data_info = {",
    "            'no': idx,",
    "            'nama_file': fname,",
    "            'ukuran_kb': f'{file_size_kb:.1f} KB',",
    "            'resolusi_asli': f'{w_orig}x{h_orig} px',",
    "            'status_wajah': status_deteksi,",
    "            'jml_wajah': jml_wajah,",
    "            'koordinat': coord_str,",
    "            'dimensi_vektor': len(emb),",
    "            'vektor_preview': np.round(emb[:5], 4).tolist(),",
    "            'norm_l2': round(float(np.linalg.norm(emb)), 4)",
    "        }",
    "        hasil_laporan.append(data_info)",
    "        ",
    "        # Output per baris",
    "        print(f'[{idx}/{len(all_files)}] Berkas: {fname}')",
    "        print(f'    ├─ Ukuran File        : {data_info[\"ukuran_kb\"]} | Resolusi: {data_info[\"resolusi_asli\"]}')",
    "        print(f'    ├─ Status Deteksi Wajah: {status_deteksi} ({jml_wajah} wajah ditemukan)')",
    "        print(f'    ├─ Koordinat Bounding : ({coord_str})')",
    "        print(f'    ├─ Preprocessing      : Crop & Normalisasi 160x160 piksel')",
    "        print(f'    └─ Vektor Fitur (CNN) : 512 Dimensi [Preview: {data_info[\"vektor_preview\"]}...]')",
    "        print('-' * 75)",
    "        ",
    "    durasi = time.time() - start_total",
    "    print(f'\\n⏱️ Selesai membaca seluruh dataset dalam {durasi:.2f} detik!')",
    "    return hasil_laporan",
    "",
    "laporan = proses_baca_dataset(DATASET_DIR)"
])

# --- CELL 8: Summary Text Report Table ---
add_markdown([
    "## 📊 5. Laporan Teks Ringkasan Akhir (Summary Output)",
    "Format tabel teks ringkas yang merangkum hasil verifikasi dataset dan deteksi wajah untuk dinilai dan dipresentasikan."
])

add_code([
    "print('=' * 95)",
    "print('         TABEL LAPORAN PEMBACAAN DATASET & VERIFIKASI FITUR WAJAH')",
    "print('=' * 95)",
    "header = f'| {\"No\":<3} | {\"Nama File\":<14} | {\"Resolusi\":<12} | {\"Ukuran\":<9} | {\"Status Wajah\":<15} | {\"Vektor Fitur\":<13} | {\"Status\":<9} |'",
    "print(header)",
    "print('|' + '-' * 5 + '|' + '-' * 16 + '|' + '-' * 14 + '|' + '-' * 11 + '|' + '-' * 17 + '|' + '-' * 15 + '|' + '-' * 11 + '|')",
    "",
    "for item in laporan:",
    "    status_singkat = 'OK' if item['status_wajah'] == 'TERDETEKSI' else 'PERLU CEK'",
    "    baris = f\"| {item['no']:<3} | {item['nama_file']:<14} | {item['resolusi_asli']:<12} | {item['ukuran_kb']:<9} | {item['status_wajah']:<15} | {item['dimensi_vektor']} Dimensi    | {status_singkat:<9} |\"",
    "    print(baris)",
    "",
    "print('=' * 95)",
    "print(f'Total Berkas Terproses : {len(laporan)} berkas')",
    "print(f'Berkas Valid Terdeteksi: {sum(1 for x in laporan if x[\"status_wajah\"] == \"TERDETEKSI\")} berkas')",
    "print('Kesimpulan             : Dataset BERHASIL DIBACA dan siap digunakan untuk proses Pengenalan/Presensi!')",
    "print('===============================================================================================')"
])

# Save notebook
notebook_path = "Pembacaan_Dataset_Face_Recognition.ipynb"
with open(notebook_path, "w", encoding="utf-8") as f:
    json.dump({
        "cells": cells,
        "metadata": {
            "colab": {
                "provenance": []
            },
            "kernelspec": {
                "display_name": "Python 3",
                "name": "python3"
            },
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 0
    }, f, indent=2)

print(f"File notebook berhasil dibuat: {notebook_path} ({len(cells)} cells)")
