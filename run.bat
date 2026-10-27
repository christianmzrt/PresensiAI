@echo off
echo ===================================================
echo   PresensiAI - Smart Biometric Attendance System
echo   CNN FaceNet (512-d) ^& Real-Time Anti-Spoofing
echo ===================================================
echo.
echo Mengaktifkan Virtual Environment...
call .\venv\Scripts\activate

echo.
echo Menjalankan Server Flask...
python app.py
pause
