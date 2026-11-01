@echo off
echo Starting Loan Default Prediction System...
start "Loan Prediction Backend" python back_end/main.py
start "Loan Prediction Frontend" python -m http.server 8000 --directory front_end
timeout /t 2 >nul
start http://127.0.0.1:8000
echo.
echo ========================================================
echo Backend is running at:  http://127.0.0.1:5000/
echo Frontend is running at: http://127.0.0.1:8000/
echo ========================================================

