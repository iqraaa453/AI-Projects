@echo off
cd /d "%~dp0"
echo === RL Dynamic Pricing Project ===
echo.

python -c "import numpy, flask, torch" 2>nul || (
  echo Installing numpy...
  pip install -r requirements.txt
)

echo [1/4] Training pipeline (QL + DDPG + comparison + sensitivity)...
python train_all.py
if errorlevel 1 exit /b 1

echo.
echo [2/4] Done.
echo.
echo Live dashboard (Simulator + Competitors use real trained agents):
echo   python dashboard_server.py
echo   Open http://127.0.0.1:5050
echo.
echo Static charts only: rl_dashboard.html (file:// — live tabs need server)
echo Report: REPORT.md
pause
