@echo off
cd /d "%~dp0"
echo Starting live RL dashboard at http://127.0.0.1:5050
echo (Simulator + Competitors use your trained Python agents)
python dashboard_server.py
