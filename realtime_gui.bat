@echo off
REM RF Cavity Real-Time Control GUI
echo Starting RF Cavity Real-Time Control GUI...
echo.

REM Activate conda environment and run GUI
conda run -n RL python main.py realtime-gui

echo.
echo GUI session ended.
pause
