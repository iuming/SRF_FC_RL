@echo off
REM RF Cavity Real-Time Control (Command Line)
echo Starting RF Cavity Real-Time Control (Command Line)...
echo.

REM Activate conda environment and run real-time control
conda run -n RL python main.py realtime

echo.
echo Real-time control session ended.
pause
