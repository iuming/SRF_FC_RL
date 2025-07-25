@echo off
REM RF Cavity Control Training Script
echo Starting RF Cavity Control Training...
echo.

REM Activate conda environment and run training
conda run -n RL python main.py train

echo.
echo Training completed! Check the logs and tensorboard for results.
pause
