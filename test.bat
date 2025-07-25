@echo off
REM RF Cavity Control Testing Script
echo Starting RF Cavity Control Testing...
echo.

REM Activate conda environment and run testing
conda run -n RL python main.py test

echo.
echo Testing completed! Check the results folder for analysis plots.
pause
