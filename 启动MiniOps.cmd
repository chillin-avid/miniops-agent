@echo off
cd /d "%~dp0"
title MiniOps Agent
python launcher.py
if errorlevel 1 (
  echo.
  echo MiniOps failed to start. Please copy the error above.
)
echo.
echo Press any key to close this window.
pause >nul
