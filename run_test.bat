@echo off
REM Batch file to run test_amd.py with proper environment

echo ========================================
echo   Face Recognition Video Player
echo   AMD GPU Accelerated
echo ========================================
echo.

REM Change to script directory
cd /d "%~dp0"

REM Check if virtual environment exists
if not exist "enven\Scripts\python.exe" (
    echo ERROR: Virtual environment not found!
    echo Please ensure enven folder exists in: %CD%
    pause
    exit /b 1
)

echo Activating virtual environment...
call enven\Scripts\activate.bat

echo.
echo Starting video player...
echo.

REM Run with default model (4 - SOTA)
python test_amd.py

REM Deactivate when done
deactivate

echo.
echo ========================================
echo   Video player closed
echo ========================================
pause
