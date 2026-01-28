@echo off
REM Batch file to run model testing script

echo ========================================
echo   Face Detection Model Tester
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
echo Starting model tester...
echo.

REM Run with default model (4 - SOTA)
python test_models.py

REM Deactivate when done
deactivate

echo.
echo ========================================
echo   Testing complete
echo ========================================
pause
