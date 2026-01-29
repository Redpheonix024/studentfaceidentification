"""
Cross-Platform Training GUI Launcher
Replaces launch_training_gui.bat with Python script that works on all platforms
"""

import os
import sys
import subprocess
import platform

def find_python_executable():
    """
    Find the Python executable to use.
    Priority: virtual environment > system Python
    """
    
    # Check for virtual environments in common locations
    venv_names = ['enven', 'venv', '.venv', 'env']
    
    plat = platform.system()
    
    for venv_name in venv_names:
        if plat == 'Windows':
            # Windows: Scripts\python.exe
            python_path = os.path.join(venv_name, 'Scripts', 'python.exe')
        else:
            # Linux/macOS: bin/python
            python_path = os.path.join(venv_name, 'bin', 'python')
        
        if os.path.exists(python_path):
            print(f"✅ Found virtual environment: {venv_name}")
            return python_path
    
    # Fall back to system Python
    print("ℹ️  Using system Python (no virtual environment found)")
    return sys.executable

def check_dependencies():
    """Quick check for critical dependencies"""
    try:
        import tkinter
        return True
    except ImportError:
        print("❌ tkinter not found. This is required for the GUI.")
        print("\nInstallation instructions:")
        
        if platform.system() == "Windows":
            print("  tkinter usually comes with Python. Try reinstalling Python.")
        elif platform.system() == "Linux":
            print("  Ubuntu/Debian: sudo apt-get install python3-tk")
            print("  Fedora: sudo dnf install python3-tkinter")
        elif platform.system() == "Darwin":
            print("  macOS: tkinter should be included with Python from python.org")
        
        return False

def main():
    """Launch the training GUI"""
    print("="*60)
    print("Face Recognition Model Training GUI Launcher")
    print("="*60)
    print()
    
    # Find Python executable
    python_exe = find_python_executable()
    print(f"Python: {python_exe}")
    print()
    
    # Check dependencies
    if not check_dependencies():
        print("\n⚠️  Critical dependency missing.")
        input("Press Enter to exit...")
        return 1
    
    # Check if train_gui.py exists
    if not os.path.exists('train_gui.py'):
        print("❌ train_gui.py not found in current directory")
        print(f"   Current directory: {os.getcwd()}")
        print("\n   Make sure to run this script from the project root.")
        input("Press Enter to exit...")
        return 1
    
    # Launch the GUI
    print("Starting Face Recognition Model Training GUI...")
    print()
    
    try:
        # Run train_gui.py with the found Python executable
        result = subprocess.run(
            [python_exe, 'train_gui.py'],
            check=False
        )
        
        return result.returncode
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ Error launching GUI: {e}")
        input("Press Enter to exit...")
        return 1

if __name__ == "__main__":
    exit_code = main()
    
    # On Windows, pause before closing if there was an error
    if platform.system() == "Windows" and exit_code != 0:
        input("\nPress Enter to exit...")
    
    sys.exit(exit_code)
