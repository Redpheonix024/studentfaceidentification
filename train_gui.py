#!/usr/bin/env python3
"""
GUI Interface for Face Recognition Model Training
Provides an easy-to-use interface for training and merging models
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import threading
import subprocess
import os
import sys
import pickle
from pathlib import Path

class TrainingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition Model Trainer")
        self.root.geometry("900x700")
        
        # Configure grid weights
        self.root.grid_rowconfigure(1, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        # Variables
        self.python_exe = ".\\enven\\Scripts\\python.exe"
        self.is_training = False
        
        # Create UI
        self.create_ui()
        
        # Check environment
        self.root.after(100, self.check_environment)
    
    def create_ui(self):
        """Create the main UI"""
        
        # Title Frame
        title_frame = ttk.Frame(self.root, padding="10")
        title_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        title_label = ttk.Label(title_frame, text="Face Recognition Model Trainer", 
                               font=('Arial', 16, 'bold'))
        title_label.pack()
        
        subtitle_label = ttk.Label(title_frame, text="Train and merge face recognition models",
                                   font=('Arial', 10))
        subtitle_label.pack()
        
        # Main Container
        main_container = ttk.Frame(self.root, padding="10")
        main_container.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        main_container.grid_rowconfigure(1, weight=1)
        main_container.grid_columnconfigure(0, weight=1)
        
        # Training Options Frame
        options_frame = ttk.LabelFrame(main_container, text="Training Options", padding="10")
        options_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # Button grid
        btn_row = 0
        
        # Step 1: Train Class Photos
        ttk.Label(options_frame, text="Step 1:", font=('Arial', 9, 'bold')).grid(
            row=btn_row, column=0, sticky=tk.W, pady=5)
        self.btn_train_class = ttk.Button(options_frame, text="Train Class Photos Model",
                                          command=self.train_class_photos, width=30)
        self.btn_train_class.grid(row=btn_row, column=1, padx=5, pady=5)
        ttk.Label(options_frame, text="(students faces/)").grid(
            row=btn_row, column=2, sticky=tk.W, padx=5)
        btn_row += 1
        
        # Step 2: Train Passport Photos
        ttk.Label(options_frame, text="Step 2:", font=('Arial', 9, 'bold')).grid(
            row=btn_row, column=0, sticky=tk.W, pady=5)
        self.btn_train_passport = ttk.Button(options_frame, text="Train Passport Photos Model",
                                            command=self.train_passport_photos, width=30)
        self.btn_train_passport.grid(row=btn_row, column=1, padx=5, pady=5)
        ttk.Label(options_frame, text="(student faces paasport/)").grid(
            row=btn_row, column=2, sticky=tk.W, padx=5)
        btn_row += 1
        
        # Step 3: Merge Models
        ttk.Label(options_frame, text="Step 3:", font=('Arial', 9, 'bold')).grid(
            row=btn_row, column=0, sticky=tk.W, pady=5)
        self.btn_merge = ttk.Button(options_frame, text="Merge Models",
                                    command=self.merge_models, width=30)
        self.btn_merge.grid(row=btn_row, column=1, padx=5, pady=5)
        ttk.Label(options_frame, text="(combine class + passport)").grid(
            row=btn_row, column=2, sticky=tk.W, padx=5)
        btn_row += 1
        
        # Separator
        ttk.Separator(options_frame, orient='horizontal').grid(
            row=btn_row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        btn_row += 1
        
        # Quick Actions
        ttk.Label(options_frame, text="Quick Actions:", font=('Arial', 9, 'bold')).grid(
            row=btn_row, column=0, sticky=tk.W, pady=5)
        self.btn_train_all = ttk.Button(options_frame, text="Train All (Complete Pipeline)",
                                        command=self.train_all, width=30,
                                        style='Accent.TButton')
        self.btn_train_all.grid(row=btn_row, column=1, padx=5, pady=5)
        ttk.Label(options_frame, text="(all steps at once)").grid(
            row=btn_row, column=2, sticky=tk.W, padx=5)
        btn_row += 1
        
        # Train filtered model
        ttk.Label(options_frame, text="", font=('Arial', 9, 'bold')).grid(
            row=btn_row, column=0, sticky=tk.W, pady=5)
        self.btn_train_filtered = ttk.Button(options_frame, text="Train Passport+Class Filtered",
                                            command=self.train_filtered, width=30)
        self.btn_train_filtered.grid(row=btn_row, column=1, padx=5, pady=5)
        ttk.Label(options_frame, text="(only students with both)").grid(
            row=btn_row, column=2, sticky=tk.W, padx=5)
        btn_row += 1
        
        # Separator
        ttk.Separator(options_frame, orient='horizontal').grid(
            row=btn_row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        btn_row += 1
        
        # Utilities
        ttk.Label(options_frame, text="Utilities:", font=('Arial', 9, 'bold')).grid(
            row=btn_row, column=0, sticky=tk.W, pady=5)
        self.btn_view_models = ttk.Button(options_frame, text="View Available Models",
                                         command=self.view_models, width=30)
        self.btn_view_models.grid(row=btn_row, column=1, padx=5, pady=5)
        self.btn_open_folder = ttk.Button(options_frame, text="Open Models Folder",
                                         command=self.open_models_folder, width=20)
        self.btn_open_folder.grid(row=btn_row, column=2, padx=5, pady=5)
        
        # Progress/Output Frame
        output_frame = ttk.LabelFrame(main_container, text="Training Output", padding="10")
        output_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        output_frame.grid_rowconfigure(0, weight=1)
        output_frame.grid_columnconfigure(0, weight=1)
        
        # Output text area
        self.output_text = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD,
                                                     height=20, font=('Consolas', 9))
        self.output_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Status bar
        status_frame = ttk.Frame(self.root)
        status_frame.grid(row=2, column=0, sticky=(tk.W, tk.E))
        
        self.status_label = ttk.Label(status_frame, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, pady=5)
        
        self.progress = ttk.Progressbar(status_frame, mode='indeterminate', length=200)
        self.progress.pack(side=tk.RIGHT, padx=5, pady=5)
    
    def log(self, message, level="INFO"):
        """Log message to output window"""
        timestamp = ""
        if level == "INFO":
            prefix = "[INFO] "
        elif level == "SUCCESS":
            prefix = "[SUCCESS] "
        elif level == "ERROR":
            prefix = "[ERROR] "
        elif level == "HEADER":
            prefix = "\n" + "="*60 + "\n"
            message = f"{message}\n" + "="*60
        else:
            prefix = ""
        
        self.output_text.insert(tk.END, f"{prefix}{message}\n")
        self.output_text.see(tk.END)
        self.root.update()
    
    def check_environment(self):
        """Check if environment is properly set up"""
        self.log("Checking environment...", "HEADER")
        
        # Check Python executable
        if not os.path.exists(self.python_exe):
            self.log(f"Python executable not found: {self.python_exe}", "ERROR")
            self.log("Please make sure you have activated the virtual environment", "ERROR")
        else:
            self.log(f"Python executable: {self.python_exe}", "SUCCESS")
        
        # Check directories
        dirs_to_check = [
            ("Class photos", "students faces"),
            ("Passport photos", "student faces paasport"),
            ("Models", "models")
        ]
        
        for name, path in dirs_to_check:
            if os.path.exists(path):
                count = len([f for f in os.listdir(path) if not f.startswith('.')])
                self.log(f"{name} directory: {path} ({count} items)", "SUCCESS")
            else:
                self.log(f"{name} directory not found: {path}", "ERROR")
        
        # Check ONNX models
        onnx_models = [
            "models/retinaface_r50.onnx",
            "models/arcface_r100.onnx"
        ]
        
        for model in onnx_models:
            if os.path.exists(model):
                size_mb = os.path.getsize(model) / (1024*1024)
                self.log(f"ONNX model: {model} ({size_mb:.1f} MB)", "SUCCESS")
            else:
                self.log(f"ONNX model not found: {model}", "ERROR")
        
        self.log("\nEnvironment check complete!", "SUCCESS")
        self.set_status("Ready")
    
    def set_status(self, message):
        """Update status bar"""
        self.status_label.config(text=message)
    
    def disable_buttons(self):
        """Disable all training buttons"""
        self.is_training = True
        self.btn_train_class.config(state=tk.DISABLED)
        self.btn_train_passport.config(state=tk.DISABLED)
        self.btn_merge.config(state=tk.DISABLED)
        self.btn_train_all.config(state=tk.DISABLED)
        self.btn_train_filtered.config(state=tk.DISABLED)
        self.progress.start()
    
    def enable_buttons(self):
        """Enable all training buttons"""
        self.is_training = False
        self.btn_train_class.config(state=tk.NORMAL)
        self.btn_train_passport.config(state=tk.NORMAL)
        self.btn_merge.config(state=tk.NORMAL)
        self.btn_train_all.config(state=tk.NORMAL)
        self.btn_train_filtered.config(state=tk.NORMAL)
        self.progress.stop()
    
    def run_script(self, script_name, description):
        """Run a training script"""
        if not os.path.exists(script_name):
            self.log(f"Script not found: {script_name}", "ERROR")
            messagebox.showerror("Error", f"Script not found: {script_name}")
            return False
        
        self.log(f"\nStarting: {description}", "HEADER")
        self.log(f"Running: {script_name}")
        self.set_status(f"Running: {description}...")
        
        try:
            # Run the script with UTF-8 encoding
            process = subprocess.Popen(
                [self.python_exe, script_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                universal_newlines=True,
                encoding='utf-8',
                errors='replace'
            )
            
            # Stream output
            for line in process.stdout:
                self.log(line.rstrip())
            
            process.wait()
            
            if process.returncode == 0:
                self.log(f"\n{description} completed successfully!", "SUCCESS")
                self.set_status(f"{description} completed")
                return True
            else:
                self.log(f"\n{description} failed with code {process.returncode}", "ERROR")
                self.set_status(f"{description} failed")
                return False
                
        except Exception as e:
            self.log(f"Error running script: {e}", "ERROR")
            self.set_status("Error")
            return False
    
    def train_class_photos(self):
        """Train class photos model"""
        def task():
            self.disable_buttons()
            self.run_script("train_class_photos_r100.py", "Train Class Photos Model")
            self.enable_buttons()
        
        threading.Thread(target=task, daemon=True).start()
    
    def train_passport_photos(self):
        """Train passport photos model"""
        def task():
            self.disable_buttons()
            self.run_script("train_passport_insightface.py", "Train Passport Photos Model")
            self.enable_buttons()
        
        threading.Thread(target=task, daemon=True).start()
    
    def merge_models(self):
        """Merge models"""
        def task():
            self.disable_buttons()
            success = self.run_script("merge_models_r100.py", "Merge Models")
            if success:
                messagebox.showinfo("Success", "Models merged successfully!\n\n"
                                   "Output: models/face_embeddings_merged_r100.pkl")
            self.enable_buttons()
        
        threading.Thread(target=task, daemon=True).start()
    
    def train_filtered(self):
        """Train filtered passport+class model"""
        def task():
            self.disable_buttons()
            success = self.run_script("train_and_merge_passport_class.py", 
                                     "Train Passport+Class Filtered Model")
            if success:
                messagebox.showinfo("Success", 
                                   "Filtered model created successfully!\n\n"
                                   "Output: models/face_embeddings_passport_class_filtered_r100.pkl\n"
                                   "Contains only students with both passport and class photos.")
            self.enable_buttons()
        
        threading.Thread(target=task, daemon=True).start()
    
    def train_all(self):
        """Run complete training pipeline"""
        response = messagebox.askyesno(
            "Train All",
            "This will run the complete training pipeline:\n\n"
            "1. Train Class Photos Model\n"
            "2. Train Passport Photos Model\n"
            "3. Merge Models\n\n"
            "This may take several minutes. Continue?"
        )
        
        if not response:
            return
        
        def task():
            self.disable_buttons()
            
            # Step 1
            if not self.run_script("train_class_photos_r100.py", "Train Class Photos Model"):
                self.enable_buttons()
                return
            
            # Step 2
            if not self.run_script("train_passport_insightface.py", "Train Passport Photos Model"):
                self.enable_buttons()
                return
            
            # Step 3
            if not self.run_script("merge_models_r100.py", "Merge Models"):
                self.enable_buttons()
                return
            
            self.log("\n" + "="*60, "SUCCESS")
            self.log("COMPLETE PIPELINE FINISHED SUCCESSFULLY!", "SUCCESS")
            self.log("="*60, "SUCCESS")
            
            messagebox.showinfo("Success", 
                               "Complete training pipeline finished!\n\n"
                               "Output: models/face_embeddings_merged_r100.pkl")
            
            self.enable_buttons()
        
        threading.Thread(target=task, daemon=True).start()
    
    def view_models(self):
        """View available models"""
        models_dir = "models"
        
        if not os.path.exists(models_dir):
            messagebox.showerror("Error", f"Models directory not found: {models_dir}")
            return
        
        # Find all .pkl files
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
        
        if not model_files:
            messagebox.showinfo("No Models", f"No .pkl model files found in {models_dir}")
            return
        
        # Create info window
        info_window = tk.Toplevel(self.root)
        info_window.title("Available Models")
        info_window.geometry("800x500")
        
        # Title
        title_label = ttk.Label(info_window, text=f"Found {len(model_files)} models in {models_dir}/",
                               font=('Arial', 12, 'bold'))
        title_label.pack(pady=10)
        
        # Scrolled text for model info
        text_area = scrolledtext.ScrolledText(info_window, wrap=tk.WORD, font=('Consolas', 9))
        text_area.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Load and display model info
        for i, filename in enumerate(sorted(model_files), start=1):
            model_path = os.path.join(models_dir, filename)
            size_kb = os.path.getsize(model_path) / 1024
            
            text_area.insert(tk.END, f"\n[{i}] {filename}\n")
            text_area.insert(tk.END, f"    Size: {size_kb:.1f} KB\n")
            
            # Try to load model info
            try:
                with open(model_path, 'rb') as f:
                    embeddings, names = pickle.load(f)
                num_embeddings = len(embeddings)
                num_students = len(set(names))
                text_area.insert(tk.END, f"    Students: {num_students}\n")
                text_area.insert(tk.END, f"    Embeddings: {num_embeddings}\n")
                
                if num_students <= 10:
                    text_area.insert(tk.END, f"    Names: {', '.join(sorted(set(names)))}\n")
            except:
                text_area.insert(tk.END, f"    (Unable to load model info)\n")
        
        text_area.config(state=tk.DISABLED)
    
    def open_models_folder(self):
        """Open models folder in file explorer"""
        models_dir = os.path.abspath("models")
        
        if not os.path.exists(models_dir):
            messagebox.showerror("Error", f"Models directory not found: {models_dir}")
            return
        
        # Open in file explorer (Windows, Mac, Linux)
        if sys.platform == "win32":
            os.startfile(models_dir)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", models_dir])
        else:
            subprocess.Popen(["xdg-open", models_dir])

def main():
    root = tk.Tk()
    
    # Set style
    style = ttk.Style()
    style.theme_use('clam')
    
    app = TrainingGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
