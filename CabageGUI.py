import cv2
import torch
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from ultralytics import YOLO
from PIL import Image, ImageTk
import numpy as np
import os
import csv
import time
from datetime import datetime
from collections import Counter
import webbrowser

class YoloApp:
    def __init__(self, root, model_path="best.pt"):
        self.root = root
        self.root.title("YOLOv8 Cabbage CV Station")
        self.root.geometry("1100x800")
        
        # Theme Colors
        self.light_bg = "#f4f4f9"
        self.light_fg = "#333333"
        self.dark_bg = "#1e1e1e"
        self.dark_fg = "#ffffff"
        self.current_theme = "light"
        self.root.configure(bg=self.light_bg)
        
        # Load Model
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            messagebox.showerror("Model Error", f"Failed to load model:\n{e}")
            self.root.destroy()
            return

        # State Variables
        self.cap = cv2.VideoCapture(0)
        self.is_video_playing = True
        self.current_frame = None
        self.prev_time = 0  # For FPS calculation
        
        # Control Variables
        self.live_inference_var = tk.BooleanVar(value=False)
        self.conf_var = tk.DoubleVar(value=0.50)
        self.iou_var = tk.DoubleVar(value=0.45)
        self.dark_mode_var = tk.BooleanVar(value=False)
        
        # Setup Core Components
        self.setup_directories()
        self.setup_menu()
        self.setup_gui()
        
        # Handle Window Close Event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Start Video Loop
        self.update_video()

    def setup_directories(self):
        """Creates folders for raw/processed images and sets up the CSV log file."""
        os.makedirs("output/raw", exist_ok=True)
        os.makedirs("output/processed", exist_ok=True)
        os.makedirs("output/reports", exist_ok=True)
        self.csv_path = "output/inference_logs.csv"
        
        if not os.path.exists(self.csv_path):
            with open(self.csv_path, mode='w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Filename", "Inference Time (ms)", "Detections", "Confidence", "IoU"])

    def setup_menu(self):
        """Sets up the top menu bar."""
        menubar = tk.Menu(self.root)
        
        # File Menu
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Open Image...", command=self.upload_image)
        file_menu.add_command(label="Open Video...", command=self.upload_video)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_closing)
        menubar.add_cascade(label="File", menu=file_menu)
        
        # Tools Menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        tools_menu.add_command(label="Export Session Report", command=self.export_report)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        
        # View Menu
        view_menu = tk.Menu(menubar, tearoff=0)
        view_menu.add_checkbutton(label="Dark Mode", variable=self.dark_mode_var, command=self.toggle_dark_mode)
        menubar.add_cascade(label="View", menu=view_menu)
        
        self.root.config(menu=menubar)

    def setup_gui(self):
        """Constructs the Tkinter UI layout."""
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.apply_theme() # Sets colors based on current mode

        # Main Layout
        self.main_frame = ttk.Frame(self.root, padding="20")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        self.display_frame = ttk.Frame(self.main_frame)
        self.display_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.control_frame = ttk.Frame(self.main_frame, width=320)
        self.control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(20, 0))
        
        # Image Display Label & FPS overlay
        self.video_label = ttk.Label(self.display_frame, text="Loading feed...", anchor=tk.CENTER)
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # --- CONTROLS SECTION ---
        ttk.Label(self.control_frame, text="System Controls", style='Header.TLabel').pack(pady=(0, 10))
        
        self.btn_infer = ttk.Button(self.control_frame, text="Capture & Infer", style='BigClear.TButton', command=self.capture_and_infer)
        self.btn_infer.pack(fill=tk.X, pady=4)
        
        self.btn_resume = ttk.Button(self.control_frame, text="Resume Camera/Video", style='BigClear.TButton', command=self.resume_feed)
        self.btn_resume.pack(fill=tk.X, pady=4)
        
        # --- SETTINGS SECTION ---
        self.settings_frame = ttk.LabelFrame(self.control_frame, text="Inference Settings", padding="10")
        self.settings_frame.pack(fill=tk.X, pady=(15, 5))
        
        # Live Toggle
        ttk.Checkbutton(self.settings_frame, text="Live Continuous Inference", variable=self.live_inference_var, style='TCheckbutton').pack(anchor=tk.W, pady=(0, 10))
        
        # Confidence Slider
        ttk.Label(self.settings_frame, text="Confidence Threshold:").pack(anchor=tk.W)
        self.conf_slider = ttk.Scale(self.settings_frame, from_=0.1, to=0.95, orient=tk.HORIZONTAL, variable=self.conf_var, command=self.update_sliders)
        self.conf_slider.pack(fill=tk.X, pady=2)
        self.conf_label = ttk.Label(self.settings_frame, text="50%")
        self.conf_label.pack(anchor=tk.E)
        
        # IoU Slider
        ttk.Label(self.settings_frame, text="IoU Threshold (NMS):").pack(anchor=tk.W)
        self.iou_slider = ttk.Scale(self.settings_frame, from_=0.1, to=0.95, orient=tk.HORIZONTAL, variable=self.iou_var, command=self.update_sliders)
        self.iou_slider.pack(fill=tk.X, pady=2)
        self.iou_label = ttk.Label(self.settings_frame, text="45%")
        self.iou_label.pack(anchor=tk.E)

        # --- DETECTION REPORT SECTION ---
        self.report_frame = ttk.LabelFrame(self.control_frame, text="Detection Report", padding="15")
        self.report_frame.pack(fill=tk.X, pady=(15, 5))

        self.report_fps_var = tk.StringVar(value="FPS: --")
        self.report_time_var = tk.StringVar(value="Time: -- ms")
        self.report_counts_var = tk.StringVar(value="Detections: None")

        ttk.Label(self.report_frame, textvariable=self.report_fps_var, font=("Segoe UI", 10, "bold"), foreground="#0078D7").pack(anchor=tk.W, pady=2)
        ttk.Label(self.report_frame, textvariable=self.report_time_var).pack(anchor=tk.W, pady=2)
        ttk.Label(self.report_frame, textvariable=self.report_counts_var, wraplength=260, font=("Segoe UI", 11, "bold")).pack(anchor=tk.W, pady=2)
        
        # Status Label
        self.status_var = tk.StringVar(value="Status: Ready")
        self.status_label = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W, style='Status.TLabel', padding=5)
        self.status_label.pack(side=tk.BOTTOM, fill=tk.X)

    def apply_theme(self):
        """Applies light or dark theme colors to the ttk elements."""
        bg = self.dark_bg if self.current_theme == "dark" else self.light_bg
        fg = self.dark_fg if self.current_theme == "dark" else self.light_fg
        
        self.root.configure(bg=bg)
        self.style.configure('TFrame', background=bg)
        self.style.configure('TLabel', background=bg, foreground=fg, font=("Segoe UI", 10))
        self.style.configure('TLabelframe', background=bg, foreground=fg)
        self.style.configure('TLabelframe.Label', background=bg, font=("Segoe UI", 11, "bold"), foreground=fg)
        self.style.configure('TCheckbutton', background=bg, foreground=fg, font=("Segoe UI", 10))
        self.style.configure('Header.TLabel', font=("Segoe UI", 14, "bold"), background=bg, foreground=fg)
        
        self.style.configure('BigClear.TButton', font=('Segoe UI', 11, 'bold'), padding=(10, 10), background="#0078D7", foreground="white")
        self.style.map('BigClear.TButton', background=[('active', '#005A9E'), ('disabled', '#555555')])
        
        status_bg = "#2d2d30" if self.current_theme == "dark" else "#e1e1e6"
        self.style.configure('Status.TLabel', background=status_bg, foreground=fg, font=("Segoe UI", 9))

    def toggle_dark_mode(self):
        """Switches between dark and light themes."""
        self.current_theme = "dark" if self.dark_mode_var.get() else "light"
        self.apply_theme()

    def update_sliders(self, event=None):
        """Updates the numerical labels next to sliders."""
        self.conf_label.config(text=f"{int(self.conf_var.get() * 100)}%")
        self.iou_label.config(text=f"{int(self.iou_var.get() * 100)}%")

    def update_video(self):
        """Pulls frames from webcam or video file and handles live inference."""
        if self.is_video_playing:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                
                if self.live_inference_var.get():
                    self.run_inference(frame, log_to_csv=False) # Run live without spamming CSV
                else:
                    self.display_image(frame)
                    self.report_fps_var.set("FPS: -- (Live Info Off)")
            else:
                # Video ended, loop back to start if it's a video file
                if not isinstance(self.cap, int) and self.cap.get(cv2.CAP_PROP_POS_FRAMES) == self.cap.get(cv2.CAP_PROP_FRAME_COUNT):
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        self.video_loop_id = self.root.after(15, self.update_video)

    def display_image(self, frame_bgr):
        """Updates the UI with the image."""
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)
        img_pil.thumbnail((800, 650), Image.Resampling.LANCZOS)
        
        imgtk = ImageTk.PhotoImage(image=img_pil)
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)

    def upload_image(self):
        file_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if file_path:
            self.is_video_playing = False
            frame = cv2.imread(file_path)
            if frame is not None:
                self.current_frame = frame
                self.display_image(frame)
                self.status_var.set(f"Status: Loaded Image {os.path.basename(file_path)}")

    def upload_video(self):
        file_path = filedialog.askopenfilename(title="Select Video", filetypes=[("Video Files", "*.mp4 *.avi *.mkv")])
        if file_path:
            self.cap.release()
            self.cap = cv2.VideoCapture(file_path)
            self.is_video_playing = True
            self.status_var.set(f"Status: Playing Video {os.path.basename(file_path)}")

    def resume_feed(self):
        """Resumes the active feed (webcam or video file)."""
        # If no file is loaded or cap is closed, default to webcam
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(0)
        
        self.is_video_playing = True
        self.status_var.set("Status: Feed Active")
        self.report_counts_var.set("Detections: None")
        self.report_time_var.set("Time: -- ms")

    def capture_and_infer(self):
        """Triggers manual inference on the current frame and logs to CSV."""
        if self.current_frame is None:
            messagebox.showwarning("Warning", "No frame available.")
            return
        
        self.is_video_playing = False # Pause video to view result
        self.status_var.set("Status: Running Manual Inference...")
        self.root.update()
        
        self.run_inference(self.current_frame.copy(), log_to_csv=True)
        self.status_var.set("Status: Inference Saved & Logged.")

    def run_inference(self, frame, log_to_csv=False):
        """Core inference logic. Updates UI, calculates FPS, and optionally logs."""
        start_time = time.time()
        
        # Run Model with Slider Values
        results = self.model(frame, conf=self.conf_var.get(), iou=self.iou_var.get(), verbose=False)
        result = results[0]
        detected_img = result.plot()
        
        # FPS Calculation
        current_time = time.time()
        fps = 1 / (current_time - self.prev_time) if self.prev_time else 0
        self.prev_time = current_time
        
        # Extraction
        inf_time = round(sum(result.speed.values()), 2)
        classes_detected = result.boxes.cls.cpu().numpy()
        class_names = [self.model.names[int(c)] for c in classes_detected]
        counts = Counter(class_names)
        detections_str = ", ".join([f"{k}: {v}" for k, v in counts.items()]) if counts else "None"
        
        # Update UI
        self.display_image(detected_img)
        self.report_fps_var.set(f"FPS: {int(fps)}")
        self.report_time_var.set(f"Time: {inf_time} ms")
        self.report_counts_var.set(f"Detections:\n{detections_str}")
        
        if log_to_csv:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"infer_{timestamp}.jpg"
            cv2.imwrite(os.path.join("output/raw", filename), frame)
            cv2.imwrite(os.path.join("output/processed", filename), detected_img)
            
            with open(self.csv_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([timestamp, filename, inf_time, detections_str, self.conf_var.get(), self.iou_var.get()])

    def export_report(self):
        """Generates a professional HTML report from the CSV and opens it."""
        if not os.path.exists(self.csv_path):
            messagebox.showinfo("Info", "No data to export yet.")
            return
            
        report_path = os.path.join(os.getcwd(), "output", "reports", f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
        
        html_content = f"""
        <html>
        <head>
            <title>YOLOv8 Session Report</title>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px; background-color: #f4f4f9; color: #333; }}
                h1 {{ color: #0078D7; border-bottom: 2px solid #0078D7; padding-bottom: 10px; }}
                table {{ width: 100%; border-collapse: collapse; margin-top: 20px; background-color: #fff; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
                th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid #ddd; }}
                th {{ background-color: #0078D7; color: white; }}
                tr:hover {{ background-color: #f1f1f1; }}
            </style>
        </head>
        <body>
            <h1>Computer Vision Inspection Report</h1>
            <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <table>
                <tr><th>Timestamp</th><th>Filename</th><th>Inference Time (ms)</th><th>Detections</th><th>Conf</th><th>IoU</th></tr>
        """
        
        with open(self.csv_path, mode='r') as f:
            reader = csv.reader(f)
            next(reader) # Skip header
            for row in reader:
                html_content += f"<tr><td>{row[0]}</td><td>{row[1]}</td><td>{row[2]}</td><td>{row[3]}</td><td>{row[4]}</td><td>{row[5]}</td></tr>"
                
        html_content += """
            </table>
        </body>
        </html>
        """
        
        with open(report_path, "w") as f:
            f.write(html_content)
            
        webbrowser.open(f"file://{report_path}")
        self.status_var.set("Status: Report Exported successfully.")

    def on_closing(self):
        if self.cap.isOpened():
            self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = YoloApp(root, model_path="best.pt")
    root.mainloop()