import cv2
import torch
import tkinter as tk
from tkinter import Label, Button
from ultralytics import YOLO
from PIL import Image, ImageTk
import numpy as np

# Load the YOLO model (change 'best.pt' to your trained model)
MODEL_PATH = "best.pt"
model = YOLO(MODEL_PATH)

# Initialize Tkinter window
root = tk.Tk()
root.title("YOLOv8 Webcam Inference")
root.geometry("800x600")

# Capture video from webcam
cap = cv2.VideoCapture(0)  # Change to 1 or 2 if using external webcams

# Tkinter Labels
video_label = Label(root)
video_label.pack()

# Function to update the webcam feed
def update_video():
    ret, frame = cap.read()
    if ret:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)
    root.after(10, update_video)  # Refresh every 10ms

# Function to capture an image and perform inference
def capture_and_infer():
    ret, frame = cap.read()
    if ret:
        # Perform YOLO inference
        results = model(frame)
        for result in results:
            detected_img = result.plot()  # Draw bounding boxes on the frame

        # Convert to PIL Image for display
        img_rgb = cv2.cvtColor(detected_img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)

        # Show the detection results in a new window
        show_result_window(img_pil)

# Function to show inference results in a new window
def show_result_window(image):
    new_window = tk.Toplevel(root)
    new_window.title("Inference Result")
    new_window.geometry("800x600")

    img_display = ImageTk.PhotoImage(image=image)
    result_label = Label(new_window, image=img_display)
    result_label.image = img_display
    result_label.pack()

# Button to capture and infer
capture_button = Button(root, text="Capture & Infer", command=capture_and_infer, font=("Arial", 14), bg="blue", fg="white")
capture_button.pack(pady=20)

# Start updating video
update_video()

# Run Tkinter loop
root.mainloop()

# Release webcam after closing the window
cap.release()
cv2.destroyAllWindows()
