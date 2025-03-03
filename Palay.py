import cv2
import torch
from ultralytics import YOLO
import os
from pathlib import Path

# Load your trained YOLOv8 model (replace 'best.pt' with your actual trained weights)
MODEL_PATH = "best.pt"
model = YOLO(MODEL_PATH)

# Define input and output folders
INPUT_FOLDER = "input_images"  # Folder containing test images
OUTPUT_FOLDER = "output_results"  # Folder to save images with detections

# Ensure output directory exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Process each image in the input folder
for img_file in os.listdir(INPUT_FOLDER):
    img_path = os.path.join(INPUT_FOLDER, img_file)
    
    # Perform inference
    results = model(img_path)

    # Draw bounding boxes on the image
    for result in results:
        img = result.plot()  # Render the detections on the image

        # Save the image with detections
        output_path = os.path.join(OUTPUT_FOLDER, img_file)
        cv2.imwrite(output_path, img)
        print(f"Processed: {img_file} -> {output_path}")

print("✅ Inference complete! Results saved in", OUTPUT_FOLDER)
