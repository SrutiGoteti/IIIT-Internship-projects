# vision_project/src/train_model.py

from ultralytics import YOLO

# --- Part A: Train the Model ---

# CRITICAL: Use YOLOv8n (nano) for stability and low RAM usage on CPU
model = YOLO('yolov8n.pt')

# 2. Start the training process
print("--- Starting Stable Training on Laptop (YOLOv8n) ---")
results = model.train(
    # NOTE: Assuming your data.yaml file has the correct local paths (e.g., train/images)
    data='../dataset/data.yaml',            
    epochs=50,                              # Enough epochs to learn from 84 images
    # CRITICAL: Reduced batch size for low RAM/CPU usage (Try 4 if 8 fails)
    batch=8,                                
    imgsz=640,
    project='../runs/train',                
    name='custom_yolo_n_stable_run'         # New descriptive name
)

print("\nTraining complete! Trained weights are saved to: runs/train/custom_yolo_n_stable_run/weights/best.pt")

# --- Part B: Evaluate the Trained Model and Save Metrics ---

# 1. Reload the 'best' weights from the new run
best_model_path = '../runs/train/custom_yolo_n_stable_run/weights/best.pt'
model_trained = YOLO(best_model_path)

# 2. Evaluate the model on the validation set
print("\n--- Starting Evaluation ---")
metrics = model_trained.val(data='../dataset/data.yaml')

# 3. Extract and save metrics to a clean JSON file
import json
import os
output_path = '../results/metrics_stable_run.json'
metrics_dict = metrics.results_dict

# Ensure the '../results' directory exists before writing
os.makedirs('../results', exist_ok=True) 

with open(output_path, "w") as f:
    json.dump(metrics_dict, f, indent=4)

print(f"\nEvaluation complete. Metrics (mAP, precision, recall) saved to: {output_path}")