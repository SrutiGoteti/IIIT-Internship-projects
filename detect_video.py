from ultralytics import YOLO
import os

# --- STEP 1: PATHS (Change these!) ---
# Point to your gold-medal model
model_path = r"C:\fire-yolo\models\trained\best.pt"

# Point to your test video (make sure the name matches your file!)
video_source = r"C:\fire-yolo\data\test_videos\small-and-distant-fire2.mp4"

# Where the finished video will go
output_dir = r"C:\fire-yolo\outputs\video"

# --- STEP 2: LOAD & RUN ---
model = YOLO(model_path)

print("Starting video processing... this may take a few minutes on CPU.")

results = model.predict(
    source=video_source,
    conf=0.25,           # START HERE. If no boxes appear, try 0.15
    save=True,           # This creates the output video
    imgsz=320,           # Fast mode for CPU
    project=output_dir,  # Saves in outputs/video
    name="final_demo"    # Creates a sub-folder called 'final_demo'
)

print(f"Success! Your detected video is here: {output_dir}/final_demo")