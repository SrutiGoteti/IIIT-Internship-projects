from ultralytics import YOLO

# 1. Define Paths (ADJUST THESE if your run name was different)
# Path to your trained model weights from Task 2
WEIGHTS_PATH = 'runs/train/custom_yolo_n_stable_run/weights/best.pt' 

# The folder where the unannotated frames are located
SOURCE_FOLDER = 'frames_input'

# The folder where YOLO will save the annotated frames 
OUTPUT_PROJECT = 'video_output_run' 

# 2. Load the Model
print(f"Loading model from: {WEIGHTS_PATH}")
# The model will check for GPU (CUDA) automatically for speed
model = YOLO(WEIGHTS_PATH) 

# 3. Run Prediction on the entire folder
# YOLO is smart enough to detect all images in the folder specified by 'source'
print(f"Starting prediction on all images in: {SOURCE_FOLDER}")
results = model.predict(
    source=SOURCE_FOLDER, 
    save=True,               # Save the images with bounding boxes drawn
    save_conf=True,          # Save the confidence scores on the image
    project=OUTPUT_PROJECT,  # Root output directory 
    name='run_processed_frames', # Subfolder name for this run
    exist_ok=True,           # Don't create new 'run_processed_frames1', 'run_processed_frames2', etc.
    verbose=False,           # Keep the output cleaner while it runs
    conf=0.25                # Only show detections with confidence > 25%
)

# 4. Determine the Output Path
# The annotated frames will be saved inside a new folder created by YOLOv8.
# It is: OUTPUT_PROJECT/name/
FINAL_FRAME_FOLDER = f'{OUTPUT_PROJECT}/run_processed_frames'
print(f"\nProcessing complete. Annotated frames saved in: {FINAL_FRAME_FOLDER}")