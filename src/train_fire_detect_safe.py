from ultralytics import YOLO

# 1. Load your very best weights from your last run
model = YOLO(r"C:\fire-yolo\runs\fire_detect_finetune_2\weights\best.pt")

# 2. Start the long-term run
model.train(
    data=r"C:\fire-yolo\data\raw\fire_dataset\data.yaml",
    epochs=100,         # Setting the target high so we don't have to restart
    imgsz=320,          # Dropped to 320 for a SIGNIFICANT speed boost on CPU
    batch=8,
    project=r"C:\fire-yolo\runs",
    name="fire_final_run",
    plots=True,
    exist_ok=True       # Allows overwriting the same folder if needed
)