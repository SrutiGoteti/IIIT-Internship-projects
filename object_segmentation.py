from ultralytics import YOLO

# Load a pre-trained YOLOv8 segmentation model
model = YOLO('yolov8n-seg.pt')

# Run inference on an image
results = model('https://ultralytics.com/images/bus.jpg')

# Display results
results[0].show()        # opens the image with segmentation masks overlaid
# OR
results[0].plot()        # returns a numpy array with the results
