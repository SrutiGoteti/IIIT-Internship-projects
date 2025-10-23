# Object Detection and Segmentation using YOLOv8

## Project Overview
This project demonstrates **Object Detection** and **Image Segmentation** using the **Ultralytics YOLOv8** model.  
It loads pretrained YOLOv8 models, performs detection and segmentation on sample images, and displays the results visually.

The purpose of this project is to understand how **YOLO (You Only Look Once)** works for detecting and segmenting multiple objects in real-time images.

---

## What is YOLOv8?
**YOLOv8** (by [Ultralytics](https://github.com/ultralytics/ultralytics)) is one of the latest and most efficient object detection algorithms.  
It can perform:
-  **Object Detection** — Locating objects and drawing bounding boxes.  
-  **Segmentation** — Identifying and coloring each object’s exact region.

---

## Setup Instructions

### 1️. Create Virtual Environment
```bash
python -m venv ultralytics_env
```
### 2️. Activate the Environment(Windows)
```bash
ultralytics_env\Scripts\activate
```
### For Linux
```bash
source ultralytics_env/bin/activate
```
### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
### 4. Verify Intallation
```bash
yolo help
```
If you see the help menu, then setup is complete.

## Project Structure

```css
vision_project/
│
├── src/
│   ├── object_detection.py
│   └── object_segmentation.py
│
├── results/
│   ├── detection_result.jpg
│   └── segmentation_result.jpg
│
├── venv/
│
├── requirements.txt
└── README.md
```
### Code Explanation(object_detection.py)

```python
from ultralytics import YOLO

# Load a pre-trained YOLOv8 model for object detection
model = YOLO('yolov8n.pt')

# Run inference on an example image
results = model('https://ultralytics.com/images/zidane.jpg')

# Display detection results
results[0].show()
```
- Loads a pretrained YOLOv8 model (yolov8n.pt) for detection

- Runs it on an online image of Zidane

- Displays objects with bounding boxes drawn on the image

### Code Explanation(object_segmentation.py)

```python
from ultralytics import YOLO

# Load a pre-trained YOLOv8 segmentation model
model = YOLO('yolov8n-seg.pt')

# Run inference on an image
results = model('https://ultralytics.com/images/bus.jpg')

# Display segmentation results
results[0].show()        # Opens image with segmentation masks overlaid
# OR
results[0].plot()        # Returns an image array for matplotlib display
```
- Loads YOLOv8 segmentation model (yolov8n-seg.pt)

- Performs inference on a sample image (bus)

- Displays segmentation masks overlayed on detected objects

## Results

Screenshots of the detection and segmentation outputs are stored in the results/ folder.

### Tasks Completed

✅ Created and activated a virtual environment
✅ Installed the ultralytics package
✅ Tested YOLOv8 for object detection and segmentation on example online images
✅ Pushed project and documentation to GitHub
✅ Organized results and created README documentation

## References

- [Ultralytics Official Documentation](https://docs.ultralytics.com/)
- [YOLOv8 GitHub Repository](https://github.com/ultralytics/ultralytics)
- [Python venv Documentation](https://docs.python.org/3/library/venv.html)

##  Author

Sruti Goteti






