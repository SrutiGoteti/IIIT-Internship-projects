# Object Detection and Segmentation using YOLOv8

## Project Overview
This project demonstrates **Object Detection** and **Image Segmentation** using the **Ultralytics YOLOv8** model.
It focuses on understanding the entire machine learning workflow, from environment setup and testing pre-trained models to custom dataset training and performance analysis.

The purpose of this project is to understand how **YOLO (You Only Look Once)** works for detecting and segmenting multiple objects in real-time images.

---

## What is YOLOv8?
**YOLOv8** (by [Ultralytics](https://github.com/ultralytics/ultralytics)) is one of the latest and most efficient object detection algorithms.
It can perform:
- **Object Detection** — Locating objects and drawing bounding boxes.
- **Segmentation** — Identifying and coloring each object’s exact region.

---

## Setup Instructions

### 1. Create Virtual Environment
```bash
python -m venv ultralytics_env
## Setup Instructions

### 1. Create Virtual Environment
```bash
python -m venv ultralytics_env
````

### 2\. Activate the Environment (Windows)

```bash
ultralytics_env\Scripts\activate
```

### For Linux

```bash
source ultralytics_env/bin/activate
```

### 3\. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4\. Verify Installation

```bash
yolo help
```

If you see the help menu, then setup is complete.

## Project Structure

```css
vision_project/
│
├── src/
│   ├── object_detection.py
│   ├── object_segmentation.py
│   └── train_model.py          <-- Added for custom training logic
│
├── dataset/                    <-- Custom training data (Images, Labels, data.yaml)
│   ├── train/
│   └── val/
│
├── results/
│   ├── metrics.json            <-- Numerical results (mAP, loss)
│   ├── BoxPR_curve.png         <-- Performance graph
│   └── confusion_matrix_normalized.png  <-- Classification breakdown
│   └── ... (All result images) 
│
├── venv/
│
├── requirements.txt
└── README.md
```

-----

## Phase II: Custom Model Training & Results Analysis (50 Epochs)

A custom YOLOv8n model was trained on a small, five-class dataset (car, person, tie, traffic light, truck) to evaluate the end-to-end training process.

### Training Details

| Parameter | Value | Notes |
| :--- | :--- | :--- |
| **Model** | `yolov8n.pt` (nano) | Used for efficiency and quick iteration. |
| **Epochs** | 50 | Initial training length. |
| **Batch Size** | 8 | Maintained stability on resource-limited hardware. |

### Key Results and Performance Summary

The training successfully converged, as shown by the consistently decreasing loss curves.

| Metric / Class | Value | Conclusion |
| :--- | :--- | :--- |
| **Overall mAP@0.5** | **\~0.85** | Strong result, indicating good localization and classification at a 50% Intersection over Union (IoU) threshold. |
| **mAP@0.5:0.95** | \~0.30 | Acceptable, but suggests difficulties in achieving highly precise bounding boxes. |

### Bottleneck Analysis (The Problem)

The **Confusion Matrix** (available in the `results/` folder) confirms that performance is bottlenecked by the low volume and imbalance of data for specific classes.  For instance, true instances of 'truck' are predicted as 'Background' over 60% of the time.

  * **Truck:** Only predicted correctly 32% of the time. The model predicts **Background** for true truck instances in over 60% of cases. This is a severe **Recall** issue.
  * **Traffic Light:** Low mAP due to a limited and non-diverse set of training examples.

-----

## Current Challenge: Data Imbalance & Next Steps

The immediate priority is to fix the data imbalance before moving to the video processing challenge.

### 1\. Data Augmentation Plan (Phase 1)

The model will be re-trained after increasing the instance count for struggling classes and implementing stronger data augmentation flags:

| Class to Fix | Target Instances (Minimum) | Primary Improvement Goal |
| :--- | :--- | :--- |
| **truck** | **50+** | Fix low Recall against **Background** confusion. |
| **traffic light** | **50+** | Improve overall confidence and mAP. |
| **tie** | **50+** | Stabilize detection of small objects. |

### 2\. Video Processing Challenge (Phase 2)

Once the model's performance is stabilized, the next task is to apply the object detection model to video:

1.  Use `ffmpeg` to extract frames from a public video clip.
2.  Run the trained YOLOv8 model on all extracted images.
3.  Use `ffmpeg` to stitch the processed, annotated images back into a final output video.

-----

## Tasks Completed

  - Created and activated a virtual environment
  - Installed the ultralytics package
  - Tested YOLOv8 for object detection and segmentation on example online images
  - **Trained a custom YOLOv8n model on a 5-class dataset**
  - **Analyzed training progress (Loss curves, mAP)**
  - **Identified critical class-specific performance bottlenecks (Truck/Background confusion)**
  - Pushed project and documentation to GitHub
  - Organized results and created README documentation

-----

## References

  - [Ultralytics Official Documentation](https://docs.ultralytics.com/)
  - [YOLOv8 GitHub Repository](https://github.com/ultralytics/ultralytics)
  - [Python venv Documentation](https://docs.python.org/3/library/venv.html)

## Author

Sruti Goteti
