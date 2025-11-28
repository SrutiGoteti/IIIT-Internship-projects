````markdown
# Object Detection and Segmentation using YOLOv8

## Project Overview
This project implements a complete **Object Detection and Video Processing Pipeline** using the **Ultralytics YOLOv8** model.

It focuses on understanding the entire machine learning workflow: from environment setup and testing pre-trained models, to **custom dataset training** (Task 2), and finally, **deploying the model to process video** (Task 3).

The purpose of this project is to understand how **YOLO (You Only Look Once)** works for detecting and segmenting multiple objects in real-time images and videos.

---

## What is YOLOv8?
**YOLOv8** (by [Ultralytics](https://github.com/ultralytics/ultralytics)) is one of the latest and most efficient object detection algorithms.
It can perform:
- **Object Detection** — Locating objects and drawing bounding boxes.
- **Segmentation** — Identifying and coloring each object’s exact region.

---

## Setup Instructions

### 1. Create and Activate Virtual Environment
```bash
python -m venv ultralytics_env
````

### 2\. Activate the Environment

| OS | Command |
| :--- | :--- |
| **Windows** | `ultralytics_env\Scripts\activate` |
| **Linux/Mac** | `source ultralytics_env/bin/activate` |

### 3\. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4\. Install FFmpeg

**FFmpeg** is required for video processing (frame extraction and stitching). Ensure it is installed and added to your system's PATH.

### 5\. Verify Installation

```bash
yolo help
```

If you see the help menu, the setup is complete.

-----

## Phase II: Custom Model Training & Results Analysis (Task 2)

A custom YOLOv8n model was trained on a small, five-class dataset (car, person, tie, traffic light, truck) to evaluate the end-to-end training process.

### Training Details

| Parameter | Value | Notes |
| :--- | :--- | :--- |
| **Model** | `yolov8n.pt` (nano) | Used for efficiency and quick iteration. |
| **Epochs** | 50 | Initial training length. |
| **Batch Size** | 8 | Maintained stability on resource-limited hardware. |

### Key Results and Performance Summary

| Metric / Class | Value | Conclusion |
| :--- | :--- | :--- |
| **Overall mAP@0.5** | **\~0.85** | Strong result, indicating good localization and classification. |
| **mAP@0.5:0.95** | \~0.30 | Acceptable, but suggests difficulties in achieving highly precise bounding boxes. |

### Bottleneck Analysis (Training Problem)

The **Confusion Matrix** confirms that performance is bottlenecked by the low volume and imbalance of data for specific classes. For instance, true instances of 'truck' are predicted as 'Background' in over 60% of cases (a severe **Recall** issue).

-----

## Phase III: End-to-End Video Processing Pipeline (Task 3)

The trained model was successfully deployed to process a video clip, validating its end-to-end utility.

### Pipeline Implementation (3 Steps)

| Step | Purpose | Command/Script |
| :--- | :--- | :--- |
| **1. Extraction** | Decompose video into frames (25 FPS). | `ffmpeg -i ./input_video/input.mp4 -q:v 2 -r 25 ./frames_input/frame_%04d.jpg` |
| **2. Inference** | Run custom YOLOv8 model on all frames (using `process_video_pipeline.py`). | `python process_video_pipeline.py` |
| **3. Stitching** | Reassemble annotated frames into the final video. | `ffmpeg -framerate 25 -i ./video_output_run/run_processed_frames/frame_%04d.jpg -c:v libx264 -pix_fmt yuv420p ./final_output/output_video_processed.mp4` |

-----

## 🚧 Current Limitations and Future Work

The video processing confirmed a major flaw from the training analysis, which is the current focus for iteration.

### 1\. Critical Limitation: Persistent False Positives (Truck Misclassification)

The training bottleneck (low 'truck' Recall) manifested as a False Positive problem in the video:

  * **Observed Issue:** Sections of the static background in the video (e.g., road signs, overhead structures, and non-vehicle shapes) are consistently misclassified as **'truck'** with moderate confidence scores.
  * **Root Cause:** The model failed to generalize the 'truck' class well due to insufficient and non-diverse training data, causing it to confuse truck features with complex background elements.

### 2\. Immediate Next Steps (Iteration Plan)

| Goal | Action | Primary Improvement Goal |
| :--- | :--- | :--- |
| **Data Fix** | Increase instance count for **truck**, **traffic light**, and **tie** classes (Target: 50+ instances each). | **Immediate:** Resolve 'truck' confusion with Background (False Positives in video). |
| **Post-Processing**| Implement a script to filter out low-confidence, stationary truck detections. | Mitigate persistent false positives without retraining. |
| **Architecture** | Retrain using the larger **YOLOv8s** (Small) architecture. | Evaluate the trade-off between increased accuracy (better mAP) and inference speed. |

-----

## ✅ Tasks Completed (End-to-End)

  - Created and activated a virtual environment.
  - Installed the `ultralytics` package and tested pre-trained models.
  - **Trained a custom YOLOv8n model on a 5-class dataset (Task 2).**
  - **Analyzed training progress (Loss curves, mAP) and identified critical performance bottlenecks.**
  - **Implemented the full video processing pipeline using FFmpeg and the custom YOLOv8 model (Task 3).**
  - Pushed project and documentation to GitHub.

-----

## Project Structure

```css
vision_project/
│
├── src/
│   ├── object_detection.py
│   ├── object_segmentation.py
│   └── train_model.py
│   └── process_video_pipeline.py           
│
├── dataset/
├── runs/
│   └── train/custom_yolo_n_stable_run/weights/best.pt
│
├── frames_input/                          
├── video_output_run/                      
├── final_output/                          
│   └── output_video_processed.mp4
│
├── venv/
├── requirements.txt
└── README.md
```

-----

## References

  - [Ultralytics Official Documentation](https://docs.ultralytics.com/)
  - [YOLOv8 GitHub Repository](https://github.com/ultralytics/ultralytics)
  - [Python venv Documentation](https://docs.python.org/3/library/venv.html)

## Author

Sruti Goteti
