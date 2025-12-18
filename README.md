---

# Object Detection and Segmentation using YOLOv8

## Project Overview

This project implements a complete **Object Detection and Video Processing Pipeline** using the **Ultralytics YOLOv8** model.

It focuses on understanding the entire machine learning workflow: from environment setup and testing pre-trained models, to **custom dataset training** (Task 2), and finally, **deploying the model to process video** (Task 3).

The purpose of this project is to understand how **YOLO (You Only Look Once)** works for detecting and segmenting multiple objects in real-time images and videos.

---

## What is YOLOv8?

**YOLOv8** (by [Ultralytics](https://github.com/ultralytics/ultralytics)) is one of the latest and most efficient object detection algorithms.
It can perform:

* **Object Detection** — Locating objects and drawing bounding boxes.
* **Segmentation** — Identifying and coloring each object’s exact region.

---

## Setup Instructions

### 1. Create and Activate Virtual Environment

```bash
python -m venv ultralytics_env

```

### 2. Activate the Environment

| OS | Command |
| --- | --- |
| **Windows** | `ultralytics_env\Scripts\activate` |
| **Linux/Mac** | `source ultralytics_env/bin/activate` |

### 3. Install Dependencies

```bash
pip install -r requirements.txt

```

### 4. Install FFmpeg

**FFmpeg** is required for video processing (frame extraction and stitching). Ensure it is installed and added to your system's PATH.

### 5. Verify Installation

```bash
yolo help

```

If you see the help menu, the setup is complete.

---

## Phase II: Custom Model Training & Results Analysis (Task 2)

A custom YOLOv8n model was trained on a small, five-class dataset (car, person, tie, traffic light, truck) to evaluate the end-to-end training process.

### Training Details

| Parameter | Value | Notes |
| --- | --- | --- |
| **Model** | `yolov8n.pt` (nano) | Used for efficiency and quick iteration. |
| **Epochs** | 50 | Initial training length. |
| **Batch Size** | 8 | Maintained stability on resource-limited hardware. |

### Key Results and Performance Summary

| Metric / Class | Value | Conclusion |
| --- | --- | --- |
| **Overall mAP@0.5** | **~0.85** | Strong result, indicating good localization and classification. |
| **mAP@0.5:0.95** | ~0.30 | Acceptable, but suggests difficulties in achieving highly precise bounding boxes. |

---

## Phase III: End-to-End Video Processing Pipeline (Task 3)

The trained model was successfully deployed to process a video clip, validating its end-to-end utility.

### Pipeline Implementation (3 Steps)

| Step | Purpose | Command/Script |
| --- | --- | --- |
| **1. Extraction** | Decompose video into frames (25 FPS). | `ffmpeg -i ./input_video/input.mp4 -q:v 2 -r 25 ./frames_input/frame_%04d.jpg` |
| **2. Inference** | Run custom YOLOv8 model on all frames (using `process_video_pipeline.py`). | `python process_video_pipeline.py` |
| **3. Stitching** | Reassemble annotated frames into the final video. | `ffmpeg -framerate 25 -i ./video_output_run/run_processed_frames/frame_%04d.jpg -c:v libx264 -pix_fmt yuv420p ./final_output/output_video_processed.mp4` |

---

## Phase IV: Specialized Fire Detection Project (Final Task)

This phase involved the implementation of a specialized Fire Detection system. The project utilized a dedicated fire dataset and extended training to optimize the model for real-world safety scenarios.

### Training Strategy

To ensure high performance on CPU hardware, a multi-session training approach was used. The model was trained for 100 epochs, leveraging the YOLO resume capability to maintain learning rate stability across sessions.

| Parameter | Value | Strategy |
| --- | --- | --- |
| **Total Epochs** | 100 | Full convergence achieved over extended training. |
| **Image Size** | 320 | Optimized for CPU processing speed. |
| **Model** | YOLOv8n | Selected for rapid real-time inference. |

### Performance Metrics Analysis

Based on final evaluation charts, the model achieved stable metrics across the fire class.

| Metric | Score | Analysis |
| --- | --- | --- |
| **mAP@0.5** | **0.458** | Solid detection accuracy for a specialized single-class model. |
| **F1 Score** | **0.50** | Achieved at a confidence threshold of 0.310, providing a balance between precision and recall. |
| **Recall** | **0.67** | The model successfully captures the majority of fire instances at lower confidence levels. |

### Visual Validation and Outputs

Validation was performed through various outputs to confirm the model's reliability:

* **Confusion Matrix:** Correctly identifies fire instances with clear distinction from background elements.
* **Precision-Confidence Curve:** Reaches a precision of 1.00 at high confidence (0.883), ensuring minimal false alarms.
* **Inference Results:** Processed videos demonstrate active bounding box tracking on clear, challenging, and distant fire sources.

---

## Tasks Completed (End-to-End)

* Created and activated a virtual environment.
* Installed the `ultralytics` package and tested pre-trained models.
* **Trained a custom YOLOv8n model on a 5-class dataset (Task 2).**
* **Analyzed training progress (Loss curves, mAP) and identified critical performance bottlenecks.**
* **Implemented the full video processing pipeline using FFmpeg and the custom YOLOv8 model (Task 3).**
* **Developed a dedicated Fire Detection system with 100 epochs of training.**
* **Managed complex Git workflows, including merge conflict resolution and specialized .gitignore configurations.**
* **Produced high-fidelity video inference results across multiple fire scenarios.**

---

## Project Structure: Internship (Tasks 1-3)

```css
Internship/
│
├── src/
│   ├── object_detection.py
│   ├── object_segmentation.py
│   ├── train_model.py
│   └── process_video_pipeline.py           
│
├── dataset/
├── runs/
│   └── train/custom_yolo_n_stable_run/weights/best.pt
│
├── frames_input/                          
├── video_output_run/                      
├── final_output/                          
│   └── output_video_processed.mp4
│
├── venv/
├── requirements.txt
└── README.md

```

## Project Structure: Fire-YOLO (Final Task)

```css
fire-yolo/
│
├── src/
│   └── train_fire_detect.py
│
├── models/
│   ├── trained/
│   │   └── best.pt                  <-- Optimized Fire Model
│   └── pretrained/
│       └── yolov8n.pt               <-- Base Model
│
├── data/
│   ├── fire_dataset/                <-- Training Images/Labels
│   └── test_videos/                 <-- Input videos for testing
│
├── outputs/
│   ├── images/                      <-- Confusion Matrix, F1/PR Curves
│   └── videos/                      <-- Final detected video outputs
│
├── runs/                            <-- YOLO training logs and checkpoints
│
├── venv/
├── .gitignore
└── README.md

```

---

## References

* [Ultralytics Official Documentation](https://docs.ultralytics.com/)
* [YOLOv8 GitHub Repository](https://github.com/ultralytics/ultralytics)
* [Python venv Documentation](https://docs.python.org/3/library/venv.html)

## Author

Sruti Goteti
