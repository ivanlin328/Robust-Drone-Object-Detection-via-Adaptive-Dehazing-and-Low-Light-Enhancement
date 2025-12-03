# Robust Drone Object Detection via Adaptive Dehazing and Low-Light Enhancement

This repository contains the implementation and evaluation of various image enhancement algorithms designed to improve object detection performance on drone imagery under adverse weather conditions (Low-Light and Fog).

## 📌 Overview

Drone object detection faces significant challenges in uncontrolled environments, particularly in **low-light** and **foggy** conditions. This project explores whether improving visual image quality (measured by NIQE/BRISQUE) directly correlates with improved object detection performance.

We evaluate two categories of enhancement:
1.  **Low-Light Enhancement:** CLAHE, Zero-DCE++
2.  **Dehazing:** DCP (Dark Channel Prior), DehazeFormer

## 📊 Datasets used

* **Training/Validation:** VisDrone Dataset
* **Low-Light Scenarios:** ExDark Dataset (mixed with VisDrone for fine-tuning)
* **Foggy Scenarios:** RTTS Dataset(mixed with VisDrone for fine-tuning)

## 🛠️ Methods Evaluated

### 1. Low-Light Enhancement
* **Original:** Baseline raw imagery.
* **CLAHE:** Contrast Limited Adaptive Histogram Equalization.
* **Zero-DCE++:** A deep learning-based curve estimation method for dynamic range adjustment.

### 2. Dehazing
* **Original:** Baseline foggy imagery.
* **DCP:** Dark Channel Prior (Traditional method).
* **DehazeFormer:** Transformer-based image dehazing network.

## 📈 Experimental Results

We evaluated the performance using **Image Quality Metrics** (NIQE, BRISQUE) and **Detection Metrics** (Precision, Recall, mAP@50).

### 1. Image Quality Assessment
Lower scores indicate better perceptual quality.

| Scenario | Method | Average NIQE (↓) | Average BRISQUE (↓) |
| :--- | :--- | :--- | :--- |
| **Low Light** | Original | ~6.42 | ~36.00 |
| | **CLAHE** | **~4.40** | **~16.20** |
| | Zero-DCE++ | ~4.20 | ~10.00 |
| **Fog** | Original | ~6.70 | ~31.00 |
| | DCP | ~5.50 | ~30.00 |
| | **DehazeFormer** | **~4.80** | **~6.50** |

### 2. Detection Performance (Fine-Tuned Models)

#### 🌙 Low-Light Scenarios
Contrary to expectation, visual enhancement did not always yield higher mAP, though Zero-DCE++ improved precision in some cases.

| Method | Precision (All) | Recall (All) | mAP@50 (All) | mAP@50 (Car) | mAP@50 (Person) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Original** | **0.815** | 0.581 | **0.730** | 0.661 | 0.798 |
| CLAHE | 0.696 | **0.597** | 0.658 | 0.625 | 0.692 |
| Zero-DCE++ | 0.738 | 0.560 | 0.668 | **0.714** | 0.622 |

> **Observation:** While Zero-DCE++ and CLAHE significantly improved visual quality scores (NIQE/BRISQUE), the Original low-light images often retained the best Feature consistency for the detector, resulting in the highest overall mAP@50.

#### 🌫️ Foggy Scenarios
DehazeFormer showed significant improvements in detection recall and mAP compared to the baseline and traditional DCP.

| Method | Precision (All) | Recall (All) | mAP@50 (All) | mAP@50 (Car) | mAP@50 (Person) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Original | **0.944** | 0.384 | 0.660 | 0.599 | 0.722 |
| DCP | 0.762 | 0.273 | 0.533 | 0.584 | 0.482 |
| **DehazeFormer**| 0.929 | **0.647** | **0.781** | **0.617** | **0.944** |

> **Observation:** DehazeFormer significantly outperformed the original foggy images, particularly in **Recall (0.647 vs 0.384)**, making it highly effective for drone safety applications where missing an object is critical.

## 📝 Conclusion
Fog: Deep learning-based dehazing (DehazeFormer) is highly recommended. It drastically improves object recovery (Recall) and overall accuracy (mAP).

Low-Light: Traditional metrics (NIQE) do not perfectly correlate with detection performance. While Zero-DCE++ makes images look better to humans, the detector (YOLO) often performs best on the Original raw data or requires significantly more fine-tuning on enhanced data to match the baseline.

