# Solar Panel Object Detection Project

This project implements an object detection pipeline for detecting solar panels in images. The project covers data exploration, custom metric implementations, and model training/evaluation using a YOLO-based approach. All answers and results are taken directly from the IPython Notebook outputs.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset Details](#dataset-details)
- [Dataset Splitting & Statistics](#dataset-splitting--statistics)
- [Data Exploration Results](#data-exploration-results)
  - [Solar Panel Instance Count](#solar-panel-instance-count)
  - [Label Count Distribution](#label-count-distribution)
  - [Solar Panel Area Statistics](#solar-panel-area-statistics)
  - [Histogram Observations & Outlier Analysis](#histogram-observations--outlier-analysis)
- [Metric Evaluations](#metric-evaluations)
  - [IoU Comparison](#iou-comparison)
  - [Average Precision (AP50) Evaluations](#average-precision-ap50-evaluations)
  - [Precision, Recall, and F1-Score Tables](#precision-recall-and-f1-score-tables)
- [Model Training and Inference](#model-training-and-inference)

---

## Project Overview

The project focuses on detecting solar panels using a deep learning model (YOLO) and evaluating performance with custom implementations for Intersection over Union (IoU) and Average Precision (AP) metrics. The data exploration includes analyzing the dataset distribution, bounding box area calculations (using a Ground Sampling Distance (GSD) of 31 cm per pixel), and outlier analysis.

---

## Dataset Details

- **Total Images:** 2553  
- **Dataset Split:**
  - **Training:** 2042 images (80.0%)
  - **Validation:** 255 images (10.0%)
  - **Testing:** 256 images (10.0%)
- **Label Verification:**
  - Training: 2042 images, 2032 labels  
  - Validation: 255 images, 254 labels  
  - Testing: 256 images, 256 labels

---

## Dataset Splitting & Statistics

- **Total solar panel instances in the dataset:** 29,625

---

## Data Exploration Results

### Solar Panel Instance Count

- **Total Instances:** 29,625

### Label Count Distribution

The distribution of labels per image is as follows:

- 11 images have 0 labels  
- 81 images have 1 label  
- 167 images have 2 labels  
- 221 images have 3 labels  
- 218 images have 4 labels  
- 217 images have 5 labels  
- 189 images have 6 labels  
- 170 images have 7 labels  
- 184 images have 8 labels  
- 169 images have 9 labels  
- 121 images have 10 labels  
- 97 images have 11 labels  
- 84 images have 12 labels  
- 69 images have 13 labels  
- 49 images have 14 labels  
- 46 images have 15 labels  
- 41 images have 16 labels  
- 36 images have 17 labels  
- 25 images have 18 labels  
- 29 images have 19 labels  
- 14 images have 20 labels  
- 4 images have 21 labels  
- 1 image has 22 labels  
- 4 images have 23 labels  
- 2 images have 24 labels  
- 4 images have 25 labels  
- 3 images have 26 labels  
- 5 images have 27 labels  
- 5 images have 28 labels  
- 15 images have 29 labels  
- 20 images have 30 labels  
- 8 images have 31 labels  
- 7 images have 32 labels  
- 13 images have 33 labels  
- 19 images have 34 labels  
- 10 images have 35 labels  
- 6 images have 36 labels  
- 17 images have 37 labels  
- 13 images have 38 labels  
- 6 images have 39 labels  
- 9 images have 40 labels  
- 10 images have 41 labels  
- 12 images have 42 labels  
- 11 images have 43 labels  
- 4 images have 44 labels  
- 2 images have 45 labels  
- 5 images have 46 labels  
- 9 images have 47 labels  
- 3 images have 48 labels  
- 5 images have 49 labels  
- 6 images have 50 labels  
- 9 images have 51 labels  
- 16 images have 52 labels  
- 4 images have 53 labels  
- 6 images have 54 labels  
- 1 image has 55 labels  
- 1 image has 56 labels  
- 3 images have 58 labels  
- 2 images have 59 labels  
- 2 images have 60 labels  
- 1 image has 61 labels  
- 6 images have 62 labels  
- 3 images have 63 labels  
- 1 image has 64 labels  
- 3 images have 65 labels  
- 4 images have 66 labels  
- 1 image has 67 labels  
- 1 image has 71 labels  
- 1 image has 72 labels  
- 1 image has 73 labels  
- 5 images have 74 labels  
- 1 image has 75 labels  
- 2 images have 76 labels  
- 2 images have 77 labels  
- 1 image has 78 labels  

### Solar Panel Area Statistics

Calculated using bounding box dimensions and a GSD of 31 cm per pixel:

- **Method:** Bounding box dimensions with a GSD of 31 cm per pixel  
- **Mean area:** 191.52 m²  
- **Median area:** 91.68 m²  
- **Standard deviation:** 630.70 m²  
- **Minimum area:** 1.06 m²  
- **Maximum area:** 12177.41 m²

### Histogram Observations & Outlier Analysis

- **Histogram Observations:**
  - The area distribution is multimodal with 24 distinct peaks.
  - The distribution is positively skewed (skewness = 12.09); the mean (191.52 m²) is greater than the median (91.68 m²).
  - This suggests a predominance of smaller solar panels, with fewer larger panels increasing the mean.
  - Distinct groups appear centered around areas such as: 74.19 m², 2109.67 m², 2402.20 m², 2987.25 m², 3206.64 m², 3645.43 m², 4254.86 m², 4449.87 m², 5193.37 m², 5571.22 m², 5888.12 m², 6326.91 m², 7046.03 m², 7326.37 m², 7813.91 m², 8216.13 m², 8654.92 m², 9361.86 m², 9556.87 m², 10190.68 m², 10507.58 m², 10995.12 m², 11397.34 m², 11689.87 m².
- **Outlier Analysis (using the IQR method):**
  - **Outliers detected:** 2635 instances (8.89% of all instances)
  - **Outlier boundaries:** < -139.44 m² or > 339.52 m²  
    *(Note: Negative values arise from the IQR calculation method; practically, only values > 339.52 m² are considered outliers.)*
  - **Minimum outlier:** 340.00 m²  
  - **Maximum outlier:** 12177.41 m²

---

## Metric Evaluations

### IoU Comparison

- **IoU using Shapely:** 0.3913  
- **IoU using supervision:** 0.3913  
- **Difference:** 0.000000

### Average Precision (AP50) Evaluations

For 10 randomly generated test images (100×100) with 10 boxes each (ground truth and predicted boxes of size 20×20, single class):

- **AP50 (Pascal VOC 11-point):** 0.4718  
- **AP50 (COCO 101-point):** 0.4623  
- **AP50 (AUC):** 0.5538

### Precision, Recall, and F1-Score Tables

#### Precision Table

|      IoU \ Conf      | 0.1   | 0.3   | 0.5   | 0.7   | 0.9   |
|----------------------|-------|-------|-------|-------|-------|
| **IoU = 0.1**        | 0.9442| 0.9534| 0.9834| 0.9945| 1.0000|
| **IoU = 0.3**        | 0.9431| 0.9520| 0.9825| 0.9942| 1.0000|
| **IoU = 0.5**        | 0.9392| 0.9483| 0.9789| 0.9923| 1.0000|
| **IoU = 0.7**        | 0.9203| 0.9303| 0.9647| 0.9855| 1.0000|
| **IoU = 0.9**        | 0.6183| 0.6257| 0.6576| 0.6988| 0.9540|

#### Recall Table

|      IoU \ Conf      | 0.1   | 0.3   | 0.5   | 0.7   | 0.9   |
|----------------------|-------|-------|-------|-------|-------|
| **IoU = 0.1**        | 0.9724| 0.9699| 0.9506| 0.8883| 0.2931|
| **IoU = 0.3**        | 0.9713| 0.9684| 0.9498| 0.8880| 0.2931|
| **IoU = 0.5**        | 0.9673| 0.9647| 0.9463| 0.8863| 0.2931|
| **IoU = 0.7**        | 0.9477| 0.9463| 0.9325| 0.8803| 0.2931|
| **IoU = 0.9**        | 0.6368| 0.6365| 0.6357| 0.6242| 0.2796|

#### F1 Score Table

|      IoU \ Conf      | 0.1   | 0.3   | 0.5   | 0.7   | 0.9   |
|----------------------|-------|-------|-------|-------|-------|
| **IoU = 0.1**        | 0.9581| 0.9616| 0.9667| 0.9384| 0.4534|
| **IoU = 0.3**        | 0.9570| 0.9601| 0.9658| 0.9381| 0.4534|
| **IoU = 0.5**        | 0.9530| 0.9564| 0.9623| 0.9363| 0.4534|
| **IoU = 0.7**        | 0.9338| 0.9382| 0.9483| 0.9299| 0.4534|
| **IoU = 0.9**        | 0.6274| 0.6311| 0.6464| 0.6594| 0.4325|

---

## Model Training and Inference

- **Training Summary:**
  - Pre-trained YOLO model loaded from `runs/detect/train2/weights/best.pt`
  - Inference performed on 256 test images (all with `.tif` extension).
  - Example inference timings (per image):
    - Preprocess: ~2.9–10.3 ms
    - Inference: ~7.3–15.5 ms
    - Postprocess: ~1.2–2.3 ms

- **Inference Example Outputs:**
  - Images processed at 640×640 resolution with varying numbers of detected solar panels (e.g., 9, 13, 17, etc.) along with detailed speed statistics.

