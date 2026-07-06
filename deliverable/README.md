# Butterfly Scale Pedicel Detection

## Overview

This project focuses on analyzing a large dataset of butterfly scale microscopy images to automatically identify the **pedicel** (the point where the scale attaches to the butterfly wing) in each image.

Identifying the pedicel enables all scales to be aligned to a common orientation, providing a standardized representation of each scale. This standardization is an important preprocessing step for downstream computer vision, morphometric, and statistical analyses, allowing meaningful comparisons across thousands of butterfly scales.

---

## Project Workflow

The complete workflow is divided into three main stages.

### 1. Butterfly Scale Segmentation and Contour Extraction

The notebook

```
ERISE_scale_shape_extraction.ipynb
```

developed by **Dr. Remi Mégret**, processes the raw microscopy images and extracts:

- Binary masks
- Scale contours

These contours serve as the geometric representation of each butterfly scale used throughout the remainder of the project.

PLEASE REFER TO ORIGINAL WORK WHEN UTILIZING THE SCRIPT

---

### 2. Pedicel Annotation

The project also includes the annotation GUI developed by **Dr. Remi Mégret**.

Using this graphical interface, the extracted contours are manually annotated by selecting the pedicel location on each scale.

The output of this stage is a dataset containing:

- Scale contour coordinates
- Manually annotated pedicel coordinates

This annotated dataset serves as the ground truth for training and evaluating machine learning models.

PLEASE REFER TO ORIGINAL WORK WHEN UTILIZING THE SCRIPT

---

### 3. Machine Learning Pipeline

The notebook developed as part of this project takes the annotated contour dataset and performs the following steps:

1. Load and clean the original contour dataset.
2. Convert contour coordinate lists into numerical contour arrays.
3. Detect biologically meaningful pivot points, including:
   - Stem neighbor points
   - High-curvature points
4. Generate augmented training samples centered around these pivot points.
5. Normalize contours through translation and rotation.
6. Convert normalized contours into Fourier descriptors.
7. Build a machine learning feature matrix.
8. Train and evaluate a baseline classifier for pedicel identification.

---

## Repository Structure

```
.
├── ERISE_scale_shape_extraction.ipynb     # Scale segmentation and contour extraction (Dr. Remi Mégret)
├── GUI/                                   # Annotation interface (Dr. Remi Mégret)
├── pedicel_classification.ipynb           # Machine learning pipeline
├── requirements.txt
└── README.md
```
## Original data located in:
```
/mnt/data/users/jsoto/sample_img'
```
---

## Requirements

Install the required Python packages with:

```bash
pip install -r requirements.txt
```

---

## Expected Pipeline

```
Raw microscopy images
        │
        ▼
ERISE_scale_shape_extraction.ipynb
        │
        ▼
Scale masks + contours
        │
        ▼
Annotation GUI
        │
        ▼
Annotated contour dataset
        │
        ▼
Pedicel_Classification.ipynb
        │
        ├── Data cleaning
        ├── Pivot detection
        ├── Data augmentation
        ├── Contour normalization
        ├── Fourier descriptors
        ├── Feature extraction
        └── Machine learning classifier
```

---

## Acknowledgments

This project builds upon software developed by **Dr. Remi Mégret**, including:

- **ERISE_scale_shape_extraction.ipynb** for butterfly scale segmentation and contour extraction.
- The **annotation GUI** for manual pedicel labeling.

The machine learning pipeline, contour preprocessing, feature extraction, and classification framework were developed as part of this project.
