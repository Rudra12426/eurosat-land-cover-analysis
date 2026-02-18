# EuroSAT Land Cover Analysis

## 📌 Overview
This project performs automated satellite image analysis using HSV-based color segmentation.  
It estimates percentage coverage of Water, Vegetation, and Urban regions from satellite images.

## 🚀 Features
- Automated batch image processing
- HSV-based land cover segmentation
- Mask generation for each class
- Percentage coverage calculation
- CSV export of results

## 🛰 Dataset
Place satellite images (.jpg) inside the `dataset/` folder before running.

## 🛠 Tech Stack
- Python
- OpenCV
- NumPy
- Pandas

## ▶️ How to Run

1. Install dependencies:
   pip install -r requirements.txt

2. Run:
   python main.py

Results will be saved inside the `output/` folder.
