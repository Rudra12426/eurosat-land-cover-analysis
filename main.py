# use any satellite image dataset....
"""
EuroSAT Land Cover Analysis
Author: Rudra 
Description:
Performs automated satellite image analysis using HSV-based
color segmentation to estimate water, vegetation, and urban coverage.
"""

import cv2
import os
import numpy as np
import pandas as pd


# ----------------------------
# Dataset & Output Paths
# ----------------------------
DATASET_PATH = "dataset" # use the path where dataset exist.
OUTPUT_PATH = "output"   # where outputs save of this project..

os.makedirs(OUTPUT_PATH, exist_ok=True)

MASK_FOLDERS = ["Water", "Vegetation", "Urban"]
for folder in MASK_FOLDERS:
    os.makedirs(os.path.join(OUTPUT_PATH, folder), exist_ok=True)


# ----------------------------
# Get Image Files
# ----------------------------
def get_image_files(path):
    return [f for f in os.listdir(path) if f.lower().endswith(".jpg")]


# ----------------------------
# Preprocess Image
# ----------------------------
def preprocess_image(img, size=(256, 256)):
    img_resized = cv2.resize(img, size)
    img_hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    return img_resized, img_hsv


# ----------------------------
# Color Segmentation
# ----------------------------
def segment_colors(img_hsv):
    lower_blue = np.array([85, 50, 50])
    upper_blue = np.array([135, 255, 255])
    mask_water = cv2.inRange(img_hsv, lower_blue, upper_blue)

    lower_green = np.array([25, 40, 40])
    upper_green = np.array([95, 255, 255])
    mask_veg = cv2.inRange(img_hsv, lower_green, upper_green)

    lower_urban = np.array([0, 0, 50])
    upper_urban = np.array([180, 60, 200])
    mask_urban = cv2.inRange(img_hsv, lower_urban, upper_urban)

    return mask_water, mask_veg, mask_urban


# ----------------------------
# Calculate Coverage %
# ----------------------------
def calculate_coverage(mask):
    total_pixels = mask.size
    white_pixels = cv2.countNonZero(mask)
    return round((white_pixels / total_pixels) * 100, 2)


# ----------------------------
# Main Execution
# ----------------------------
if __name__ == "__main__":

    if not os.path.exists(DATASET_PATH):
        print("Dataset folder not found!")
        exit()

    image_files = get_image_files(DATASET_PATH)

    if not image_files:
        print("No images found in dataset folder!")
        exit()

    print(f"Processing {len(image_files)} images...")

    results = []

    for idx, file_name in enumerate(image_files):
        img_path = os.path.join(DATASET_PATH, file_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        img_resized, img_hsv = preprocess_image(img)
        masks = segment_colors(img_hsv)

        mask_water, mask_veg, mask_urban = masks

        cv2.imwrite(os.path.join(OUTPUT_PATH, "Water", file_name), mask_water)
        cv2.imwrite(os.path.join(OUTPUT_PATH, "Vegetation", file_name), mask_veg)
        cv2.imwrite(os.path.join(OUTPUT_PATH, "Urban", file_name), mask_urban)

        results.append({
            "Image": file_name,
            "Water_%": calculate_coverage(mask_water),
            "Vegetation_%": calculate_coverage(mask_veg),
            "Urban_%": calculate_coverage(mask_urban)
        })

        if (idx + 1) % 50 == 0:
            print(f"Processed {idx + 1} images...")

    df = pd.DataFrame(results)
    csv_path = os.path.join(OUTPUT_PATH, "EuroSAT_Analysis.csv")
    df.to_csv(csv_path, index=False)

    print("Analysis Complete!")
    print(f"Results saved at: {csv_path}")
