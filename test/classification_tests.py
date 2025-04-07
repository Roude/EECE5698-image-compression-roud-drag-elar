import os
import unittest
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import pillow_avif
import pandas as pd
import numpy as np
from typing import Callable
from datetime import datetime as dt
import warnings

from src.compression import BaselineJpeg

results_name = f"iou_results_{dt.now():%Y-%m-%d_%H-%M-%S}.csv"

def preprocess_image(image_path: str):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

def classify_image(model, image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    top_classes = torch.topk(probabilities, k=5).indices.tolist()
    return set(top_classes)

def intersection_over_union(set1, set2):
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0

def compressed_iou(image_file, compress_function):
    warnings.simplefilter('ignore')
    model = models.resnet50(pretrained=True)
    model.eval()

    # Process and classify original image
    original_tensor = preprocess_image(image_file)
    original_classes = classify_image(model, original_tensor)

    # Compress image and classify compressed version
    compressed_image_file = compress_function(image_file)
    compressed_tensor = preprocess_image(compressed_image_file)
    compressed_classes = classify_image(model, compressed_tensor)

    # Compute Intersection over Union
    iou = intersection_over_union(original_classes, compressed_classes)
    return iou

def df_add_row(df, row):
    df = pd.concat([df, pd.DataFrame([row], columns=df.columns)], ignore_index=True)
    return df

def clean_temp_results():
    for filename in os.listdir("tmp"):
        file_path = os.path.join("tmp", filename)
        if os.path.isfile(file_path):
            os.remove(file_path)

class TestImageCompression(unittest.TestCase):
    def __init__(self, methodName="runTest", image_file=None, settings_file=None, compress_function=None):
        super().__init__(methodName)
        self.image_file = image_file
        self.settings_file = settings_file
        self.compress_function = compress_function

    def test_jpeg_baseline_compression(self, settings_filepath=None):
        clean_temp_results()
        if not settings_filepath:
            settings_filepath = "../compression_configurations/baseline_jpeg_q100.yaml"
        results_name = f"iou_results_{os.path.basename(settings_filepath)}_{dt.now():%Y-%m-%d_%H-%M-%S}.csv"
        baseline_jpeg_compression_engine = BaselineJpeg(settings_filepath)
        results_df = pd.DataFrame(columns=["Image Name", "Compression Name", "Intersection over Union"])

        for root, dirs, files in os.walk("../assets/test_images"):
            for file in files:
                test_image_path = os.path.join(root, file)
                iou = compressed_iou(test_image_path, baseline_jpeg_compression_engine)
                results_df = df_add_row(results_df, [file, os.path.basename(settings_filepath), iou])
        results_df.to_csv(os.path.join(os.getcwd(),"results",results_name))

    def test_quality_factor_sweep_jpeg_compression(self):
        settings_path_list = ["../compression_configurations/baseline_jpeg_q100.yaml",
                              "../compression_configurations/baseline_jpeg_q80.yaml",
                              "../compression_configurations/baseline_jpeg_q60.yaml",
                              "../compression_configurations/baseline_jpeg_q40.yaml",
                              "../compression_configurations/baseline_jpeg_q20.yaml"]
        for settings_filepath in settings_path_list:
            self.test_jpeg_baseline_compression(settings_filepath=settings_filepath)

if __name__ == "__main__":
    unittest.main()