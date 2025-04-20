import os
import unittest
import torch
import torchvision.transforms as transforms
import yaml
from torchvision import models
from PIL import Image
import pillow_avif
import pandas as pd
import numpy as np
from datetime import datetime as dt
import warnings

from src.compression import BaselineJpeg, FlexibleJpeg
from src.decompression import FlexibleJpegDecompress, DecompressImage

results_name = f"iou_results_{dt.now():%Y-%m-%d_%H-%M-%S}.csv"

def preprocess_image(image_path: str):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = Image.open(image_path).convert("RGB")
    return transform(image).unsqueeze(0)

def classify_image(model, image_tensor, k=5):
    with torch.no_grad():
        outputs = model(image_tensor)
    probs = torch.nn.functional.softmax(outputs, dim=1)
    return probs.squeeze()

def intersection_over_union(set1, set2):
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0

def compare_topk_to_uncompressed_reference_flexible(uncompressed_img,
                                                    compression_engine,
                                                    decompression_engine,
                                                    topk: int = 5):
    """
    Args:
        uncompressed_img: PIL Image (original)
        compressed_img: PIL Image (compressed version)
        topk: Top-k value for comparison

    Returns:
        Dictionary with match result and predictions
    """

    warnings.simplefilter('ignore')

    def get_probs(img):
        model = models.resnet50(pretrained=True)
        tensor = preprocess_image(img)
        return classify_image(model, tensor)

        input_tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        return probs.squeeze()

    # Get probabilities for both images
    uncompressed_probs = get_probs(uncompressed_img)

    compressed_img_location = compression_engine(uncompressed_img)
    compressed_probs = get_probs(decompression_engine(compressed_img_location)[1])

    # Get Top-1 prediction from uncompressed image
    top1_idx = torch.argmax(uncompressed_probs).item()
    uncompressed_prob = uncompressed_probs[top1_idx].item()
    compressed_prob = compressed_probs[top1_idx].item()
    delta = abs(compressed_prob/uncompressed_prob - 1)

    return {
        "top1_class_index": top1_idx,
        "uncompressed_prob": uncompressed_prob,
        "compressed_prob": compressed_prob,
        "confidence_delta": delta
    }

def compare_topk_to_uncompressed_reference_jpeg(uncompressed_img,
                                           compression_engine,
                                           topk: int = 5):
    """
    Args:
        uncompressed_img: PIL Image (original)
        compressed_img: PIL Image (compressed version)
        topk: Top-k value for comparison

    Returns:
        Dictionary with match result and predictions
    """

    warnings.simplefilter('ignore')

    def get_probs(img):
        model = models.resnet50(pretrained=True)
        tensor = preprocess_image(img)
        return classify_image(model, tensor)

        input_tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        return probs.squeeze()

    # Get probabilities for both images
    uncompressed_probs = get_probs(uncompressed_img)
    compressed_probs = get_probs(compression_engine(uncompressed_img))

    # Get Top-1 prediction from uncompressed image
    top1_idx = torch.argmax(uncompressed_probs).item()
    uncompressed_prob = uncompressed_probs[top1_idx].item()
    compressed_prob = compressed_probs[top1_idx].item()
    delta = abs(compressed_prob/uncompressed_prob - 1)

    return {
        "top1_class_index": top1_idx,
        "uncompressed_prob": uncompressed_prob,
        "compressed_prob": compressed_prob,
        "confidence_delta": delta
    }


def compressed_classification_accuracy(image_file, compress_function):
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
    # iou = intersection_over_union(original_classes, compressed_classes)
    return original_classes, compressed_classes

def df_add_row(df, row):
    df = pd.concat([df, pd.DataFrame([row], columns=df.columns)], ignore_index=True)
    return df

def clean_temp_results():
    for filename in os.listdir("tmp"):
        file_path = os.path.join("tmp", filename)
        if os.path.isfile(file_path):
            os.remove(file_path)


def test_homemade_compression(self,
                              settings_filepath=None,
                              results_filename=None,
                              test_image_dir="../assets/test_images/short test"):
    clean_temp_results()
    if not settings_filepath:
        settings_filepath = "../compression_configurations/homemade_compression_gauss.yaml"
    results_name = f"{results_filename}_{os.path.basename(settings_filepath)}_{dt.now():%Y-%m-%d_%H-%M-%S}.csv"
    compression_engine = FlexibleJpeg(settings_filepath)
    decompresion_engine = FlexibleJpegDecompress()
    results_df = pd.DataFrame(
        columns=["Image Name", "Compression Name", "Uncompressed Reference Label", "Reference Probability",
                 "Compressed Probabilty of Reference Label", "Confidence Drift"])

    for root, dirs, files in os.walk(test_image_dir):
        for file in files:
            test_image_path = os.path.join(root, file)
            topk_result = compare_topk_to_uncompressed_reference_flexible(test_image_path,
                                                                          compression_engine,
                                                                          decompresion_engine)

            results_df = df_add_row(results_df, [file,
                                                 os.path.basename(settings_filepath),
                                                 topk_result["top1_class_index"],
                                                 topk_result["uncompressed_prob"],
                                                 topk_result["compressed_prob"],
                                                 topk_result["confidence_delta"]])

    results_df.to_csv(os.path.join(os.getcwd(), "results", results_name))

class TestImageCompression(unittest.TestCase):
    def __init__(self, methodName="runTest", image_file=None, settings_file=None, compress_function=None):
        super().__init__(methodName)
        self.image_file = image_file
        self.settings_file = settings_file
        self.compress_function = compress_function

    def test_sml_images_jpeg_like(self):
        clean_temp_results()
        settings_filepath = "../compression_configurations/homemade_compression_jpeg_like.yaml"
        with open(settings_filepath,'r') as settings_file:
            settings = yaml.safe_load(settings_file)

        compression_engine = FlexibleJpeg(settings_filepath)
        decompression_engine = FlexibleJpegDecompress()
        for root, dirs, imgs in os.walk("../assets/unit_test_images"):
            for img in imgs:
                test_image_path = os.path.join(root, img)
                compressed_img_filepath = f"tmp/{img.split(".")[0]}"
                # compression_engine.save_location = compressed_img_filepath
                compression_engine(test_image_path, settings, save_location=compressed_img_filepath)
                decompression_engine.save_location = f"tmp/{img.split(".")[0]}.png"
                decompression_engine(f"{compressed_img_filepath}.rde")


    def test_jpeg_baseline_compression(self,
                                       settings_filepath=None,
                                       results_filename=None,
                                       test_path="../assets/test_images"):
        clean_temp_results()
        if not settings_filepath:
            settings_filepath = "../compression_configurations/baseline_jpeg_q100.yaml"
        results_name = f"{results_filename}_{os.path.basename(settings_filepath)}_{dt.now():%Y-%m-%d_%H-%M-%S}.csv"
        baseline_jpeg_compression_engine = BaselineJpeg(settings_filepath)
        results_df = pd.DataFrame(columns=["Image Name", "Compression Name", "Uncompressed Reference Label", "Reference Probability", "Compressed Probabilty of Reference Label", "Confidence Drift"])

        for root, dirs, files in os.walk(test_path):
            for file in files:
                test_image_path = os.path.join(root, file)
                topk_result = compare_topk_to_uncompressed_reference_jpeg(test_image_path,
                                                                     baseline_jpeg_compression_engine)

                results_df = df_add_row(results_df, [file,
                                                     os.path.basename(settings_filepath),
                                                     topk_result["top1_class_index"],
                                                     topk_result["uncompressed_prob"],
                                                     topk_result["compressed_prob"],
                                                     topk_result["confidence_delta"]])

        results_df.to_csv(os.path.join(os.getcwd(),"results",results_name))

    def test_quality_factor_sweep_jpeg_compression(self):
        settings_path_list = ["../compression_configurations/baseline_jpeg_q100.yaml",
                              "../compression_configurations/baseline_jpeg_q60.yaml",
                              "../compression_configurations/baseline_jpeg_q20.yaml",
                              "../compression_configurations/baseline_jpeg_q10.yaml",
                              "../compression_configurations/baseline_jpeg_q5.yaml",
                              "../compression_configurations/baseline_jpeg_q4.yaml",
                              "../compression_configurations/baseline_jpeg_q3.yaml",
                              "../compression_configurations/baseline_jpeg_q2.yaml",
                              "../compression_configurations/baseline_jpeg_q1.yaml"]
        for settings_filepath in settings_path_list:
            self.test_jpeg_baseline_compression(settings_filepath=settings_filepath)
if __name__ == "__main__":
    test_homemade_compression()