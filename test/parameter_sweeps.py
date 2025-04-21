from PIL import Image
import pillow_avif
import pandas as pd
import numpy as np
from typing import Callable
from datetime import datetime as dt
import warnings
import os

from src.compression import BaselineJpeg, FlexibleJpeg
from src.decompression import FlexibleJpegDecompress, BaselineJpegDecompress
from test.classification_tests import compare_topk_to_uncompressed_reference_flexible, df_add_row, clean_temp_results, class_labels_resnet50

def test_compression_baseline_resnet50(settings_filepath=None,
                              results_dir=None,
                              test_image_dir="../assets/test_images/short test",
                              results_df=None):
    clean_temp_results()
    if not settings_filepath:
        settings_filepath = "../compression_configurations/homemade_compression_gauss.yaml"
    # results_name = f"{results_filename}_{os.path.basename(settings_filepath)}_{dt.now():%Y-%m-%d_%H-%M-%S}.csv"
    compression_engine = BaselineJpeg(settings_filepath)
    decompresion_engine = BaselineJpegDecompress(settings_filepath)
    decompresion_engine.save_location = os.path.join(results_dir, "processed_images")


    for root, dirs, files in os.walk(test_image_dir):
        for file in files:
            test_image_path = os.path.join(root, file)
            compression_engine.save_location = os.path.join(results_dir,
                                                            "compressed_files",
                                                            f"{file.split(".")[0]}_{os.path.basename(settings_filepath)}_{dt.now().strftime("%Y%m%d_%H-%M-%S")}")
            topk_result = compare_topk_to_uncompressed_reference_flexible(test_image_path,
                                                                          compression_engine,
                                                                          decompresion_engine)

            results_df = df_add_row(results_df, [file,
                                                 os.path.basename(settings_filepath),
                                                 compression_engine.config["quality_factor"],
                                                 round(topk_result["compression_ratio"],3),
                                                 topk_result["top1_class_index"],
                                                 class_labels_resnet50[topk_result["top1_class_index"]],
                                                 round(topk_result["uncompressed_prob"],2),
                                                 round(topk_result["compressed_prob"],2),
                                                 round(topk_result["confidence_delta"],3)])
            # print(results_df)
    return results_df

def test_compression_resnet50(settings_filepath=None,
                              results_dir=None,
                              test_image_dir="../assets/test_images/short test",
                              results_df=None):
    clean_temp_results()
    if not settings_filepath:
        settings_filepath = "../compression_configurations/homemade_compression_gauss.yaml"
    # results_name = f"{results_filename}_{os.path.basename(settings_filepath)}_{dt.now():%Y-%m-%d_%H-%M-%S}.csv"
    compression_engine = FlexibleJpeg(settings_filepath)
    decompresion_engine = FlexibleJpegDecompress()
    decompresion_engine.save_location = os.path.join(results_dir, "processed_images")


    for root, dirs, files in os.walk(test_image_dir):
        for file in files:
            test_image_path = os.path.join(root, file)
            compression_engine.save_location = os.path.join(results_dir,
                                                            "compressed_files",
                                                            f"{file.split(".")[0]}_{os.path.basename(settings_filepath)}_{dt.now().strftime("%Y%m%d_%H-%M-%S")}")
            topk_result = compare_topk_to_uncompressed_reference_flexible(test_image_path,
                                                                          compression_engine,
                                                                          decompresion_engine)

            results_df = df_add_row(results_df, [file,
                                                 os.path.basename(settings_filepath),
                                                 compression_engine.config["chrominance_downsample_factor"],
                                                 compression_engine.config["block_size"],
                                                 compression_engine.config["luminance_quantization"]["max_quantization"],
                                                 compression_engine.config["luminance_quantization"]["standard_dev"],
                                                 compression_engine.config["chromiance_quantization"]["max_quantization"],
                                                 compression_engine.config["chromiance_quantization"]["standard_dev"],
                                                 round(topk_result["compression_ratio"],3),
                                                 topk_result["top1_class_index"],
                                                 class_labels_resnet50[topk_result["top1_class_index"]],
                                                 round(topk_result["uncompressed_prob"],2),
                                                 round(topk_result["compressed_prob"],2),
                                                 round(topk_result["confidence_delta"],3)])
            # print(results_df)
    return results_df

def sweep_gaussian_quantization(settings_dir, image_dir, results_dir, results_name):
    result_df = pd.DataFrame(
        columns=["Image Name",
                 "Compression Name",
                 "Chromiance Downsample Factor",
                 "Block Size",
                 "Luma Quantization Max",
                 "Luma Quantization Stdv",
                 "Chroma Quantization Max",
                 "Chroma Quantizatoin Stdv",
                 "Compression Rate (uncompressed image size / compressed file size)",
                 "Uncompressed Reference Index",
                 "Uncompressed Reference Label",
                 "Reference Probability",
                 "Compressed Probability of Reference Label",
                 "Confidence Drift"])

    for root, dirs, files in os.walk(settings_dir):
        for settings_file in files:
            settings_filepath = os.path.join(root, settings_file)
            # results_filepath = os.path.join(results_dir, f"{results_name}_{os.path.basename(settings_filepath)}_{dt.now():%Y-%m-%d_%H-%M-%S}.csv")
            result_df = test_compression_resnet50(settings_filepath, results_dir, image_dir, result_df)
    result_df.to_csv(os.path.join(results_dir, f"{results_name}.csv"))

def sweep_baseline_jpeg(settings_dir, image_dir, results_dir, results_name):
    result_df = pd.DataFrame(
            columns=["Image Name",
                     "Compression Name",
                     "Quality Factor",
                     "Compression Rate (uncompressed image size / compressed file size)",
                     "Uncompressed Reference Index",
                     "Uncompressed Reference Label",
                     "Reference Probability",
                     "Compressed Probability of Reference Label",
                     "Confidence Drift"])

    for root, dirs, files in os.walk(settings_dir):
        for settings_file in files:
            settings_filepath = os.path.join(root, settings_file)
            # results_filepath = os.path.join(results_dir, f"{results_name}_{os.path.basename(settings_filepath)}_{dt.now():%Y-%m-%d_%H-%M-%S}.csv")
            result_df = test_compression_baseline_resnet50(settings_filepath, results_dir, image_dir, result_df)
    result_df.to_csv(os.path.join(results_dir, f"{results_name}.csv"))

def sweep_LN_quantization(settings_dir, image_dir, results_dir):
    pass

def sweep_quantization():
    settings_dir = os.path.join(os.getcwd(), 'compression_configurations', 'quantization_sweep')
    results_dir = os.path.join(os.getcwd(), 'test', 'results', 'quantization_sweep')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, "compressed_files"), exist_ok=True)
    # os.makedirs(os.path.join(results_dir, "processed_images"), exist_ok=True)
    img_dir = os.path.join(os.getcwd(), 'assets', 'test_images', 'short test')

    sweep_gaussian_quantization(settings_dir, img_dir, results_dir, 'quarter_gauss_quantization_sweep')

def sweep_block_size():
    settings_dir = os.path.join(os.getcwd(), 'compression_configurations', 'block_size_sweep')
    results_dir = os.path.join(os.getcwd(), 'test', 'results', 'block_size_sweep')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, "compressed_files"), exist_ok=True)
    # os.makedirs(os.path.join(results_dir, "processed_images"), exist_ok=True)
    img_dir = os.path.join(os.getcwd(), 'assets', 'test_images', 'short test')

    sweep_gaussian_quantization(settings_dir, img_dir, results_dir, 'block_size_quantization_sweep')

def sweep_downsample_chromiance():
    settings_dir = os.path.join(os.getcwd(), 'compression_configurations', 'downsample_sweep')
    results_dir = os.path.join(os.getcwd(), 'test', 'results', 'downsample_sweep')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, "compressed_files"), exist_ok=True)
    # os.makedirs(os.path.join(results_dir, "processed_images"), exist_ok=True)
    img_dir = os.path.join(os.getcwd(), 'assets', 'test_images', 'short test')

    sweep_gaussian_quantization(settings_dir, img_dir, results_dir, 'block_size_quantization_sweep')

def sweep_baseline_jpeg_compression():
    settings_dir = os.path.join(os.getcwd(), 'compression_configurations', 'baseline_jpeg_sweep')
    results_dir = os.path.join(os.getcwd(), 'test', 'results', 'jpeg_sweep')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, "compressed_files"), exist_ok=True)
    # os.makedirs(os.path.join(results_dir, "processed_images"), exist_ok=True)
    img_dir = os.path.join(os.getcwd(), 'assets', 'test_images', 'short test')

    sweep_baseline_jpeg(settings_dir, img_dir, results_dir, 'block_size_quantization_sweep')

if __name__ == '__main__':
    sweep_baseline_jpeg_compression()
    sweep_quantization()
    sweep_block_size()
    sweep_downsample_chromiance()
