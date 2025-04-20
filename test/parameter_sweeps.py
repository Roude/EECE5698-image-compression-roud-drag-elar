from PIL import Image
import pillow_avif
import pandas as pd
import numpy as np
from typing import Callable
from datetime import datetime as dt
import warnings
import os

from src.compression import BaselineJpeg, FlexibleJpeg
from src.decompression import FlexibleJpegDecompress
from test.classification_tests import compare_topk_to_uncompressed_reference_flexible, df_add_row, clean_temp_results


def test_homemade_compression(settings_filepath=None,
                              results_filename=None,
                              test_image_dir="../assets/test_images/short test"):
    clean_temp_results()
    if not settings_filepath:
        settings_filepath = "../compression_configurations/homemade_compression_gauss.yaml"
    results_name = f"{results_filename}_{os.path.basename(settings_filepath)}_{dt.now():%Y-%m-%d_%H-%M-%S}.csv"
    compression_engine = FlexibleJpeg(settings_filepath)
    decompresion_engine = FlexibleJpegDecompress()
    results_df = pd.DataFrame(
        columns=["Image Name", "Compression Name", "Uncompressed Reference Index", "Uncompressed Reference Label", "Reference Probability",
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

def sweep_settings(settings_dir, image_dir, results_dir, results_name):
    for root, dirs, files in os.walk(settings_dir):
        for settings_file in files:
            settings_filepath = os.path.join(root, settings_file)
            results_filepath = os.path.join(results_dir, f"{results_name}_{os.path.basename(settings_filepath)}_{dt.now():%Y-%m-%d_%H-%M-%S}.csv")
            test_homemade_compression(settings_filepath, results_filepath, image_dir)

if __name__ == '__main__':
    settings_dir = os.path.join(os.getcwd(), 'compression_configurations', 'quantization_sweep')
    results_dir = os.path.join(os.getcwd(), 'test', 'results', 'quantization_sweep')
