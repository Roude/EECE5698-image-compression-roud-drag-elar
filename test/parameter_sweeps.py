from PIL import Image
import pillow_avif
import pandas as pd
import numpy as np
from typing import Callable
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage import io
from datetime import datetime as dt
import warnings
import os

from src.compression import BaselineJpeg, FlexibleJpeg
from src.decompression import FlexibleJpegDecompress, BaselineJpegDecompress
from test.classification_tests import compare_topk_to_uncompressed_reference_flexible, df_add_row, clean_temp_results, class_labels_resnet50

def compression_performance(image_path, settings, results_dir):
    uncompressed_img = Image.open(image_path)
    compression_engine = FlexibleJpeg(settings)
    compression_engine.save_location = os.path.join(results_dir,
                                                    "compressed_files",
                                                    f"{image_path.split(".")[0]}_{os.path.basename(settings)}_{dt.now().strftime("%Y%m%d_%H-%M-%S")}")
    decompression_engine = FlexibleJpegDecompress(settings)
    comp_result, comp_metrics = compression_engine(image_path)
    _, decomp_img, decomp_timings = decompression_engine(comp_result)
    score, diff = ssim(uncompressed_img, decomp_img, full=True)
    psrn_val = psnr(uncompressed_img, decomp_img)
    io.imsave(os.path.join(results_dir, "compressed_files", f"{image_path.split(".")[0]}_{os.path.basename(settings)}_difference_{dt.now().strftime("%Y%m%d_%H-%M-%S")}" ))

    return {"ssim_score": score,
            "psnr": psrn_val,
            "compression_metrics": comp_metrics,
            "decompression_metrics": decomp_timings}

# def sweep_image_compression(image_dir, settings_dir, results_dir):
#     results_df = pd.DataFrame(columns=["Image Name",
#                  "Compression Config",
#                  "Chromiance Downsample Factor",
#                  "Block Size",
#                  "Luma Quantization Max",
#                  "Luma Quantization Stdv",
#                  "Chroma Quantization Max",
#                  "Chroma Quantization Stdv",
#                  "Encoded Image Data Size",
#                  "Huffman Table Size (kB)",
#                  "Header Info Size (kB)",
#                  "Compression Ratio",
#                  "Compression Preprocessing Time (mS)",
#                  "Image Compression Time (mS)",
#                  "Entropy Encoding Time (mS)",
#                  "File Encoding Time (mS)",
#                  "Decompression Preprocessing Time (mS)",
#                  "Decompression Entropy Decoding Time (mS)",
#                  "Decompression Block Processing time (mS)",])
#     for root, dirs, settings_files in os.walk(settings_dir):
#         for settings in settings_files:
#             for root, dirs, imgs in os.walk(image_dir):
#                 for img in imgs:
#                     result = compression_performance(img, settings, results_dir)
#                     results_df = df_add_row(results_df, [img, settings, ])


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
            topk_result, comp_metrics, decomp_metrics, ssim_diff  = compare_topk_to_uncompressed_reference_flexible(test_image_path,
                                                                          compression_engine,
                                                                          decompresion_engine)

            io.imsave(os.path.join(results_dir, "compressed_files",
                                   f"{file.split(".")[0]}_{os.path.basename(settings_filepath)}_difference_{dt.now().strftime("%Y%m%d_%H-%M-%S")}"), ssim_diff)

            results_df = df_add_row(results_df, [file,
                                                 os.path.basename(settings_filepath),
                                                 compression_engine.config["quality_factor"],
                                                 round(comp_metrics["compression_metrics"]["compression_ratio"],3),
                                                 round(topk_result["ssim_score"],3),
                                                 round(topk_result["psnr_value"],3),
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
            topk_result, compression_metrics, decompression_timings, ssim_diff = compare_topk_to_uncompressed_reference_flexible(test_image_path,
                                                                          compression_engine,
                                                                          decompresion_engine)
            io.imsave(os.path.join(results_dir, "compressed_files",
                                   f"{file.split(".")[0]}_{os.path.basename(settings_filepath)}_difference_{dt.now().strftime("%Y%m%d_%H-%M-%S")}"), ssim_diff)

            results_df = df_add_row(results_df, [file,
                                                 os.path.basename(settings_filepath),
                                                 compression_engine.config["chrominance_downsample_factor"],
                                                 compression_engine.config["block_size"],
                                                 compression_engine.config["luminance_quantization"]["max_quantization"],
                                                 compression_engine.config["luminance_quantization"]["standard_dev"],
                                                 compression_engine.config["chromiance_quantization"]["max_quantization"],
                                                 compression_engine.config["chromiance_quantization"]["standard_dev"],
                                                 compression_metrics["compressed_size_components"]["encoded_image_data_kb"],
                                                 compression_metrics["compressed_size_components"]["huffman_tables_kb"],
                                                 compression_metrics["compressed_size_components"]["header_overhead_kb"],
                                                 compression_metrics["compression_metrics"]["compression_ratio"],
                                                 compression_metrics["time_metrics"]["preliminaries_ms"],
                                                 compression_metrics["time_metrics"]["process_blocks_ms"],
                                                 compression_metrics["time_metrics"]["entropy_encode_ms"],
                                                 compression_metrics["time_metrics"]["encode_to_file_ms"],
                                                 decompression_timings["preliminaries_ms"],
                                                 decompression_timings["entropy_decoding_ms"],
                                                 decompression_timings["process_blocks_ms"],
                                                 round(topk_result["ssim_score"],3),
                                                 round(topk_result["psnr_value"],3),
                                                 topk_result["top1_class_index"],
                                                 class_labels_resnet50[topk_result["top1_class_index"]],
                                                 round(topk_result["uncompressed_prob"],2),
                                                 round(topk_result["compressed_prob"],2),
                                                 round(topk_result["confidence_delta"],3)])
            # print(results_df)
    return results_df

def test_compression_LN_resnet50(settings_filepath=None,
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
            topk_result, compression_metrics, decompression_timings, ssim_diff = compare_topk_to_uncompressed_reference_flexible(test_image_path,
                                                                          compression_engine,
                                                                          decompresion_engine)
            io.imsave(os.path.join(results_dir, "compressed_files",
                                   f"{file.split(".")[0]}_{os.path.basename(settings_filepath)}_difference_{dt.now().strftime("%Y%m%d_%H-%M-%S")}"),
                      ssim_diff)

            results_df = df_add_row(results_df, [file,
                                                 os.path.basename(settings_filepath),
                                                 compression_engine.config["chrominance_downsample_factor"],
                                                 compression_engine.config["block_size"],
                                                 compression_engine.config["luminance_quantization"]["N"],
                                                 compression_engine.config["luminance_quantization"]["max_val"],
                                                 compression_engine.config["luminance_quantization"]["min_val"],
                                                 compression_engine.config["chromiance_quantization"]["max_val"],
                                                 compression_engine.config["chromiance_quantization"]["min_val"],
                                                 compression_metrics["compressed_size_components"]["encoded_image_data_kb"],
                                                 compression_metrics["compressed_size_components"]["huffman_tables_kb"],
                                                 compression_metrics["compressed_size_components"]["header_overhead_kb"],
                                                 compression_metrics["compression_metrics"]["compression_ratio"],
                                                 compression_metrics["time_metrics"]["preliminaries_ms"],
                                                 compression_metrics["time_metrics"]["process_blocks_ms"],
                                                 compression_metrics["time_metrics"]["entropy_encode_ms"],
                                                 compression_metrics["time_metrics"]["encode_to_file_ms"],
                                                 decompression_timings["preliminaries_ms"],
                                                 decompression_timings["entropy_decoding_ms"],
                                                 decompression_timings["process_blocks_ms"],
                                                 round(topk_result["ssim_score"],3),
                                                 round(topk_result["psnr_value"],3),
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
                 "Compression Config",
                 "Chromiance Downsample Factor",
                 "Block Size",
                 "Luma Quantization Max",
                 "Luma Quantization Stdv",
                 "Chroma Quantization Max",
                 "Chroma Quantization Stdv",
                 "Encoded Image Data Size",
                 "Huffman Table Size (kB)",
                 "Header Info Size (kB)",
                 "Compression Ratio",
                 "Compression Preprocessing Time (mS)",
                 "Image Compression Time (mS)",
                 "Entropy Encoding Time (mS)",
                 "File Encoding Time (mS)",
                 "Decompression Preprocessing Time (mS)",
                 "Decompression Entropy Decoding Time (mS)",
                 "Decompression Block Processing time (mS)",
                 "SSIM Score",
                 "PSNR",
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
                     "SSIM Score",
                     "PSNR",
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

def sweep_LN_quantization(settings_dir, image_dir, results_dir, results_name):
    result_df = pd.DataFrame(
        columns=["Image Name",
                 "Compression Config",
                 "Chromiance Downsample Factor",
                 "Block Size",
                 "Norm N",
                 "Luma Quantization Max",
                 "Luma Quantization Min",
                 "Chroma Quantization Max",
                 "Chroma Quantization Min",
                 "Encoded Image Data Size",
                 "Huffman Table Size (kB)",
                 "Header Info Size (kB)",
                 "Compression Ratio",
                 "Compression Preprocessing Time (mS)",
                 "Image Compression Time (mS)",
                 "Entropy Encoding Time (mS)",
                 "File Encoding Time (mS)",
                 "Decompression Preprocessing Time (mS)",
                 "Decompression Entropy Decoding Time (mS)",
                 "Decompression Block Processing time (mS)",
                 "SSIM Score",
                 "PSNR",
                 "Uncompressed Reference Index",
                 "Uncompressed Reference Label",
                 "Reference Probability",
                 "Compressed Probability of Reference Label",
                 "Confidence Drift"])

    for root, dirs, files in os.walk(settings_dir):
        for settings_file in files:
            settings_filepath = os.path.join(root, settings_file)
            # results_filepath = os.path.join(results_dir, f"{results_name}_{os.path.basename(settings_filepath)}_{dt.now():%Y-%m-%d_%H-%M-%S}.csv")
            result_df = test_compression_LN_resnet50(settings_filepath, results_dir, image_dir, result_df)
    result_df.to_csv(os.path.join(results_dir, f"{results_name}.csv"))

def sweep_quantization_chroma_luma():
    settings_dir = os.path.join(os.getcwd(), 'compression_configurations', 'quantization_sweep')
    results_dir = os.path.join(os.getcwd(), 'test', 'results', 'quantization_sweep')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, "compressed_files"), exist_ok=True)
    # os.makedirs(os.path.join(results_dir, "processed_images"), exist_ok=True)
    img_dir = os.path.join(os.getcwd(), 'assets', 'test_images', 'compression_performance_test')

    sweep_gaussian_quantization(settings_dir, img_dir, results_dir, 'quarter_gauss_quantization_sweep_chroma_luma')

def sweep_quantization_chroma():
    settings_dir = os.path.join(os.getcwd(), 'compression_configurations', 'quantization_sweep_chroma')
    results_dir = os.path.join(os.getcwd(), 'test', 'results', 'quantization_sweep')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, "compressed_files"), exist_ok=True)
    # os.makedirs(os.path.join(results_dir, "processed_images"), exist_ok=True)
    img_dir = os.path.join(os.getcwd(), 'assets', 'test_images', 'compression_performance_test')

    sweep_gaussian_quantization(settings_dir, img_dir, results_dir, 'quarter_gauss_quantization_sweep_chroma')

def sweep_quantization_luma():
    settings_dir = os.path.join(os.getcwd(), 'compression_configurations', 'quantization_sweep_luma')
    results_dir = os.path.join(os.getcwd(), 'test', 'results', 'quantization_sweep')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, "compressed_files"), exist_ok=True)
    # os.makedirs(os.path.join(results_dir, "processed_images"), exist_ok=True)
    img_dir = os.path.join(os.getcwd(), 'assets', 'test_images', 'compression_performance_test')

    sweep_gaussian_quantization(settings_dir, img_dir, results_dir, 'quarter_gauss_quantization_sweep_luma')

def sweep_block_size():
    settings_dir = os.path.join(os.getcwd(), 'compression_configurations', 'block_size_sweep')
    results_dir = os.path.join(os.getcwd(), 'test', 'results', 'block_size_sweep')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, "compressed_files"), exist_ok=True)
    # os.makedirs(os.path.join(results_dir, "processed_images"), exist_ok=True)
    img_dir = os.path.join(os.getcwd(), 'assets', 'test_images', 'compression_performance_test')

    sweep_gaussian_quantization(settings_dir, img_dir, results_dir, 'block_size_quantization_sweep')

def sweep_downsample_chromiance():
    settings_dir = os.path.join(os.getcwd(), 'compression_configurations', 'downsample_sweep')
    results_dir = os.path.join(os.getcwd(), 'test', 'results', 'downsample_sweep')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, "compressed_files"), exist_ok=True)
    # os.makedirs(os.path.join(results_dir, "processed_images"), exist_ok=True)
    img_dir = os.path.join(os.getcwd(), 'assets', 'test_images', 'compression_performance_test')

    sweep_gaussian_quantization(settings_dir, img_dir, results_dir, 'downsample_chromiance_sweep')

def LN_quantization():
    settings_dir = os.path.join(os.getcwd(), 'compression_configurations', 'quantization_sweep_LN_norm')
    results_dir = os.path.join(os.getcwd(), 'test', 'results', 'quantization_sweep')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, "compressed_files"), exist_ok=True)
    # os.makedirs(os.path.join(results_dir, "processed_images"), exist_ok=True)
    img_dir = os.path.join(os.getcwd(), 'assets', 'test_images', 'compression_performance_test')

    sweep_LN_quantization(settings_dir, img_dir, results_dir, 'quantization_sweep_LN_norm')

def sweep_baseline_jpeg_compression():
    settings_dir = os.path.join(os.getcwd(), 'compression_configurations', 'baseline_jpeg_sweep')
    results_dir = os.path.join(os.getcwd(), 'test', 'results', 'jpeg_sweep')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, "compressed_files"), exist_ok=True)
    # os.makedirs(os.path.join(results_dir, "processed_images"), exist_ok=True)
    img_dir = os.path.join(os.getcwd(), 'assets', 'test_images', 'compression_performance_test')

    sweep_baseline_jpeg(settings_dir, img_dir, results_dir, 'baseline_jpeg_sweep')

if __name__ == '__main__':
    sweep_baseline_jpeg_compression()
    LN_quantization()
    sweep_quantization_chroma_luma()
    # sweep_quantization_chroma()
    # sweep_quantization_luma()
    sweep_block_size()
    sweep_downsample_chromiance()
#