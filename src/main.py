import os
from src.compression import FlexibleJpeg
from src.decompression import FlexibleJpegDecompress

if __name__ == '__main__':
    flexible_jpeg = FlexibleJpeg()
    flexible_jpeg_decompressor = FlexibleJpegDecompress()

    image_path = os.path.join(os.getcwd(), "assets", "unit_test_images","white_16x16.tif")
    compression_config = os.path.join(os.getcwd(),
                                      "compression_configurations",
                                      "homemade_compression_jpeg_like.yaml")

    flexible_jpeg(image_path, compression_config)
    print(f"Image compressed using FlexibleJpeg algorithm. Output saved to: {flexible_jpeg.save_location}")
    decomp_image, save_location = flexible_jpeg_decompressor(f"{flexible_jpeg.save_location}.rde")

