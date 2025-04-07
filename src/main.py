import os
from src.compression import FlexibleJpeg

if __name__ == '__main__':
    flexible_jpeg = FlexibleJpeg()

    image_path = os.path.join(os.getcwd(), "assets", "landscape.png")
    compression_config = os.path.join(os.getcwd(),
                                      "compression_configurations",
                                      "homemade_compression_jpeg_like.yaml")

    flexible_jpeg(image_path, compression_config)

    print(f"Image compressed using FlexibleJpeg algorithm. Output saved to: {flexible_jpeg.save_location}")