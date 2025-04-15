import os
from datetime import datetime
from src.compression import FlexibleJpeg

import os
from datetime import datetime
from src.compression import FlexibleJpeg


def compress_all_test_images():
    # Create results directory with timestamp
    timestamp = datetime.now().strftime("%d_%H%M")
    results_dir = os.path.join(os.getcwd(), "results", timestamp)
    os.makedirs(results_dir, exist_ok=True)

    # Path to test images and config
    original_images_dir = os.path.join(os.getcwd(), "assets", "test_images")
    compression_config = os.path.join(os.getcwd(), "compression_configurations", "homemade_compression_jpeg_like.yaml")

    # Verify paths exist
    if not os.path.exists(original_images_dir):
        raise FileNotFoundError(f"Test images directory not found at: {original_images_dir}")
    if not os.path.exists(compression_config):
        raise FileNotFoundError(f"Config file not found at: {compression_config}")

    # Supported formats (add more as needed)
    SUPPORTED_FORMATS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp', '.webp')

    # Get all files and filter supported ones
    all_files = [f for f in os.listdir(original_images_dir)
                 if os.path.isfile(os.path.join(original_images_dir, f))]

    image_files = [f for f in all_files
                   if f.lower().endswith(SUPPORTED_FORMATS)]

    unsupported_files = [f for f in all_files
                         if f not in image_files]

    if unsupported_files:
        print("\nSkipping unsupported file formats:")
        for f in unsupported_files:
            print(f" - {f}")

    if not image_files:
        print("No supported image files found")
        return


    # Initialize compressor once
    flexible_jpeg = FlexibleJpeg(compression_config)

    # Process each image
    for image_file in image_files:
        try:
            # Get base filename without extension
            base_name = os.path.splitext(image_file)[0]

            # Set paths
            input_path = os.path.join(original_images_dir, image_file)
            save_location = os.path.join(results_dir, base_name)

            print(f"\nProcessing: {image_file}")

            # Compress the image
            flexible_jpeg(input_path, save_location=save_location)

            # Verify output was created
            expected_files = [
                f"{save_location}.rde",
                f"{save_location}.metrics.json"
            ]

            missing_files = [f for f in expected_files if not os.path.exists(f)]
            if missing_files:
                print(f"Warning: Expected output files not created for {image_file}: {missing_files}")
            else:
                print(f"Successfully compressed {image_file}")
                print(f"  - Compressed data: {save_location}.rde")
                print(f"  - Metrics: {save_location}.metrics.json")

        except Exception as e:
            print(f"Error processing {image_file}: {str(e)}")
            continue

    print(f"\nProcessing complete. Results saved to: {results_dir}")


if __name__ == '__main__':
    compress_all_test_images()