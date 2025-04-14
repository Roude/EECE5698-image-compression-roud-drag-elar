import unittest
import os
from src.compression import BaselineJpeg

class BaselineCompressionTest(unittest.TestCase):
    def test_compare_quality_factors(self):
    # path to a sample image we'll compress repeatedly. We'll use this same image to test compression across various JPEG quality levels.
    image_path = "assets/test_images/landscape.png"

    # JPEG quality levels we want to test. these match the YAML config files we’ve defined (baseline_jpeg_q100.yaml, ETC...)
    qualities = [100, 80, 60, 40, 20]

    for q in qualities:
        # each quality level has its own config file; just swapping them dynamically here
        config_path = f"compression_configurations/baseline_jpeg_q{q}.yaml"

        # create a compressor instance using the selected config
        jpeg_compressor = BaselineJpeg(config_path)

        # compress the test image using the compressor. according to how BaselineJpeg is implemented, this returns the output path.
        save_path = jpeg_compressor(image_path)

        # check that the file actually exists after compression
        # at a minimum, we want to confirm the file was actually saved.in the future, we could extend this test to check file size or visual quality...
        self.assertTrue(os.path.exists(save_path), f"Image not saved at quality {q}")

#run the test with: python -m unittest test/baselinecompressiontest.py
if __name__ == '__main__':
    unittest.main()
