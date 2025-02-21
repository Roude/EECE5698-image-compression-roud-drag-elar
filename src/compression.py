# This file to contain high-level functions consisting of a series of sub-routines
# defined in utilities.py
# For example:
# jpeg_compression(file, compression_configuration)
#   convert_color_space(image)
#   perform_jpeg_compression(converted_image)
#   save_jpeg(compressed_image)

# Functions in this file should be composed of as few sub-routines as possible for
# readability and modularity.

from tokenize import String

import yaml
from skimage import io
import imageio.v3 as imageio
import os
from datetime import datetime
import numpy as np
import plotly.express as px

class CompressImage:
    def __init__(self, config=None):
        """
        Here is where we define what configured string maps to what compression function:
        """
        self.compression_algorithm_reference = {
            "jpeg_baseline" : self.jpeg_baseline
        }

        if not config:
            self.config = {}
            self.compression_function = None
            print("No configuration provided")
        else:
            with open(config,'r') as config_file:
                self.config = yaml.load(config_file, yaml.SafeLoader)
            self.update_configuration(self.config)
            # print(self.config)



    def update_configuration(self, config):
        if isinstance(config, str):
            with open(config,'r') as config_file:
                self.config = yaml.load(config_file, yaml.SafeLoader)
        else:
            self.config = config
        self.compression_function = self.get_compression_algorithm()


    def get_compression_algorithm(self):
        return  self.compression_algorithm_reference[self.config["compression_algorithm"]]

    def __call__(self, raw_image_file, config=None):
        if config:
            self.update_configuration(config)

        image_array = io.imread(raw_image_file)
        return self.compression_function(image_array, save_location=None, **self.config["compression_parameters"])

    def jpeg_baseline(self, image_uncompressed, save_location=None, **kwargs):
        """
        :param image_uncompressed: image as a numpy array
        :param save_location: if none is provided, will be saved to the tmp directory. If it is provided,
        omit the .jpeg extension from the filename.
        :return:
        """
        if "quality_factor" in kwargs.keys():
            quality_factor = kwargs["quality_factor"]
        else:
            quality_factor = 100

        if image_uncompressed.dtype != np.uint8:
            image_uncompressed = (255 * (image_uncompressed / np.max(image_uncompressed))).astype(np.uint8)

        # Remove alpha channel if present (JPEG does not support transparency)
        if image_uncompressed.shape[-1] == 4:
            image_uncompressed = image_uncompressed[:, :, :3]

        if not save_location:
            file_save_location = os.path.join(os.getcwd(), "tmp", f"temp_img_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jpeg")
        else:
            file_save_location = os.path.join(os.getcwd(),f"{save_location}.jpeg")
        imageio.imwrite(file_save_location, image_uncompressed,
                  quality=quality_factor)
        return f"{save_location}.jpeg"