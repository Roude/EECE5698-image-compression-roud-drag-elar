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
import os
from datetime import datetime
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
        else:
            self.config = config
            self.update_configuration(self.config)



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
        return self.compression_function(image_array, config)

    def jpeg_baseline(self, image_uncompressed, config, save_location=None):
        """
        :param image_uncompressed: image as a numpy array
        :param save_location: if none is provided, will be saved to the tmp directory. If it is provided,
        omit the .jpeg extension from the filename.
        :return:
        """
        if not save_location:
            temp_location = os.path.join(os.getcwd(), "tmp", f"temp_img_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jpeg")
            io.imsave(temp_location,
                      image_uncompressed)
            return temp_location
        else:
            io.imsave(f"{save_location}.jpeg", image_uncompressed)
            return f"{save_location}.jpeg"