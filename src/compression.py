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
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from skimage.color import convert_colorspace
from skimage.io import imread


class CompressImage:
    def __init__(self, config=None):
        """
        Here is where we define what configured string maps to what compression function:
        """
        self.compression_algorithm_reference = {
            "jpeg_baseline" : self.jpeg_baseline,
            "homemade_jpeg_like" : self.jpeg_like
        }

        self.YCbCr_conversion_matrix = np.array([[65.738, 129.057, 25.064],
                                                 [-37.945, -74.494, 112.439],
                                                 [112.439, -94.154, -18.285]],dtype=np.float32)/256


        self.YCbCr_conversion_offset = np.array([16, 128, 128]).transpose()

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

    def jpeg_like(self, image_uncompressed, save_location=None, **kwargs):
        """
        Implementation of JPEG-like compression with individual functions broken out.
        :param image_uncompressed:
        :param save_location:
        :param kwargs:
        :return:
        """

        # Represent as 8-bit unsigned int.
        if image_uncompressed.dtype != np.uint8:
            image_uncompressed = (255 * (image_uncompressed / np.max(image_uncompressed))).astype(np.uint8)

        YCbCrImage = convert_colorspace(image_uncompressed, **kwargs)


    def convert_colorspace(self, image_uncompressed, **kwargs):
        """
        Apply YCbCr color conversion to image
        :param image_uncompressed:
        :param kwargs:
        :return:
        """

        return np.array(np.tensordot(image_uncompressed, self.YCbCr_conversion_matrix, axes=([2],[1])) + self.YCbCr_conversion_offset,dtype=np.uint8)

    def downsample_chromiance(self, YCbCr_image, **kwargs):
        """
        Apply downsampling factor to YCbCr formatted image and save the image as a tuple of 3 matricies with the
        luminance and two chromiance channels.
        :param YCbCr_image:
        :return:
        """

        if "chromiance_downsample_factor" in kwargs:
            downsample_factor = kwargs["chromiance_downsample_factor"]
        else:
            downsample_factor = 4

        lumiance = YCbCr_image[:, :, 0]

        ch_CbCr = YCbCr_image[:,:,1:]

        running_average = np.array(ch_CbCr[::downsample_factor, ::downsample_factor,:]/downsample_factor**2,dtype=np.uint8)
        # print(running_average)
        for idx in range(downsample_factor):
            for jdx in range(downsample_factor):
                if idx == 0 and jdx == 0:
                    continue
                else:
                    running_average = (running_average +
                                       ch_CbCr[idx::downsample_factor, jdx::downsample_factor,:]/downsample_factor**2)

        return lumiance, np.array(running_average,dtype=np.uint8)


"""
Use this function block to test things out.
"""
if __name__ == '__main__':

    image_compressor = jpeg_compressor = CompressImage(os.path.join(os.getcwd(),
                                             "compression_configurations",
                                             "baseline_jpeg_compression.yaml"))

    image_array = imread(os.path.join(os.getcwd(), "assets", "landscape.png"))


    fig = make_subplots(rows=2, cols=3, shared_xaxes=True, shared_yaxes=True)

    converted_colorspace_image = image_compressor.convert_colorspace(image_array)
    downsampled_chromiance_image = image_compressor.downsample_chromiance(converted_colorspace_image, chromiance_downsample_factor=2)

    fig.add_trace(go.Image(z=image_array),row=1, col=1)
    fig.add_trace(go.Image(z=converted_colorspace_image), row=1,col=2)

    # print(downsampled_chromiance_image[0])
    # print(downsampled_chromiance_image[1][:,:])
    fig.add_trace(go.Heatmap(z=downsampled_chromiance_image[0],colorscale='gray'),
                                                            row=2,
                                                            col=1,
                                                            )
    fig.add_trace(go.Heatmap(z=downsampled_chromiance_image[1][:,:,0], colorscale='gray'),row=2,col=2)
    fig.add_trace(go.Heatmap(z=downsampled_chromiance_image[1][:,:,1], colorscale='gray'),row=2, col=3)

    fig.update_yaxes(autorange='reversed', scaleanchor='x', constrain='domain')
    fig.update_xaxes(constrain='domain')
    fig.update_layout(coloraxis_showscale=False)  # Remove color scale

    fig.show()
