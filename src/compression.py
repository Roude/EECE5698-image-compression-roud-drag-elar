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
from scipy.fftpack import dct, idct
from src.utilities import display_greyscale_image
from src.huffman import zigzag_order,run_length_encoding,build_huffman_tree,generate_huffman_codes,huffman_encode
from collections import Counter
import pkgutil

class CompressImage:
    def __init__(self, config=None):
        """
        Here is where we define what configured string maps to what compression function:
        """

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


    # def __call__(self, raw_image_file, config=None):
    #     if config:
    #         self.update_configuration(config)
    #
    #     image_array = io.imread(raw_image_file)
    #     return self.compression_function(image_array, save_location=None, **self.config["compression_parameters"])

    def set_datatype_and_channels(self, image_uncompressed):
        if image_uncompressed.dtype != np.uint8:
            image_uncompressed = (255 * (image_uncompressed / np.max(image_uncompressed))).astype(np.uint8)

        # Remove alpha channel if present (JPEG does not support transparency)
        if image_uncompressed.shape[-1] == 4:
            image_uncompressed = image_uncompressed[:, :, :3]
        return image_uncompressed

class BaselineJpeg(CompressImage):
    def __init__(self, config):
        super().__init__(config)

    def __call__(self, image, save_location=None, **kwargs):
        """
        :param image_uncompressed: image as a numpy array
        :param save_location: if none is provided, will be saved to the tmp directory. If it is provided,
        omit the .jpeg extension from the filename.
        :return:
        """
        quality_factor = kwargs.get("quality_factor", 100)
        save_location = kwargs.get("save_location", None)

        if isinstance(image, str):
            image_uncompressed = io.imread(image)
        else:
            image_uncompressed = image

        image_uncompressed = self.set_datatype_and_channels(image_uncompressed)

        if not save_location:
            file_save_location = os.path.join(os.getcwd(), "tmp", f"temp_img_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jpeg")
        else:
            file_save_location = os.path.join(os.getcwd(),f"{save_location}.jpeg")

        imageio.imwrite(file_save_location, image_uncompressed,
                  quality=quality_factor)
        return f"{save_location}.jpeg"

class FlexibleJpeg(CompressImage):
    def __init__(self, config=None):
        super().__init__(config)
        self.block_size = 8
        self.downsample_factor = 2
        self.YCbCr_conversion_matrix = None
        self.YCbCr_conversion_offset = None
        self.default_YCbCr_conversion_matrix = np.array([[65.738, 129.057, 25.064],
                                                 [-37.945, -74.494, 112.439],
                                                 [112.439, -94.154, -18.285]], dtype=np.float32) / 256

        self.default_YCbCr_conversion_offset = np.array([16, 128, 128]).transpose()

        self.default_chromiance_quantization_table = np.array([[10, 8, 9, 9, 9, 8, 10, 9],
                                                               [9, 9, 10, 10, 10, 11, 12, 17],
                                                               [13, 12, 12, 12, 12, 20, 16, 16],
                                                               [14, 17, 18, 20, 23, 23, 22, 20],
                                                               [25, 25, 25, 25, 25, 25, 25, 25],
                                                               [25, 25, 25, 25, 25, 25, 25, 25],
                                                               [25, 25, 25, 25, 25, 25, 25, 25],
                                                               [25, 25, 25, 25, 25, 25, 25, 25]],
                                                              dtype=np.uint8)
        self.chromiance_quantization_table = self.default_chromiance_quantization_table

        self.default_lumiance_quantization_table = np.array([[6, 4, 4, 6, 10, 16, 20, 24],
                                                             [5, 5, 6, 8, 10, 23, 24, 22],
                                                             [6, 5, 6, 10, 16, 23, 28, 22],
                                                             [6, 7, 9, 12, 20, 35, 32, 25],
                                                             [7, 9, 15, 22, 27, 44, 41, 31],
                                                             [10, 14, 22, 26, 32, 42, 45, 37],
                                                             [20, 26, 31, 35, 41, 48, 48, 40],
                                                             [29, 37, 38, 39, 45, 40, 41, 40]],
                                                            dtype=np.uint8)
        self.lumiance_quantization_table = self.default_lumiance_quantization_table
        self.lumiance_datatype = np.uint8
        self.chromiance_datatype = np.uint8

    def __call__(self, image, settings):
        """
        Implementation of JPEG-like compression with individual functions broken out.
        :param image_uncompressed:
        :param kwargs:
        :return:
        """
        # Represent as 8-bit unsigned int.
        if isinstance(image, str):
            image_uncompressed = io.imread(image)
        else:
            image_uncompressed = image
        if isinstance(settings, str):
            with open(settings, 'r') as settings_file:
                settings = yaml.safe_load(settings_file)

        image_uncompressed = self.set_datatype_and_channels(image_uncompressed)
        YCbCrImage = self.convert_colorspace(image_uncompressed, **settings)
        downsampled_image = self.downsample_chromiance(YCbCrImage, **settings)
        block_processed_channels = self.process_blocks(downsampled_image, **settings)
        compressed_image_datastream = self.entropy_encode(block_processed_channels)
        self.encode_to_file(compressed_image_datastream, settings)

    def convert_colorspace(self, image_uncompressed, **kwargs):
        """
        Apply YCbCr color conversion to image
        :param image_uncompressed:
        :param kwargs:
        :return:
        """
        self.YCbCr_conversion_matrix = kwargs.get("YCbCr_conversion_matrix", self.default_YCbCr_conversion_matrix)
        self.YCbCr_conversion_offset = kwargs.get("YCbCr_conversion_offset", self.default_YCbCr_conversion_offset)
        return np.array(np.tensordot(image_uncompressed,
                                     self.YCbCr_conversion_matrix,
                                     axes=([2],[1])) +
                        self.YCbCr_conversion_offset,
                        dtype=np.uint8)

    def downsample_chromiance(self, YCbCr_image, **kwargs):
        """
        Apply downsampling factor to YCbCr formatted image and save the image as a tuple of 3 matricies with the
        luminance and two chromiance channels.
        :param YCbCr_image: The image already converted to YCbCr format
        :return: A Tuple, element 0 is the lumiance channel, element 1 is the chromiance channels
        """

        self.downsample_factor = kwargs.get("chromiance_downsample_factor", 4)

        lumiance = YCbCr_image[:, :, 0]
        ch_CbCr = YCbCr_image[:,:,1:]

        running_average = np.array(ch_CbCr[::self.downsample_factor, ::self.downsample_factor,:]/self.downsample_factor**2,
                                   dtype=np.uint8)
        for idx in range(self.downsample_factor):
            for jdx in range(self.downsample_factor):
                if idx == 0 and jdx == 0:
                    continue
                else:
                    running_average = (running_average +
                                       ch_CbCr[idx::self.downsample_factor, jdx::self.downsample_factor,:]/self.downsample_factor**2)

        return [lumiance,
                np.array(running_average[:,:,0],dtype=np.uint8),
                np.array(running_average[:,:,1],dtype=np.uint8)]

    def process_blocks(self, downsampled_image, **kwargs):
        """
        Perform compression steps that are executed on individual blocks of a user defined size.
        :param downsampled_image: The output of the function downsample_chromiance
        :param kwargs:
        :return: The uncompressed converted and quantized matricies.
        """
        self.block_size = kwargs.get("block_size", 8)

        block_processed_channels = []
        for ch_num, channel in enumerate(downsampled_image):
            block_processed_channels.append(np.zeros(shape=np.shape(channel),dtype=np.int8))
            for idx in range(0,np.shape(channel)[0], self.block_size):
                for jdx in range(0, np.shape(channel)[1], self.block_size):
                    end_idx = idx + self.block_size if np.shape(channel)[0] - idx > self.block_size else -1
                    end_jdx = idx + self.block_size if np.shape(channel)[1] - jdx > self.block_size else -1
                    block_processed_channels[ch_num][idx:end_idx, jdx:end_jdx] = self.block_dct(channel[idx:end_idx, jdx:end_jdx], **kwargs)
                    self.quantize_block(block_processed_channels[ch_num][idx:end_idx, jdx:end_jdx], ch_num, **kwargs)
        return block_processed_channels

    def block_dct(self, image_block, **kwargs):
        """
        First scale pixels from 0 to 255 to -128 to 128, then perform DCT on the block
        :param image_block: An image block of a flexible size, for the DCT to be performed on.
        :param kwargs:
        :return: the cosine transformed block
        """
        block_for_transform = (image_block.astype(np.int16) - 128).astype(np.int8)

        return np.array(dct(block_for_transform), dtype=np.int8)


    def quantize_block(self, frequency_domain_block, ch_num, **kwargs):
        """
        Divide each block by the quantization table (passed as a keyword argument or the default).
        Round to the nearest integer
        :param frequency_domain_block: The output of block_dct
        :param ch_num: automatically passed in by process_blocks. Indicates which of the quantization tables to use.
        :param kwargs:
        :return: the quantized block
        """
        self.chromiance_quantization_table = kwargs.get("chromiance_quantization_table",
                                                        self.default_chromiance_quantization_table)

        self.lumiance_quantization_table = kwargs.get("lumiance_quantization_table",
                                                      self.default_lumiance_quantization_table)
        if ch_num == 0:
            frequency_domain_block = (frequency_domain_block/self.lumiance_quantization_table).astype(np.int8)
        else:
            frequency_domain_block = (frequency_domain_block/self.chromiance_quantization_table).astype(np.int8)
        return frequency_domain_block

    def entropy_encode(self, quantized_blocks):
        """
        Applies Zig-Zag ordering, Run-Length Encoding, and Huffman Encoding
        :param: quantized_blocks: The output of quantize_block
        :return: compressed bits, the respective huffman code
        """
        encoded_data = []
        for block in quantized_blocks:
            zigzag_block = zigzag_order(block)
            rle_block = run_length_encoding(zigzag_block)
            encoded_data.append(rle_block)

        # Build Huffman Tree based on symbol frequency
        flat_rle = [val for block in encoded_data for val, _ in block]
        freq_dict = Counter(flat_rle)
        huffman_tree = build_huffman_tree(freq_dict)
        huffman_codes = generate_huffman_codes(huffman_tree)

        # Encode RLE data using Huffman Codes
        compressed_bits = [huffman_encode(block, huffman_codes) for block in encoded_data]
        return compressed_bits, huffman_codes

    def encode_to_file(self, full_image_transformed_and_blocked, settings):
        """
        Turn the image matrices into a data stream. Pre-append the configurations necessary
        to reconstruct the full image.
        .rde filetype is our proprietary image compression format :)
        :param full_image_transformed_and_blocked:
        :param kwargs:
        :return:
        """
        self.save_location = settings.get("save_location", "")
        if self.save_location == "":
            self.save_location = os.path.join(os.getcwd(),
                                                     "tmp",
                                                     f"flex_jpeg_comp_output_{datetime.now().strftime("%Y%m%d_%H%M%S")}.rde")

        with open(self.save_location,'w') as compressed_file_output:
            compressed_file_output.write("settings_start :: ")
            compressed_file_output.write(str(settings))
            compressed_file_output.write(" :: settings_end :: ")
            compressed_file_output.write(full_image_transformed_and_blocked)
            compressed_file_output.write(" :: image_end")

""" 
Use this function block to test things out.
"""

compression_algorithm_reference = {
    "jpeg_baseline": BaselineJpeg,
    "homemade_jpeg_like": FlexibleJpeg
}


if __name__ == '__main__':

    baseline_jpeg = BaselineJpeg(os.path.join(os.getcwd(),
                                              "compression_configurations",
                                              "baseline_jpeg_compression.yaml"))

    flexible_jpeg = FlexibleJpeg()

    test_image_path = os.path.join(os.getcwd(), "assets", "landscape.png")
    compression_config = os.path.join(os.getcwd(),
                                              "compression_configurations",
                                              "homemade_compression_jpeg_like.yaml")
    flexible_jpeg(test_image_path, compression_config)

    # image_array = imread(os.path.join(os.getcwd(), "assets", "landscape.png"))
    #
    # fig = make_subplots(rows=2, cols=3, shared_xaxes=True, shared_yaxes=True)
    #
    # converted_colorspace_image = flexible_jpeg.convert_colorspace(image_array)
    # downsampled_chromiance_image = flexible_jpeg.downsample_chromiance(converted_colorspace_image, chromiance_downsample_factor=2)
    #
    # fig.add_trace(go.Image(z=image_array),row=1, col=1)
    # fig.add_trace(go.Image(z=converted_colorspace_image), row=1,col=2)
    #
    # print(downsampled_chromiance_image)
    #
    # display_greyscale_image(fig, downsampled_chromiance_image[0],row=2, col=1)
    # display_greyscale_image(fig, downsampled_chromiance_image[1], row=2, col=2)
    # display_greyscale_image(fig, downsampled_chromiance_image[2], row=2, col=3)
    #
    # fig.update_yaxes(autorange='reversed', scaleanchor='x', constrain='domain')
    # fig.update_xaxes(constrain='domain')
    # fig.update_layout(coloraxis_showscale=False)  # Remove color scale
    #
    # fig.show()
