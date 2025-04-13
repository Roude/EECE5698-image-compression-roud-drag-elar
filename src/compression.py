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
import sys
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
from torch.nn.functional import channel_shuffle

from src.utilities import display_greyscale_image, make_serializable_table
from src.huffman import generate_zigzag_pattern, zigzag_order, run_length_encoding, build_huffman_tree, generate_huffman_codes, huffman_encode
from collections import Counter
import pkgutil
import json

import time #for debugging purposes
#makes it so it doesn't print the np.int16 shit
np.set_printoptions(formatter={
    'object': lambda x: str(x) if not isinstance(x, tuple) else f"({x[0]}, {x[1]})"
})

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

    def __call__(self, image,  **kwargs):
        """
        :param image_uncompressed: image as a numpy array
        :param save_location: if none is provided, will be saved to the tmp directory. If it is provided,
        omit the .jpeg extension from the filename.
        :return:
        """
        quality_factor = self.config.get("quality_factor", 100)
        save_location = kwargs.get("save_location", None)

        if isinstance(image, str):
            image_uncompressed = io.imread(image)
        else:
            image_uncompressed = image

        image_uncompressed = self.set_datatype_and_channels(image_uncompressed)

        if not save_location:
            save_location = os.path.join(os.getcwd(), "tmp", f"temp_img_{datetime.now().strftime("%Y%m%d_%H%M%S")}.jpeg")
        else:
            save_location = os.path.join(os.getcwd(),f"{save_location}.jpeg")

        imageio.imwrite(save_location, image_uncompressed,
                  quality=quality_factor)
        return save_location

class FlexibleJpeg(CompressImage):
    def __init__(self, config=None):
        super().__init__(config)
        self.zigzag_pattern = None
        self.block_size = 8
        self.quality_factor = 50
        self.chrominance_aggression_factor = 3
        self.lumiance_aggression_factor = 1.5
        self.downsample_factor = 2
        self.YCbCr_conversion_matrix = None
        self.YCbCr_conversion_offset = None
        self.default_YCbCr_conversion_matrix = np.array([[65.738, 129.057, 25.064],
                                                 [-37.945, -74.494, 112.439],
                                                 [112.439, -94.154, -18.285]], dtype=np.float32) / 256

        self.default_YCbCr_conversion_offset = np.array([16, 128, 128]).transpose()

        self.default_chrominance_quantization_table = np.array([[10, 8, 9, 9, 9, 8, 10, 9],
                                                               [9, 9, 10, 10, 10, 11, 12, 17],
                                                               [13, 12, 12, 12, 12, 20, 16, 16],
                                                               [14, 17, 18, 20, 23, 23, 22, 20],
                                                               [25, 25, 25, 25, 25, 25, 25, 25],
                                                               [25, 25, 25, 25, 25, 25, 25, 25],
                                                               [25, 25, 25, 25, 25, 25, 25, 25],
                                                               [25, 25, 25, 25, 25, 25, 25, 25]],
                                                              dtype=np.uint8)
        self.chrominance_quantization_table = self.default_chrominance_quantization_table

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
        self.chrominance_datatype = np.uint8

        self.default_zigzag_pattern = np.array([[0, 1, 5, 6, 14, 15, 27, 28],
                                                [2, 4, 7, 13, 16, 26, 29, 42],
                                                [3, 8, 12, 17, 25, 30, 41, 43],
                                                [9, 11, 18, 24, 31, 40, 44, 53],
                                                [10, 19, 23, 32, 39, 45, 52, 54],
                                                [20, 22, 33, 38, 46, 51, 55, 60],
                                                [21, 34, 37, 47, 50, 56, 59, 61],
                                                [35, 36, 48, 49, 57, 58, 62, 63]])


    def __call__(self, image, settings):
        """
        Implementation of JPEG-like compression with individual functions broken out.
        :param image_uncompressed:
        :param kwargs:
        :return:
        """
        #TODO implement quality factor
        #TODO look into how imread works

        if isinstance(image, str):
            image_uncompressed = io.imread(image)
            self.original_path = image
        else:
            image_uncompressed = image
        if isinstance(settings, str):
            with open(settings, 'r') as settings_file:
                settings = yaml.safe_load(settings_file)

        self.image_dimensions = image_uncompressed.shape[:2]
        self.channel_amount = image_uncompressed.shape[2]
        #print(self.image_dimensions)
        # Calculate total blocks per channel (must match decompression logic)
        self.chrominance_dimensions = (self.image_dimensions[0] // self.downsample_factor,
                                  self.image_dimensions[1] // self.downsample_factor)

        self.num_y_blocks = (self.image_dimensions[0] // self.block_size) * (self.image_dimensions[1] // self.block_size)
        self.num_c_blocks = (self.chrominance_dimensions[0] // self.block_size) * (self.chrominance_dimensions[1] // self.block_size)
        #num_total_blocks = num_y_blocks + 2 * num_c_blocks

        #TODO try to make it compatible - might not even be an issue?
        if self.image_dimensions[0] % self.block_size != 0 or self.image_dimensions[1] % self.block_size != 0:
            print('Warning! Image Dimensions not divisible by', {self.blocksize})

        if self.chrominance_dimensions[0] % self.block_size != 0 or self.chrominance_dimensions[1] % self.block_size != 0:
            print('Warning! Chromiance dimensions not divisible by', {self.blocksize})

        image_uncompressed = self.set_datatype_and_channels(image_uncompressed)
        YCbCrImage = self.convert_colorspace(image_uncompressed, **settings)
        downsampled_image = self.downsample_chrominance(YCbCrImage, **settings)
        block_processed_channels = self.process_blocks(downsampled_image, **settings)
        compressed_image_datastream, huffman_tables = self.entropy_encode(block_processed_channels)
        self.encode_to_file(compressed_image_datastream, huffman_tables, settings)

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
    #TODO perhaps establish vertical and horizontal downsampling instead of all as one
    def downsample_chrominance(self, YCbCr_image, **kwargs):
        """
        Apply downsampling factor to YCbCr formatted image and save the image as a tuple of 3 matricies with the
        luminance and two chrominance channels.
        :param YCbCr_image: The image already converted to YCbCr format
        :return: A Tuple, element 0 is the lumiance channel, element 1 is the chrominance channels
        """

        self.downsample_factor = kwargs.get("chrominance_downsample_factor", 4)

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
        :param downsampled_image: The output of the function downsample_chrominance
        :param kwargs:
        :return: The uncompressed converted and quantized matricies.
        """
        self.block_size = kwargs.get("block_size", 8)

        block_processed_channels = []
        for ch_num, channel in enumerate(downsampled_image):
            block_processed_channels.append(np.zeros(shape=np.shape(channel),dtype=np.int8))
            for idx in range(0,np.shape(channel)[0], self.block_size):
                for jdx in range(0, np.shape(channel)[1], self.block_size):
                    end_idx = idx + self.block_size if np.shape(channel)[0] - idx > self.block_size else None
                    end_jdx = jdx + self.block_size if np.shape(channel)[1] - jdx > self.block_size else None
                    image_block = channel[idx:end_idx, jdx:end_jdx]
                    frequency_block = self.block_DCT(image_block, **kwargs)
                    if idx == 0 and jdx == 0 and ch_num == 0:
                        print(block_processed_channels)
                    quantized_block = self.quantize_block(frequency_block, ch_num, **kwargs)
                    block_processed_channels[ch_num][idx:end_idx, jdx:end_jdx] = quantized_block
                    if idx == 0 and jdx == 0 and ch_num == 0:
                        print(block_processed_channels)
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

        def pad_matrix(matrix, target_rows, target_cols):
            """
            Pads a matrix with zeros on the bottom and right side to reach the target dimensions.

            :param matrix: 2D list or NumPy array
            :param target_rows: Desired number of rows
            :param target_cols: Desired number of columns
            :return: Padded NumPy array
            """
            matrix = np.array(matrix)
            original_rows, original_cols = matrix.shape

            if original_rows > target_rows or original_cols > target_cols:
                raise ValueError("Target dimensions must be greater than or equal to the original dimensions.")

            padded_matrix = np.zeros((target_rows, target_cols), dtype=matrix.dtype)
            padded_matrix[:original_rows, :original_cols] = matrix

            return padded_matrix

        def create_emphasis_matrix(self, emphasis_factor):
            matrix = np.zeros((self.block_size, self.block_size), dtype=np.float32)
            for i in range(self.block_size):
                for j in range(self.block_size):
                    # Calculate distance from top-left corner (0,0)
                    distance = np.sqrt(i * i + j * j)
                    # Base value that increases with distance
                    matrix[i, j] = max(1, np.floor(1/(self.block_size * self.block_size) * emphasis_factor * distance))
            return matrix

        #quality_factor = kwargs.get("quality_factor", 50)
        #ToDO implement emphasis matrix for chrominance or lumiance
        self.lumiance_quantization_table = kwargs.get("lumiance_quantization_table",
                                                      self.default_lumiance_quantization_table)

        padded_matrix = pad_matrix(frequency_domain_block, self.block_size, self.block_size)
        #print(padded_matrix)
        if ch_num == 0:
            padded_frequency_domain_matrix = (padded_matrix / self.lumiance_quantization_table).astype(np.int8)
        else:
            padded_frequency_domain_matrix = (padded_matrix / self.chrominance_quantization_table).astype(np.int8)
        # needs to be zero for RLE to be successful
        padded_frequency_domain_matrix[-1] = 0
        return padded_frequency_domain_matrix[0:frequency_domain_block.shape[0],0:frequency_domain_block.shape[1]]

    def entropy_encode(self, quantized_blocks, **kwargs):
        """
        Applies Delta encoding for DC coefficients, Zig-Zag ordering, Run-Length Encoding, and Huffman Encoding
        :param: quantized_blocks: The output of quantize_block
        :return: compressed bits, the huffman code
        """
        print('Entropy encoding started')
        print(' - Begin preliminary encoding')
        if self.block_size == 8:
            self.zigzag_pattern = self.default_zigzag_pattern
        else:
            self.zigzag_pattern = generate_zigzag_pattern(self.block_size)

        # Create separate DC and AC symbols for each channel type
        #Human vision is less sensitive to color variations than to brightness variations
        #so we can be more aggressive with compression in the chrominance channels
        # ac and dc use different distributions thus it makes sense to separate them
        # using different dictionaries is only beneficial as long as the quantization earlier is strong enough
        dc_lum_symbols = []  # DC coefficients for luminance (Y)
        dc_chrom_symbols = []  # DC coefficients for chrominance (Cb, Cr combined)
        ac_lum_symbols = []  # AC coefficients for luminance
        ac_chrom_symbols = []  # AC coefficients for chrominance

        # Track previous DC values separately for each channel
        prev_dc = [0, 0, 0]  # [Y, Cb, Cr]

        # Process blocks in decompression order: ALL Y -> ALL Cb -> ALL Cr
        all_blocks = []
        block_index = 0
        for channel_idx, channel in enumerate(quantized_blocks):
            # easiest fix to stop integer overflow later during the calculations
            channel = quantized_blocks[channel_idx].astype(np.int16)
            for i in range(0, channel.shape[0], self.block_size):
                for j in range(0, channel.shape[1], self.block_size):
                    block = self._get_padded_block(channel, i, j)
                    zigzagged = zigzag_order(block, self.zigzag_pattern)
                    #if block_index == 0:
                        #print(zigzagged)

                    # DC encoding (Y channel)
                    delta_dc = zigzagged[0] - prev_dc[channel_idx]
                    prev_dc[channel_idx] = zigzagged[0]

                    # AC encoding
                    rle_block = run_length_encoding(zigzagged[1:])
                    #adding both to symbols
                    if channel_idx == 0:
                        dc_lum_symbols.append(delta_dc)
                        ac_lum_symbols.extend(rle_block)
                    else:
                        dc_chrom_symbols.append(delta_dc)
                        ac_chrom_symbols.extend(rle_block)
                    combined = np.array([delta_dc] + rle_block, dtype=object)
                    all_blocks.append(combined)
                    block_index += 1

        print(' - Begin Huffman table building')

        def build_huffman_table(symbols):
            freq = Counter(symbols)
            tree = build_huffman_tree(freq)
            codes = generate_huffman_codes(tree)
            return codes

        # Build Huffman tables for each coefficient type
        # TODO make the values of the dicts not strings - probably not necessary as long as they also get printed
        dc_lum_codes = build_huffman_table(dc_lum_symbols)
        dc_chrom_codes = build_huffman_table(dc_chrom_symbols)
        ac_lum_codes = build_huffman_table(ac_lum_symbols)
        ac_chrom_codes = build_huffman_table(ac_chrom_symbols)
        # actual JPEG usually uses predefined Huffman codes - makes it quicker
        # Combine all Huffman tables
        huffman_tables = {
            'dc_lum': dc_lum_codes,  # DC luminance (Y)
            'dc_chrom': dc_chrom_codes,  # DC chrominance (Cb, Cr)
            'ac_lum': ac_lum_codes,  # AC luminance (Y)
            'ac_chrom': ac_chrom_codes  # AC chrominance (Cb, Cr)
        }

        # Encode all blocks using appropriate Huffman tables
        compressed_bits = []

        # Reset the delta DC tracking for encoding
        print(' - Begin Huffman encoding')

        block_index = 0
        for block in all_blocks:
            # Determine which channel we're processing based on block index
            if block_index < self.num_y_blocks:
                channel_idx = 0  # Y
            elif block_index < self.num_y_blocks + self.num_c_blocks:
                channel_idx = 1  # Cb
            else:
                channel_idx = 2  # Cr

            # Extract delta_dc and rle_block from the combined array
            delta_dc = block[0]
            rle_block = block[1:]

            # DC encoding
            if channel_idx == 0:
                dc_bits = dc_lum_codes.get(delta_dc, '')
            else:
                dc_bits = dc_chrom_codes.get(delta_dc, '')

            # AC encoding
            ac_bits = []
            for symbol in rle_block:
                if channel_idx == 0:
                    ac_bits.append(ac_lum_codes.get(symbol, ''))
                else:
                    ac_bits.append(ac_chrom_codes.get(symbol, ''))
            # Combine DC and AC bits
            compressed_bits.append(dc_bits + ''.join(ac_bits))
            block_index += 1

        print('Entropy encoding ended')
        return compressed_bits, huffman_tables


    def _get_padded_block(self, channel, i, j):
        """Helper to handle edge blocks with padding"""
        end_i = min(i + self.block_size, channel.shape[0])
        end_j = min(j + self.block_size, channel.shape[1])
        block = channel[i:end_i, j:end_j]

        if block.shape != (self.block_size, self.block_size):
            padded_block = np.zeros((self.block_size, self.block_size), dtype=block.dtype)
            padded_block[:block.shape[0], :block.shape[1]] = block
            return padded_block
        return block
    def encode_to_file(self, encoded_data_stream, huffman_tables, settings):
        """
        Turn the image matrices into a data stream. Pre-append the configurations necessary
        to reconstruct the full image.
        .rde filetype is our proprietary image compression format :)
        :param encoded_data_stream:
        :param huffman_table:
        :param settings:
        :return:
        """
        self.save_location = settings.get("save_location", "")
        if self.save_location == "":
            #self.save_location = os.path.join(os.getcwd(), "tmp", f"flex_jpeg_comp_output_{datetime.now().strftime("%Y%m%d_%H%M%S")}")
            self.save_location = os.path.join(os.getcwd(), "tmp",
                                              f"flex_jpeg_comp")

        # Set file paths for both formats
        self.binary_save_location = f"{self.save_location}.rde"
        self.debugging_save_location = f"{self.save_location}.verbose.rde"

        serializable_tables = make_serializable_table(huffman_tables)
        serializable_settings = {}
        for key, value in settings.items():
            if hasattr(value, 'tolist') and callable(getattr(value, 'tolist')):
                serializable_settings[key] = value.tolist()
            else:
                serializable_settings[key] = value

        # Process all bit strings into a single binary stream
        all_bits = "".join(encoded_data_stream)

        # Ensure the length is multiple of 8 for byte conversion
        padding_needed = 8 - (len(all_bits) % 8) if len(all_bits) % 8 != 0 else 0
        all_bits += '0' * padding_needed

        binary_data = bytearray()
        for i in range(0, len(all_bits), 8):
            byte = int(all_bits[i:i + 8], 2)  # Convert 8 bits to a byte
            binary_data.append(byte)

        # Create header as JSON for better parsing
        header = {
            "settings": serializable_settings,
            "huffman_tables": serializable_tables,
            "padding_bits": padding_needed,
            "image_dimensions": self.image_dimensions
        }
        header_json = json.dumps(header).encode('utf-8')

        # Write binary version
        with open(self.binary_save_location, 'wb') as binary_file:
            # Write header length as 4-byte integer (big-endian)
            binary_file.write(len(header_json).to_bytes(4, byteorder='big'))
            # Write header JSON
            binary_file.write(header_json)
            # Write binary data
            binary_file.write(binary_data)

        #does not include some overhead from signifiers
        compressed_size_theoretical = self.calculate_size(all_bits, serializable_tables, serializable_settings)
        print('Total theoretical size: ', compressed_size_theoretical, 'kB')

        compressed_size = os.path.getsize(self.binary_save_location)
        uncompressed_size = os.path.getsize(self.original_path)

        # For a standard RGB image:
        height, width = self.image_dimensions
        channels = self.channel_amount
        # assuming one byte per channel
        uncompressed_size_theoretical = height * width * channels

        # Calculate compression ratio
        # Or other way around?
        compression_ratio = compressed_size / uncompressed_size

        # Print results
        print('Theoretical compressed size: ', compressed_size_theoretical, 'kB')
        print(f'Actual compressed size: {compressed_size / 1024:.2f} kB')
        print(f'Theoretical uncompressed  size: {uncompressed_size_theoretical / 1024:.2f} kB')
        print(f'Actual uncompressed size: {uncompressed_size / 1024:.2f} kB')
        print(f'Compression ratio: {compression_ratio:.2f}:1')
        print(f'Space savings: {(1 - compression_ratio) * 100:.2f}%')

        #with open(self.debugging_save_location, 'w') as text_file:
        """
            text_file.write("image_dimensions :: ")
            text_file.write(str(self.image_dimensions))
            text_file.write(" :: settings_start :: ")
            text_file.write(str(serializable_settings))
            text_file.write(" :: settings_end :: ")
            text_file.write("huffman_table :: ")
            text_file.write(str(serializable_tables))
            text_file.write(" :: huffman_table_end :: ")
            text_file.write("bit_data :: ")
            text_file.write(str(all_bits))
            text_file.write(" :: image_end")
        """



    def calculate_size(self, all_bits, serialized_huffman_tables, serialized_settings):
        # doesn't include miniscule header information
        # Calculate encoded image size
        encoded_image_size = len(all_bits) / 8 / 1024  # Convert bits to KB

        # Calculate huffman tables size
        huffman_json = json.dumps(serialized_huffman_tables).encode('utf-8')
        # TODO use this at some point
        #huffman_compressed = gzip.compress(huffman_json)
        #huffman_tables_size = len(huffman_compressed) / 1024
        huffman_tables_size = len(huffman_json) / 1024

        # Calculate settings size
        settings_json = json.dumps(serialized_settings).encode('utf-8')
        settings_size = len(settings_json) / 1024

        total_size = encoded_image_size + huffman_tables_size + settings_size

        print(f"Encoded data: {encoded_image_size:.2f} kB")
        print(f"Huffman table: {huffman_tables_size:.2f} kB")
        print(f"Settings: {settings_size:.2f} kB")

        return round(total_size, 2)



""" 
Use this function block to test things out.
"""

compression_algorithm_reference = {
    "jpeg_baseline": BaselineJpeg,
    "homemade_jpeg_like": FlexibleJpeg
}


if __name__ == '__main__':

    #baseline_jpeg = BaselineJpeg(os.path.join(os.getcwd(),"compression_configurations", "baseline_jpeg_q100.yaml"))
    #start_time = time.time()

    flexible_jpeg = FlexibleJpeg()

    test_image_path = os.path.join(os.getcwd(), "assets", "unit_test_images", "gradients_16x32.tif")
    #test_image_path = os.path.join(os.getcwd(), "assets", "test_images", "landscape.png")

    compression_config = os.path.join(os.getcwd(),
                                              "compression_configurations",
                                              "homemade_compression_jpeg_like.yaml")
    flexible_jpeg(test_image_path, compression_config)




    #base_compression_config = os.path.join(os.getcwd(), "compression_configurations", "baseline_jpeg_q100.yaml")

    #baseline_jpeg(test_image_path)

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
