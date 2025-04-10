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
from src.utilities import display_greyscale_image
from src.huffman import generate_zigzag_pattern, zigzag_order, run_length_encoding, build_huffman_tree, generate_huffman_codes, huffman_encode
from collections import Counter
import pkgutil
import json

import time #for debugging purposes


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
        # Represent as 8-bit unsigned int.
        if isinstance(image, str):
            image_uncompressed = io.imread(image)
        else:
            image_uncompressed = image
        if isinstance(settings, str):
            with open(settings, 'r') as settings_file:
                settings = yaml.safe_load(settings_file)

        self.image_dimensions = image_uncompressed.shape[:2]
        print(self.image_dimensions)

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
                    end_idx = idx + self.block_size if np.shape(channel)[0] - idx > self.block_size else -1
                    end_jdx = jdx + self.block_size if np.shape(channel)[1] - jdx > self.block_size else -1
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
        #ToDO create emphasis matrix for chrominance or lumiance

        self.lumiance_quantization_table = kwargs.get("lumiance_quantization_table",
                                                      self.default_lumiance_quantization_table)

        padded_matrix = pad_matrix(frequency_domain_block, self.block_size, self.block_size)
        if ch_num == 0:
            padded_frequency_domain_matrix = (padded_matrix/self.lumiance_quantization_table).astype(np.int8)
        else:
            padded_frequency_domain_matrix = (padded_matrix / self.chrominance_quantization_table).astype(np.int8)
        return padded_frequency_domain_matrix[0:frequency_domain_block.shape[0],0:frequency_domain_block.shape[1]]

    def entropy_encode(self, quantized_blocks, **kwargs):
        """
        Applies Delta encoding for DC coefficients, Zig-Zag ordering, Run-Length Encoding, and Huffman Encoding
        :param: quantized_blocks: The output of quantize_block
        :return: compressed bits, the huffman code
        """
        #elapsed = time.time() - start_time
        print('Entropy encoding started')
        #print(' - Begin preliminary encoding ({self.elapsed:.3f}s')
        print(' - Begin preliminary encoding')
        if self.block_size == 8:
            self.zigzag_pattern = self.default_zigzag_pattern
        else:
            self.zigzag_pattern = generate_zigzag_pattern(self.block_size)

        # Create separate DC and AC symbols for each channel type
        #Human vision is less sensitive to color variations than to brightness variations
        #so we can be more aggressive with compression in the chrominance channels
        # ac and dc use different distributions thus it makes sense to separate them
        dc_lum_symbols = []  # DC coefficients for luminance (Y)
        dc_chrom_symbols = []  # DC coefficients for chrominance (Cb, Cr combined)
        ac_lum_symbols = []  # AC coefficients for luminance
        ac_chrom_symbols = []  # AC coefficients for chrominance

        # Store encoded data by channel and block
        encoded_blocks = []

        #quantized_blocks[0] = np.zeros(self.image_dimensions)
        # Track previous DC coefficients for delta encoding (separate for each channel type)
        prev_dc_lum = 0  # For luminance
        prev_dc_chrom = [0, 0]  # For Cb and Cr

        for channel_idx, channel in enumerate(quantized_blocks):
            # Process each block in the channel
            for i in range(0, channel.shape[0], self.block_size):
                for j in range(0, channel.shape[1], self.block_size):
                    # Extract the block
                    end_i = min(i + self.block_size, channel.shape[0])
                    end_j = min(j + self.block_size, channel.shape[1])
                    block = channel[i:end_i, j:end_j]

                    # Handle non-standard sized blocks at edges
                    if block.shape[0] != self.block_size or block.shape[1] != self.block_size:
                        padded_block = np.zeros((self.block_size, self.block_size), dtype=block.dtype)
                        padded_block[:block.shape[0], :block.shape[1]] = block
                        block = padded_block

                    # Cast to int16 to avoid overflow
                    block = block.astype(np.int16)

                    zigzagged = zigzag_order(block, self.zigzag_pattern)
                    # Get DC coefficient (first value in zigzagged data)
                    dc_coeff = zigzagged[0]

                    # Apply delta encoding to DC coefficient
                    # purpose is to create more lower absolute values
                    if channel_idx == 0:  # Luminance (Y)
                        delta_dc = dc_coeff - prev_dc_lum
                        prev_dc_lum = dc_coeff
                        dc_lum_symbols.append(delta_dc)
                    else:  # Chrominance (Cb or Cr)
                        chrom_idx = channel_idx - 1  # 0 for Cb, 1 for Cr
                        delta_dc = dc_coeff - prev_dc_chrom[chrom_idx]
                        prev_dc_chrom[chrom_idx] = dc_coeff
                        dc_chrom_symbols.append(delta_dc)

                    # Process AC coefficients (all except DC)
                    ac_coeffs = zigzagged[1:]
                    rle_block = run_length_encoding(ac_coeffs)

                    # Store the RLE-encoded AC coefficients based on channel type
                    if channel_idx == 0:  # Luminance
                        ac_lum_symbols.extend(rle_block)
                    else:  # Chrominance
                        ac_chrom_symbols.extend(rle_block)

                    # Add to our collection of encoded blocks
                    encoded_blocks.append((channel_idx, rle_block))

        print(' - Begin Huffman table building')
        # Build Huffman tables for each coefficient type
        # DC Luminance
        # see if one can't use signs for this
        dc_lum_freq = Counter(dc_lum_symbols)
        dc_lum_tree = build_huffman_tree(dc_lum_freq)
        dc_lum_codes = generate_huffman_codes(dc_lum_tree)

        # DC Chrominance (Cb, Cr combined)
        dc_chrom_freq = Counter(dc_chrom_symbols)
        dc_chrom_tree = build_huffman_tree(dc_chrom_freq)
        dc_chrom_codes = generate_huffman_codes(dc_chrom_tree)

        # AC Luminance
        ac_lum_freq = Counter(ac_lum_symbols)
        ac_lum_tree = build_huffman_tree(ac_lum_freq)
        ac_lum_codes = generate_huffman_codes(ac_lum_tree)

        # AC Chrominance (Cb, Cr combined)
        ac_chrom_freq = Counter(ac_chrom_symbols)
        ac_chrom_tree = build_huffman_tree(ac_chrom_freq)
        ac_chrom_codes = generate_huffman_codes(ac_chrom_tree)

        #actual JPEG usually uses predefined Huffman codes
        # apparently leaving out the zero counter doesn't change the final encoding data size

        # Combine all Huffman tables
        huffman_tables = {
            'dc_lum': dc_lum_codes,  # DC luminance (Y)
            'dc_chrom': dc_chrom_codes,  # DC chrominance (Cb, Cr)
            'ac_lum': ac_lum_codes,  # AC luminance (Y)
            'ac_chrom': ac_chrom_codes  # AC chrominance (Cb, Cr)
        }

        #print(huffman_tables)
        # Encode all blocks using appropriate Huffman tables
        compressed_bits = []

        # Reset the delta DC tracking for encoding
        prev_dc_lum = 0
        prev_dc_chrom = [0, 0]
        print(' - Begin encoding')

        for channel_idx, channel in enumerate(quantized_blocks):
            for i in range(0, channel.shape[0], self.block_size):
                for j in range(0, channel.shape[1], self.block_size):
                    # Extract block
                    end_i = min(i + self.block_size, channel.shape[0])
                    end_j = min(j + self.block_size, channel.shape[1])
                    block = channel[i:end_i, j:end_j]

                    if block.shape[0] != self.block_size or block.shape[1] != self.block_size:
                        padded_block = np.zeros((self.block_size, self.block_size), dtype=block.dtype)
                        padded_block[:block.shape[0], :block.shape[1]] = block
                        block = padded_block

                    block = block.astype(np.int16)

                    # Apply zigzag ordering
                    zigzagged = zigzag_order(block, self.zigzag_pattern)

                    # Get and encode DC coefficient
                    dc_coeff = zigzagged[0]

                    if channel_idx == 0:  # Luminance (Y)
                        delta_dc = dc_coeff - prev_dc_lum
                        prev_dc_lum = dc_coeff
                        dc_bits = huffman_tables['dc_lum'].get(delta_dc, '')
                    else:  # Chrominance (Cb or Cr)
                        chrom_idx = channel_idx - 1
                        delta_dc = dc_coeff - prev_dc_chrom[chrom_idx]
                        prev_dc_chrom[chrom_idx] = dc_coeff
                        dc_bits = huffman_tables['dc_chrom'].get(delta_dc, '')

                    # Process AC coefficients
                    ac_coeffs = zigzagged[1:]
                    rle_block = run_length_encoding(ac_coeffs)

                    ac_bits = ""
                    for symbol in rle_block:
                        if channel_idx == 0:  # Luminance (Y)
                            ac_code = huffman_tables['ac_lum'].get(symbol, '')
                        else:  # Chrominance (Cb or Cr)
                            ac_code = huffman_tables['ac_chrom'].get(symbol, '')
                        ac_bits += ac_code

                    # Combine DC and AC bits
                    block_bits = dc_bits + ac_bits
                    compressed_bits.append(block_bits)
        #compressed_bits contains the blocks each - to string shows that they are with commata
        return compressed_bits, huffman_tables

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
            self.save_location = os.path.join(os.getcwd(),
                                                     "tmp",
                                                     f"flex_jpeg_comp_output_{datetime.now().strftime("%Y%m%d_%H%M%S")}")

        # Set file paths for both formats
        self.binary_save_location = f"{self.save_location}.bin.rde"
        self.text_save_location = f"{self.save_location}.txt.rde"

        #huffman_table_str = str({str(k): v for k, v in huffman_tables.items()})

        serializable_tables = {}
        for table_name, table in huffman_tables.items():
            serializable_tables[table_name] = {}
            for key, value in table.items():
                # Handle different key types
                if isinstance(key, tuple):
                    # Convert tuple with numpy integers to a string representation
                    serialized_key = str(tuple(int(x) if hasattr(x, 'item') else x for x in key))
                    serializable_tables[table_name][serialized_key] = value
                elif hasattr(key, 'item'):  # Handle numpy integer types
                    serializable_tables[table_name][str(int(key))] = value
                else:
                    serializable_tables[table_name][str(key)] = value

        with open("huffman_tables_comp.json", "w") as f:
            json.dump(serializable_tables, f, indent=2)

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

        #print(len(all_bits)/8)
        #how_many = Counter(str(all_bits))
        #print(how_many)

        binary_data = bytearray()
        for i in range(0, len(all_bits), 8):
            byte = int(all_bits[i:i + 8], 2)  # Convert 8 bits to a byte
            binary_data.append(byte)


        # Convert to hex representation for cleaner viewing
        #hex_representation = binary_data.hex()

        # Create header as JSON for better parsing
        # useful for txt as well?
        header = {
            "settings": serializable_settings,
            "huffman_tables": serializable_tables,
            "padding_bits": padding_needed
        }
        #TODO use this for txt as well?
        header_json = json.dumps(header).encode('utf-8')

        # Write to file with clear separator between header and binary data

        #TODO maybe you don't need wb
        with open(self.binary_save_location, 'wb') as binary_file:
            # Write header length as 4-byte integer
            header_length = len(header_json)
            #shows the length of it
            binary_file.write(header_length.to_bytes(4, byteorder='big'))
            binary_file.write(header_json)
            #binary_file.write(binary_data)

        with open(self.text_save_location, 'w') as text_file:
            #first won't include itself
            text_file.write("theoretical_size :: ")
            text_file.write(str(self.calculate_size(all_bits, serializable_tables, serializable_settings)))
            text_file.write(" kB :: ")
            text_file.write(" :: image_dimensions :: ")
            text_file.write(str(self.image_dimensions))
            text_file.write(" :: settings_start :: ")
            # this or settings?
            text_file.write(str(serializable_settings))
            text_file.write(" :: settings_end :: ")
            text_file.write("huffman_table :: ")
            text_file.write(str(serializable_tables))
            text_file.write(" :: huffman_table_end :: ")
            text_file.write("bit_data :: ")
            #use all bits rather than the string
            text_file.write(str(all_bits))
            text_file.write(" :: image_end")

    def calculate_size(self, all_bits, serialized_huffman_tables, serialized_settings):
        # doesn't include miniscule header information
        # Calculate encoded image size
        encoded_image_size = len(all_bits) / 8 / 1024  # Convert bits to KB

        # Calculate huffman tables size
        huffman_json = json.dumps(serialized_huffman_tables).encode('utf-8')
        # TODO why are we using this here?
        #huffman_compressed = gzip.compress(huffman_json)
        #huffman_tables_size = len(huffman_compressed) / 1024
        huffman_tables_size = len(huffman_json) / 1024

        # Calculate settings size
        settings_json = json.dumps(serialized_settings).encode('utf-8')
        settings_size = len(settings_json) / 1024

        total_size = encoded_image_size + huffman_tables_size + settings_size

        print(f"Encoded data: {encoded_image_size:.2f} KB")
        print(f"Huffman table: {huffman_tables_size:.2f} KB")
        print(f"Settings: {settings_size:.2f} KB")

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

    test_image_path = os.path.join(os.getcwd(), "assets", "test_images", "landscape.png")
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
