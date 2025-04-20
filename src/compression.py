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
from yaml import safe_load
from skimage import io, color
import imageio.v3 as imageio
import os
from datetime import datetime
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from skimage.color import convert_colorspace
from skimage.io import imread
from scipy.fftpack import dct, idct, dctn,idctn
from torch.nn.functional import channel_shuffle

from src.utilities import display_greyscale_image, make_serializable_table, gaussian_matrix, ln_norm
from src.huffman import generate_zigzag_pattern, zigzag_order, run_length_encoding, build_huffman_tree, generate_huffman_codes, huffman_encode
from collections import Counter
import pkgutil
import json
import cv2
import rawpy

import time
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
            print("No configuration initially provided")
        else:
            with open(config,'r') as config_file:
                self.config = yaml.load(config_file, yaml.SafeLoader)
            self.update_configuration(self.config)

    def update_configuration(self, config):
        if isinstance(config, str):
            with open(config,'r') as config_file:
                self.config = yaml.load(config_file, yaml.SafeLoader)
        else:
            self.config = config
        #print(self.config)

    def set_datatype_and_channels(self, image_uncompressed):
        if image_uncompressed.dtype != np.uint8:
            image_uncompressed = (255 * (image_uncompressed / np.max(image_uncompressed.dtype))).astype(np.uint8)

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
        self.block_size = 8
        #self.quality_factor = 50
        #self.chrominance_aggression_factor = 3
        #self.luminance_aggression_factor = 1.5
        self.downsample_factor = 2
        self.YCbCr_conversion_matrix = None
        self.YCbCr_conversion_offset = None
        self.zigzag_pattern = None
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

        self.default_luminance_quantization_table = np.array([[6, 4, 4, 6, 10, 16, 20, 24],
                                                              [5, 5, 6, 8, 10, 23, 24, 22],
                                                              [6, 5, 6, 10, 16, 23, 28, 22],
                                                              [6, 7, 9, 12, 20, 35, 32, 25],
                                                              [7, 9, 15, 22, 27, 44, 41, 31],
                                                              [10, 14, 22, 26, 32, 42, 45, 37],
                                                              [20, 26, 31, 35, 41, 48, 48, 40],
                                                              [29, 37, 38, 39, 45, 40, 41, 40]],
                                                             dtype=np.uint8)
        self.luminance_quantization_table = self.default_luminance_quantization_table
        self.luminance_datatype = np.uint8
        self.chrominance_datatype = np.uint8

        self.default_zigzag_pattern = np.array([[0, 1, 5, 6, 14, 15, 27, 28],
                                                [2, 4, 7, 13, 16, 26, 29, 42],
                                                [3, 8, 12, 17, 25, 30, 41, 43],
                                                [9, 11, 18, 24, 31, 40, 44, 53],
                                                [10, 19, 23, 32, 39, 45, 52, 54],
                                                [20, 22, 33, 38, 46, 51, 55, 60],
                                                [21, 34, 37, 47, 50, 56, 59, 61],
                                                [35, 36, 48, 49, 57, 58, 62, 63]])
        # If config was provided, load it immediately
        if config:
            self._load_config(config)

    #put into superclass?
    @staticmethod
    def read_raw_image(filepath):
        """Helper method to read RAW image files"""
        with rawpy.imread(filepath) as raw:
            rgb = raw.postprocess()
        return rgb

    def _load_config(self, config):
        """Load and validate configuration parameters"""
        try:
            if isinstance(config, str):
                print(f"Loading config from: {config}")
                if not os.path.exists(config):
                    raise FileNotFoundError(f"Config file not found at {config}")

                with open(config, 'r') as f:
                    config = yaml.safe_load(f)
                    print("Successfully loaded config!")
                    #print(json.dumps(config, indent=2))  # Pretty print

                if not config:  # Empty dict
                    raise ValueError("Loaded config is empty!")

        except Exception as e:
            print(f"Error loading config: {e}")
            raise

        # Update core parameters
        self.block_size = config.get("block_size", self.block_size)
        self.downsample_factor = config.get("chrominance_downsample_factor", self.downsample_factor)

        # Keep as such for now
        self.YCbCr_conversion_matrix = self.default_YCbCr_conversion_matrix
        self.YCbCr_conversion_offset = self.default_YCbCr_conversion_offset

        # Generate/update dependent parameters
        if self.block_size == 8:
            self.zigzag_pattern = self.default_zigzag_pattern
        # We use the generated ones for 8 too as it makes it more comparable
        self.zigzag_pattern = generate_zigzag_pattern(self.block_size)
        self.luminance_quantization_table, self.chrominance_quantization_table = (self.generate_parameter_matrices())


    def __call__(self, image, settings=None, **kwargs):
        """
        Implementation of JPEG-like compression with individual functions broken out.
        :param image_uncompressed:
        :param kwargs:
        :return:
        """
        self.timings = {}
        self.last_time = time.time()

        self.save_location = kwargs.get("save_location",
                                        os.path.join(os.getcwd(),
                                                     "tmp", f"flex_jpeg_comp")) # define save_location here to avoid AttributeError in encode_to_file(). this ensures we can either pass a custom save path or fall back to the default inside that method

        #TODO implement quality factor
        if isinstance(image, str):
            if image.lower().endswith(('.cr2', '.nef', '.arw')):
                import rawpy
                with rawpy.imread(image) as raw:
                    image_uncompressed = raw.postprocess()
            else:
                image_uncompressed = io.imread(image)
        else:
            image_uncompressed = image


        if isinstance(image, str):
            self.original_path = image
            if image.lower().endswith(('.cr2', '.nef', '.arw', '.dng')):
                image_uncompressed = self.read_raw_image(image)
            else:
                image_uncompressed = io.imread(image)
        else:
            image_uncompressed = image

        if settings is not None:
            self._load_config(settings)
        else:
            settings = self.config

        #TODO was there a reason we didn't use it for homemade JPEG
        image_uncompressed = self.set_datatype_and_channels(image_uncompressed)

        #TODO are we going to change the quantization table with different quality factors? if so just multiply by a scalar or this?
        # how about making a plot or something regarding the distribution of values in the frequency space
        # please find generator function below
        '''
         def create_emphasis_matrix(self, emphasis_factor):
            matrix = np.zeros((self.block_size, self.block_size), dtype=np.float32)
            for i in range(self.block_size):
                for j in range(self.block_size):
                    # Calculate distance from top-left corner (0,0)
                    distance = np.sqrt(i * i + j * j)
                    # Base value that increases with distance
                    matrix[i, j] = max(1, np.floor(1/(self.block_size * self.block_size) * emphasis_factor * distance))
            return matrix
        '''

        self.image_dimensions = image_uncompressed.shape[:2]
        self.channel_amount = image_uncompressed.shape[2]

        #self.chrominance_dimensions = (self.image_dimensions[0] // self.downsample_factor, self.image_dimensions[1] // self.downsample_factor)

        if self.downsample_factor == 1:
            self.chrominance_dimensions = self.image_dimensions
        else:
            self.chrominance_dimensions = (np.ceil(self.image_dimensions[0] / self.downsample_factor).astype(np.uint16),
                                  np.ceil(self.image_dimensions[1] / self.downsample_factor).astype(np.uint16))


        self.num_y_blocks = (np.ceil(self.image_dimensions[0] / self.block_size)).astype(np.uint32) * (np.ceil(
                    self.image_dimensions[1] / self.block_size)).astype(np.uint32)
        self.num_c_blocks = (np.ceil(self.chrominance_dimensions[0] / self.block_size)).astype(np.uint32) * (np.ceil(
                    self.chrominance_dimensions[1] / self.block_size)).astype(np.uint32)

        self.num_total_blocks = self.num_y_blocks + 2 * self.num_c_blocks

        #I know that this makes it unreadable but bear with me for the time being
        self.timings['preliminaries_ms'] = int((time.time() - self.last_time) * 1000)
        self.last_time = time.time()  # Reset the stopwatch

        #### THE PIPELINE ####
        image_uncompressed = self.set_datatype_and_channels(image_uncompressed)
        self.timings['set_datatype_and_channels_ms'] = int((time.time() - self.last_time) * 1000)
        self.last_time = time.time()
        YCbCrImage = self.convert_colorspace(image_uncompressed)
        self.timings['convert_colorspace_ms'] = int((time.time() - self.last_time) * 1000)
        self.last_time = time.time()
        downsampled_image = self.downsample_chrominance(YCbCrImage)
        self.timings['downsample_chrominance_ms'] = int((time.time() - self.last_time) * 1000)
        self.last_time = time.time()
        block_processed_channels = self.process_blocks(downsampled_image)
        self.timings['process_blocks_ms'] = int((time.time() - self.last_time) * 1000)
        self.last_time = time.time()
        compressed_image_datastream, huffman_tables = self.entropy_encode(block_processed_channels)
        self.timings['entropy_encode_ms'] = int((time.time() - self.last_time) * 1000)
        self.last_time = time.time()
        self.encode_to_file(compressed_image_datastream, huffman_tables, settings)
        return self.binary_save_location

    #TODO come up with something better for this is provisional
    def generate_parameter_matrices(self):
        luminance_qtable = np.zeros((self.block_size, self.block_size), dtype=np.uint8)
        chrominance_qtable = np.zeros((self.block_size, self.block_size), dtype=np.uint8)
        #factor = 4
        factor = np.sqrt(self.block_size)
        for i in range(self.block_size):
            for j in range(self.block_size):
                distance = np.sqrt(i * i + j * j)
                if distance == 0: distance = 1
                luminance_qtable[i, j] = max(1, np.floor(1 * factor * distance))
                chrominance_qtable[i, j] = max(1, np.floor(1.5 * factor * distance))
        return luminance_qtable, chrominance_qtable

    def convert_colorspace(self, image_uncompressed):
        """
        Apply YCbCr color conversion to image
        :param image_uncompressed:
        :param kwargs:
        :return:
        """
        ycbcr_image = cv2.cvtColor(image_uncompressed, cv2.COLOR_RGB2YCrCb)
        return ycbcr_image


    #TODO perhaps establish vertical and horizontal downsampling instead of all as one
    def downsample_chrominance(self, YCbCr_image):
        """
        Apply downsampling factor to YCbCr formatted image and save the image as a tuple of 3 matricies with the
        luminance and two chrominance channels.
        :param YCbCr_image: The image already converted to YCbCr format
        :return: A Tuple, element 0 is the lumiance channel, element 1 is the chrominance channels
        """
        print('Downsample factor:', self.downsample_factor)

        luminance = YCbCr_image[:, :, 0]  # No need to convert dtype if input is already uint8
        Cb = YCbCr_image[:, :, 1]
        Cr = YCbCr_image[:, :, 2]

        if self.downsample_factor == 1:
            return [luminance, Cb, Cr]

        running_average = np.zeros((self.chrominance_dimensions[0],
                                    self.chrominance_dimensions[1], 2), dtype=np.float32)

        # Count how many samples contribute to each output pixel (for correct averaging)
        count = np.zeros((self.chrominance_dimensions[0], self.chrominance_dimensions[1]), dtype=np.float32)

        for idx in range(self.downsample_factor):
            for jdx in range(self.downsample_factor):
                slice_h = slice(idx, None, self.downsample_factor)
                slice_w = slice(jdx, None, self.downsample_factor)

                # Extract the current block of chroma values
                chroma_block = YCbCr_image[slice_h, slice_w, 1:]  # Shape: (H_block, W_block, 2)

                # Accumulate contributions
                h, w = chroma_block.shape[:2]
                running_average[:h, :w, :] += chroma_block.astype(np.float32)
                count[:h, :w] += 1

        # Compute the average (avoid division by zero)
        running_average[:, :, 0] = np.divide(running_average[:, :, 0], count, where=count != 0)
        running_average[:, :, 1] = np.divide(running_average[:, :, 1], count, where=count != 0)

        return [
            luminance,
            np.array(running_average[:, :, 0], dtype=np.uint8),
            np.array(running_average[:, :, 1], dtype=np.uint8)
        ]

    def process_blocks(self, downsampled_image):
        """
        Perform compression steps that are executed on individual blocks of a user defined size.
        :param downsampled_image: The output of the function downsample_chrominance
        :param kwargs:
        :return: The uncompressed converted and quantized matricies.
        """
        #self.block_size = kwargs.get("block_size", 8)
        block_processed_channels = []
        for ch_num, channel in enumerate(downsampled_image):
            block_processed_channels.append(np.zeros(shape=np.shape(channel),dtype=np.int16))
            for idx in range(0,np.shape(channel)[0], self.block_size):
                for jdx in range(0, np.shape(channel)[1], self.block_size):
                    end_idx = min(idx + self.block_size, channel.shape[0])
                    end_jdx = min(jdx + self.block_size, channel.shape[1])

                    image_block = channel[idx:end_idx, jdx:end_jdx]
                    frequency_block = self.block_DCT(image_block)
                    quantized_block = self.quantize_block(frequency_block, ch_num)
                    block_processed_channels[ch_num][idx:end_idx, jdx:end_jdx] = quantized_block
                    #if idx == 0 and jdx == 0 and ch_num == 0:
                        #print(frequency_block)
        return block_processed_channels

    # might be cleaner to have them as submethods
    def block_DCT(self, image_block):
        """
        First scale pixels from 0 to 255 to -128 to 128, then perform DCT on the block
        :param image_block: An image block of a flexible size, for the DCT to be performed on.
        :param kwargs:
        :return: the cosine transformed block
        """
        dct = dctn(image_block.astype(np.float32) - 128, norm='ortho')
        return dct


    def quantize_block(self, frequency_domain_block, ch_num):
        """
        Divide each block by the quantization table (passed as a keyword argument or the default).
        Round to the nearest integer
        :param frequency_domain_block: The output of block_dct
        :param ch_num: automatically passed in by process_blocks. Indicates which of the quantization tables to use.
        :param kwargs:
        :return: the quantized block
        """
        def quantize_function(f_domain_block, config):
            func_name = config["function"]
            if func_name == 'Quarter Gauss':
                return np.round(f_domain_block/ gaussian_matrix(f_domain_block.shape,
                                                                config["max_quantization"],
                                                                config["standard_dev"]))
            elif func_name == 'LN':
                return np.round(f_domain_block / ln_norm(f_domain_block.shape,
                                                         config["N"],
                                                         config["max_val"],
                                                         config["min_val"]))
            elif func_name == 'Basic':
                return np.round(f_domain_block / config["quantization_table"])
            else:
                raise NotImplementedError("Only Basic Quantization Tables, L-N Norm and Quarter Gauss Quantization functions have been implemented.")

        original_shape = frequency_domain_block.shape
        # Pad to block_size if needed
        if original_shape != (self.block_size, self.block_size):
            padded_block = np.zeros((self.block_size, self.block_size), dtype=frequency_domain_block.dtype)
            padded_block[:original_shape[0], :original_shape[1]] = frequency_domain_block
            frequency_domain_block = padded_block

        if ch_num == 0:
            quantized_block = quantize_function(frequency_domain_block, self.config["chromiance_quantization"])
        else:
            quantized_block = quantize_function(frequency_domain_block, self.config["luminance_quantization"])
        # needs to be zero for RLE to be successful
        quantized_block[-1, -1] = 0

        if original_shape != (self.block_size, self.block_size):
            quantized_block = quantized_block[:original_shape[0], :original_shape[1]]

        return quantized_block.astype(np.int16)

    def entropy_encode(self, quantized_blocks, **kwargs):
        """
        Applies Delta encoding for DC coefficients, Zig-Zag ordering, Run-Length Encoding, and Huffman Encoding
        :param: quantized_blocks: The output of quantize_block
        :return: compressed bits, the huffman code
        """
        print('Entropy encoding started')
        print(' - Begin preliminary encoding')


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
        #print(len(quantized_blocks))
        # Process blocks in decompression order: ALL Y -> ALL Cb -> ALL Cr
        all_blocks = []
        block_index = 0
        for channel_idx, channel in enumerate(quantized_blocks):
            # easiest fix to stop integer overflow later during the calculations
            channel = quantized_blocks[channel_idx].astype(np.int16)
            # we go left to right downwards
            for i in range(0, channel.shape[0], self.block_size):
                for j in range(0, channel.shape[1], self.block_size):
                    end_i = min(i + self.block_size, channel.shape[0])
                    end_j = min(j + self.block_size, channel.shape[1])
                    block = channel[i:end_i, j:end_j]

                    #print(block.shape)


                    #if block.shape[0] != self.block_size and channel_idx == 0:
                        #print(block_index - 58*(87+1))
                    #if block.shape[1] != self.block_size and channel_idx == 0:
                        #print((block_index + 1) % 88)

                    if block.shape != (self.block_size, self.block_size):
                        padded_block = np.zeros((self.block_size, self.block_size), dtype=block.dtype)
                        padded_block[:block.shape[0], :block.shape[1]] = block
                        block = padded_block

                    #block = self._get_padded_block(channel, i, j)
                    zigzagged = zigzag_order(block, self.zigzag_pattern)
                    #if block_index == 0:
                        #print(zigzagged)

                    # DC encoding (Y channel)
                    delta_dc = zigzagged[0] - prev_dc[channel_idx]
                    prev_dc[channel_idx] = zigzagged[0]

                    # AC encoding
                    rle_block = run_length_encoding(zigzagged[1:])
                    if len(rle_block) > self.block_size **2:
                        raise ValueError
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
        #print(block_index)
        #print(self.num_total_blocks)
        print(' - Begin Huffman table building')

        def build_huffman_table(symbols):
            freq = Counter(symbols)
            tree = build_huffman_tree(freq)
            codes = generate_huffman_codes(tree)
            return codes

        # Build Huffman tables for each coefficient type
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
        if settings is None:
            settings = {}

        # Update settings with current quantization tables
        settings.update({
            "luminance_quantization_table": self.luminance_quantization_table.tolist(),
            "chrominance_quantization_table": self.chrominance_quantization_table.tolist(),
        })

        # Set file paths for both formats
        self.binary_save_location = f"{self.save_location}.rde"

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
            binary_file.write(header_json)
            binary_file.write(binary_data)

        self.timings['encode_to_file_ms'] = int((time.time() - self.last_time) * 1000)
        self.timings['total_compression_time_ms'] = sum(self.timings.values())

        # Calculate size breakdowns (rounded to 3 decimal places)
        encoded_image_size = round(len(all_bits) / 8 / 1024, 3)  # in KB

        huffman_json = json.dumps(serializable_tables).encode('utf-8')
        huffman_tables_size = round(len(huffman_json) / 1024, 3)

        settings_json = json.dumps(serializable_settings).encode('utf-8')
        settings_size = round(len(settings_json) / 1024, 3)
        header_size = round(len(header_json) / 1024, 3)

        total_theoretical_size = round(encoded_image_size + header_size + 4 / 1024, 3)  # 4 bytes converted to KB

        # Actual file sizes (rounded)
        compressed_size = round(os.path.getsize(self.binary_save_location) / 1024, 3)
        uncompressed_size = round(os.path.getsize(self.original_path) / 1024, 3)

        algorithm = self.__class__.__name__
        # Image dimensions
        height, width = self.image_dimensions
        channels = self.channel_amount
        uncompressed_size_theoretical = round(height * width * channels / 1024, 3)
        # Compression metrics
        compression_ratio = round(compressed_size / uncompressed_size_theoretical, 3)
        space_savings = round((1 - compression_ratio) * 100, 1)
        bits_per_pixel = round((compressed_size * 8192) / (width * height), 3)  # 1024*8=8192 bits per KB
        bits_per_pixel_uncompressed = round((uncompressed_size * 8192) / (width * height), 3)  # 1024*8=8192 bits per KB
        # Create comprehensive metrics dictionary
        metrics = {
            # Size breakdown components
            'compressed_size_components': {
                'encoded_image_data_kb': encoded_image_size,
                'huffman_tables_kb': huffman_tables_size,
                'compression_settings_kb': settings_size,
                'header_overhead_kb': header_size,
                'total_theoretical_kb': total_theoretical_size,
                'actual_compressed_kb': compressed_size,
            },
            # Actual file sizes
            'uncompressed_sizes': {
                'actual_uncompressed_kb': uncompressed_size,
                'theoretical_uncompressed_kb': uncompressed_size_theoretical,
            },
            'algorithm_metrics': {
                'algorithm': algorithm,
                'channels': channels,
                'image_dimension': self.image_dimensions,
            },
            'time_metrics': self.timings,
            'compression_metrics': {
                'compression_ratio': compression_ratio,
                'space_savings_percent': space_savings,
                'bits_per_pixel': bits_per_pixel,
                'bits_per_pixel_uncompressed': bits_per_pixel_uncompressed,
            },
            # Image information
            'image_info': {
                'original_path': self.original_path,
                'compressed_path': self.binary_save_location,
            },
            'quality_settings': serializable_settings,
        }

        # Print results in a clean, aligned format
        print("\n=== Compression Metrics ===")
        print(f"{'Metric':<35} {'Value':>15}")
        print("-" * 50)
        print(f"{'Original image path:':<35} {self.original_path:>15}")
        print(f"{'Compressed file path:':<35} {self.binary_save_location:>15}")
        print(f"{'Image dimensions:':<35} {f'{width}x{height}':>15}")
        print(f"{'Color channels:':<35} {channels:>15}")
        print("-" * 50)
        print(f"{'Compression algorithm':<35} {algorithm:>15}")
        print(f"{'Compression Time (ms)':<35} {self.timings['total_compression_time_ms']:>15}")
        print("-" * 50)
        print(f"{'Encoded image data:':<35} {encoded_image_size:>15,.3f} KB")
        print(f"{'Huffman tables size:':<35} {huffman_tables_size:>15,.3f} KB")
        print(f"{'Compression settings:':<35} {settings_size:>15,.3f} KB")
        print(f"{'Header overhead:':<35} {header_size:>15,.3f} KB")
        print(f"{'Total theoretical size:':<35} {total_theoretical_size:>15,.3f} KB")
        print(f"{'Actual compressed size:':<35} {compressed_size:>15,.3f} KB")
        print("-" * 50)
        print(f"{'Actual uncompressed size:':<35} {uncompressed_size:>15,.3f} KB")
        print(f"{'Theoretical uncompressed:':<35} {uncompressed_size_theoretical:>15,.3f} KB")
        print("-" * 50)
        print(f"{'Compression ratio:':<35} {compression_ratio:>15,.3f}")
        print(f"{'Space savings:':<35} {space_savings:>15,.1f}%")
        print(f"{'Bits per pixel:':<35} {bits_per_pixel:>15,.3f}")
        print(f"{'Bits per pixel (uncompressed):':<35} {bits_per_pixel_uncompressed:>15,.3f}")
        print("=" * 50)

        metrics_paths = f"{self.save_location}.metrics.json"
        try:
            with open(metrics_paths, 'w') as f:  # Fixed - use metrics_paths directly
                json.dump(metrics, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save metrics files: {str(e)}")
            metrics['metrics_files'] = {'error': str(e)}

        # Print save locations
        print("\nSaved metrics files: in", metrics_paths)


if __name__ == '__main__':

    #baseline_jpeg = BaselineJpeg(os.path.join(os.getcwd(),"compression_configurations", "baseline_jpeg_q100.yaml"))

    flexible_jpeg = FlexibleJpeg()

    #test_image_path = os.path.join(os.getcwd(), "assets", "unit_test_images", "white_16x16.tif")
    #test_image_path = os.path.join(os.getcwd(), "assets", "test_images", "landscape.png")
    test_image_path = os.path.join(os.getcwd(), "assets", "test_images", "Polar_bear_over_water.webp")
    #test_image_path = os.path.join(os.getcwd(), "assets", "test_images", "20241017-elarbi-bladeeNYC-4B5A2603.cr2")


    compression_config = os.path.join(os.getcwd(),
                                              "compression_configurations",
                                              "homemade_compression_gauss.yaml")
    flexible_jpeg = FlexibleJpeg(compression_config)
    flexible_jpeg(test_image_path)

