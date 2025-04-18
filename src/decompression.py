# This file to contain high-level functions consisting of a series of sub-routines
# defined in utilities.py

# For example:
# jpeg_decompression(compressed_file, compression_configuration)
#   load_jpeg(compressed_file)
#   perform_jpeg_decompression(compressed_image)
#   color_space_deconversion(uncompressed_image)

# Functions in this file should be composed of as few sub-routines as possible for
# readability and modularity.

# This file contains the decompression algorithm for FlexibleJpeg compression
# It implements the inverse operations of compression.py in reverse order
# This file contains the decompression functionality for various image compression formats
import os
import numpy as np
from fontTools.cffLib import packEncoding0
from scipy.fftpack import idct, idctn
import imageio.v3 as imageio
from skimage import io, color
import ast
from abc import ABC, abstractmethod
import re
import json
import struct
import cv2
from collections import Counter
#from skimage.color.colorconv import ycbcr_from_rgb
from src.utilities import parse_huffman_table, make_serializable_table, bytes_to_bools

from src.compression import BaselineJpeg, FlexibleJpeg
from src.huffman import generate_zigzag_pattern, inverse_zigzag_order

#makes it so it doesn't print
np.set_printoptions(formatter={
    'object': lambda x: str(x) if not isinstance(x, tuple) else f"({x[0]}, {x[1]})"
})


class DecompressImage(ABC):
    """
    Abstract base class for all decompression algorithms
    Provides a common interface and utility methods for all decompressors
    """

    def __init__(self, config=None):
        """Initialize with optional configuration"""
        self.config = config
        self.save_location = None

    @abstractmethod
    def __call__(self, **kwargs):
        """
        Main method to decompress an image
        :param kwargs: Additional parameters for decompression
        :return: Decompressed image and save path
        """
        pass

    @staticmethod
    def detect_compression_type(file_path):
        """
        Automatically detect compression type from file extension or contents
        :param file_path: Path to the compressed file
        :return: String identifier of the compression type
        """
        if file_path.endswith('.jpeg') or file_path.endswith('.jpg'):
            return "jpeg_baseline"
        elif file_path.endswith('.bin.rde'):
            return "homemade_jpeg_like"
        else:
            # Try to infer from file contents or return default
            return "unknown"

    @staticmethod
    def factory(compression_type=None, config=None):
        """
        Factory method to create the appropriate decompressor based on compression type
        :param compression_type: String identifier of the compression algorithm
        :param config: Configuration for the decompressor
        :return: Appropriate DecompressImage subclass instance
        """
        decompressor_map = {
            "jpeg_baseline": BaselineJpegDecompress,
            "homemade_jpeg_like": FlexibleJpegDecompress,
        }

        if compression_type in decompressor_map:
            return decompressor_map[compression_type](config)
        else:
            raise ValueError(f"Unsupported compression type: {compression_type}")

    def set_save_location(self, file_path, suffix="_decompressed"):
        """
        Set a default save location based on the input file
        :param file_path: Path to the input file
        :param suffix: Suffix to add to the output filename
        :return: Full path for saving the decompressed image
        """
        input_path, input_ext = os.path.splitext(file_path)
        if input_ext.lower() == '.rde':
            input_path = os.path.splitext(input_path)[0]  # Remove .bin part for .bin.rde files

        # Default to PNG for all decompressed outputs
        save_path = f"{input_path}{suffix}.png"
        self.save_location = save_path
        return save_path


class BaselineJpegDecompress(DecompressImage):
    """
    Decompressor for standard JPEG/JFIF files
    Uses imageio to handle the decompression as it's a well-established format
    """

    def __init__(self, config=None):
        super().__init__(config)

    def __call__(self, compressed_file=None, **kwargs):
        """
        Decompress a standard JPEG file
        :param compressed_file: Path to the JPEG file (optional)
        :param kwargs: Additional parameters
        :return: Decompressed image and save path
        """
        # Default file path if none provided
        if not compressed_file:
            compressed_file = os.path.join(os.getcwd(), "tmp", "decomp_test_base.jpeg")

        self.set_save_location(compressed_file)
        save_location = kwargs.get("save_location", self.save_location)

        decompressed_image = imageio.imread(compressed_file)
        if save_location:
            imageio.imwrite(save_location, decompressed_image)

        return decompressed_image, save_location


class FlexibleJpegDecompress(DecompressImage, FlexibleJpeg):
    """
    Decompressor for the custom FlexibleJpeg format
    Inherits from both DecompressImage and FlexibleJpeg to reuse compression parameters
    """
    def __init__(self, config=None):
        # Initialize both parent classes
        DecompressImage.__init__(self, config)
        FlexibleJpeg.__init__(self, config)

        # Initialize decompression-specific attributes
        self.image_dimensions = None
        self.upsample_factor = None
        self.quantization_tables_initialized = False

    def __call__(self, compressed_file=None, **kwargs):
        """
        Decompress a FlexibleJpeg file
        :param compressed_file: Path to the compressed file (.rde)
        :param kwargs: Additional parameters
        :return: Decompressed image and save path
        """
        self.set_save_location(compressed_file)
        # Override with user-provided save location if available
        save_location = kwargs.get("save_location", self.save_location)

        # Read and parse the compressed file
        bit_data, huffman_table, settings = self.decode_from_file(compressed_file)

        self._load_decompression_settings(settings)

        print(self.block_size)
        #exit()

        self.chrominance_dimensions = (np.floor(self.image_dimensions[0] / self.upsample_factor).astype(np.int16),
                                  np.floor(self.image_dimensions[1] / self.upsample_factor).astype(np.int16))

        # Inverse of compression basically
        quantized_blocks = self.entropy_decode(bit_data, huffman_table)
        unprocessed_blocks = self.process_blocks_inverse(quantized_blocks)
        upsampled_image = self.upsample_chrominance(unprocessed_blocks)
        rgb_image = self.convert_colorspace_inverse(upsampled_image)
        print(f"Successfully decompressed image")

        # Save the decompressed image
        if save_location:
            imageio.imwrite(save_location, rgb_image.astype(np.uint8))

        print(f"Decompressed image saved to: {save_location}")

        return rgb_image, save_location

    def _load_decompression_settings(self, settings):
        """
        Load and validate settings from the compressed file header
        :param settings: Dictionary of settings from the file header
        """
        # Convert string values if needed
        if isinstance(settings, str):
            settings = ast.literal_eval(settings)

        # Update core parameters
        self.block_size = settings.get("block_size", 8)
        self.downsample_factor = settings.get("chrominance_downsample_factor", 2)
        self.upsample_factor = self.downsample_factor  # For decompression

        # Load quantization tables
        if "luminance_quantization_table" in settings:
            self.luminance_quantization_table = np.array(
                settings["luminance_quantization_table"],
                dtype=np.uint8
            )
        else:
            #print("not in settings")
            self.luminance_quantization_table = self.default_luminance_quantization_table

        if "chrominance_quantization_table" in settings:
            self.chrominance_quantization_table = np.array(
                settings["chrominance_quantization_table"],
                dtype=np.uint8
            )
        else:
            #print("not in settings")
            self.chrominance_quantization_table = self.default_chrominance_quantization_table

        # Set zigzag pattern based on block size
        if self.block_size == 8:
            self.zigzag_pattern = self.default_zigzag_pattern
        else:
            self.zigzag_pattern = generate_zigzag_pattern(self.block_size)

        # Color conversion matrices
        self.YCbCr_conversion_matrix = np.array(
            settings.get("YCbCr_conversion_matrix", self.default_YCbCr_conversion_matrix),
            dtype=np.float32
        ) / 256

        self.YCbCr_conversion_offset = np.array(
            settings.get("YCbCr_conversion_offset", self.default_YCbCr_conversion_offset),
            dtype=np.uint8
        )

        self.quantization_tables_initialized = True

    def decode_from_file(self, file_path):
        """
        Read compressed file and extract the encoded data, huffman table, and settings
        :param file_path: Path to the compressed file
        :return: Tuple of (bit_data, huffman_table, settings)
        """
        print('File decoding process started')
        print('- Begin extraction')
        if file_path.endswith('.rde'):
            with open(file_path, 'rb') as file:
                # Read header length (first 4 bytes)
                header_length = int.from_bytes(file.read(4), byteorder='big')

                # Read header JSON
                header_json = file.read(header_length).decode('utf-8')
                header = json.loads(header_json)

                byte_data = file.read()
                bit_data = bytes_to_bools(byte_data, header['padding_bits'])

                # Get other components from header
                settings = header['settings']
                huffman_tables_raw = header['huffman_tables']
                huffman_tables = parse_huffman_table(huffman_tables_raw)
                self.image_dimensions = header['image_dimensions']

        print('- Evaluated succesfully')
        print('File decoding process ended')
        return bit_data, huffman_tables, settings

    def entropy_decode(self, compressed_bits, huffman_tables):
        """
        Decode the Huffman encoded bit stream
        :param compressed_bits: Compressed bit stream
        :param huffman_tables: Huffman codes dictionary
        :return: List of quantized blocks
        """
        print('Entropy decoding process started')
        print('- Begin separating and inverting huffman tables')

        reverse_huffman = {
            table_name: {v: k for k, v in table.items()}  # Invert key-value pairs
            for table_name, table in huffman_tables.items()
        }


        def get_min_max_key_lengths(reverse_huffman):
            # Get the lengths of all keys (as binary strings)
            key_lengths = []
            for key in reverse_huffman.keys():
                # Case 1: Key is an integer (e.g., 0b101, 5, etc.)
                if isinstance(key, int):
                    # If key is 0, binary length is 1 (special case)
                    if key == 0:
                        key_lengths.append(1)
                    else:
                        key_lengths.append(key.bit_length())
                # Case 2: Key is a binary string (e.g., "101")
                elif isinstance(key, str) and all(c in '01' for c in key):
                    key_lengths.append(len(key))
                # Case 3: Key is a string but not binary (unlikely in Huffman)
                elif isinstance(key, str):
                    print(f"Warning: Key '{key}' is not a binary string. Treating as length 1.")
                    key_lengths.append(1)
            return (min(key_lengths), max(key_lengths))

        dc_lum_min, dc_lum_max = get_min_max_key_lengths(reverse_huffman['dc_lum'])
        dc_chrom_min, dc_chrom_max = get_min_max_key_lengths(reverse_huffman['dc_chrom'])
        ac_lum_min, ac_lum_max = get_min_max_key_lengths(reverse_huffman['ac_lum'])
        ac_chrom_min, ac_chrom_max = get_min_max_key_lengths(reverse_huffman['ac_chrom'])


        #TODO fix it for times when image dimension mod upsample factor != 0, maybe in that case put it back to divisible number and account for that when upsampling again, check upsampling/downsampling funciton
        # check for that in comp too
        # skip the bs if it's all divisible

        #chrominance muss später korrekt upsampled werden
        # this includes non full blocks
        #TODO make sure your ceil function does not fuck up over floating point errors
        num_total_y_blocks = (np.ceil(self.image_dimensions[0] / self.block_size) * np.ceil(self.image_dimensions[1] / self.block_size)).astype(np.int16)
        num_total_c_blocks = (np.ceil(self.chrominance_dimensions[0] / self.block_size) * np.ceil(self.chrominance_dimensions[1] / self.block_size)).astype(np.int16)
        num_total_full_blocks = num_total_y_blocks + 2 * num_total_c_blocks

        #num_total_full_y_rows = np.floor(self.image_dimensions[0] / self.block_size).astype(np.int16)
        #num_total_full_y_cols = np.floor(self.image_dimensions[1] / self.block_size).astype(np.int16)
        #num_remaining_y_rows = self.image_dimensions[0] % self.block_size
        #num_remaining_y_cols = self.image_dimensions[1] % self.block_size

        #num_total_full_c_rows = np.floor(self.chrominance_dimensions[0] / self.block_size).astype(np.int16)
        #num_total_full_c_cols = np.floor(self.chrominance_dimensions[1] / self.block_size).astype(np.int16)
        #num_remaining_c_rows = self.chrominance_dimensions[0] % self.block_size
        #num_remaining_c_cols = self.chrominance_dimensions[1] % self.block_size

        # what does it mean full? like the block has all rows
        #print(num_total_full_y_rows)
        #print(num_total_full_y_cols)
        #print(num_remaining_y_rows)
        #print(num_remaining_y_cols)
        #print(num_total_full_c_rows)
        #print(num_total_full_c_cols)
        #print(num_remaining_c_rows)
        #print(num_remaining_c_cols)

        decoded_blocks = [np.empty(self.image_dimensions, dtype=np.int16),
                          np.empty(self.chrominance_dimensions, dtype=np.int16),
                          np.empty(self.chrominance_dimensions, dtype=np.int16)]

        prev_dc_coeff = [0, 0, 0]
        current_pos = 0  # Current bit position in the array
        bit_length = len(compressed_bits)

        print('- Begin decoding the blocks')


        for block_num in range(num_total_full_blocks):
            # Determine which channel we're processing
            if block_num < num_total_y_blocks:
                channel_idx = 0  # Y
            elif block_num < num_total_y_blocks + num_total_c_blocks:
                channel_idx = 1  # Cb
            else:
                channel_idx = 2  # Cr
            # Get the appropriate reverse tables
            if channel_idx == 0:  # Luminance (Y)
                dc_table = reverse_huffman['dc_lum']
                ac_table = reverse_huffman['ac_lum']
                min_dc_length = dc_lum_min
                max_dc_length = dc_lum_max
                min_ac_length = ac_lum_min
                max_ac_length = ac_lum_max

            else:  # Chrominance (Cb or Cr)
                dc_table = reverse_huffman['dc_chrom']
                ac_table = reverse_huffman['ac_chrom']
                min_dc_length = dc_chrom_min
                max_dc_length = dc_chrom_max
                min_ac_length = ac_chrom_min
                max_ac_length = ac_chrom_max

            # Decode DC coefficient
            delta_dc = None
            # Search for matching DC code
            for l in range(min_dc_length, min(max_dc_length + 1, bit_length - current_pos + 1)):
                # Build the code by checking bits directly
                code = ''.join(str(int(bit)) for bit in compressed_bits[current_pos:current_pos + l])
                if code in dc_table:
                    delta_dc = int(dc_table[code])
                    current_pos += l
                    dc_coeff = delta_dc + prev_dc_coeff[channel_idx]
                    prev_dc_coeff[channel_idx] = dc_coeff
                    break

            #FUCK INDENTATION
            if delta_dc is None:
                raise ValueError(f"Invalid bitstream: No matching DC code found at position {current_pos}")

            # Decode AC coefficients
            ac_coeffs = []
            eob_found = False

            while not eob_found and len(ac_coeffs) < (self.block_size * self.block_size - 1):
                # Search for matching AC code
                ac_symbol = None
                for l in range(min_ac_length, min(max_ac_length + 1, bit_length - current_pos + 1)):
                    code = ''.join(str(int(bit)) for bit in compressed_bits[current_pos:current_pos + l])
                    if code in ac_table:
                        ac_symbol = ac_table[code]
                        current_pos += l
                        break

                if ac_symbol is None:
                    raise ValueError(f"Invalid bitstream: No matching AC code found at position {current_pos}")
                if ac_symbol == (0, 0):  # End of block
                    eob_found = True
                else:
                    # Extract run length and value correctly
                    # The AC symbols are stored as tuples (value, run_length) - matching the encoding
                    ac_value, run_length = ac_symbol
                    # Add zeros for the run length
                    ac_coeffs.extend([0] * run_length)
                    ac_coeffs.append(ac_value)
                    # Pad with zeros if EOB was reached early
            ac_coeffs.extend([0] * (self.block_size * self.block_size - 1 - len(ac_coeffs)))
            zigzag_coeffs = [dc_coeff] + ac_coeffs
            block = inverse_zigzag_order(zigzag_coeffs, self.zigzag_pattern, self.block_size)
            #if block_num == 10:
                #print(block)
                #print(len(block))

            # Get the channel dimensions
            if channel_idx == 0:
                num_rows, num_cols = self.image_dimensions
            else:
                num_rows, num_cols = self.chrominance_dimensions

            # Calculate block position in the channel
            if channel_idx == 0:
                block_idx = block_num
            elif channel_idx == 1:
                block_idx = block_num - num_total_y_blocks
            else:
                block_idx = block_num - num_total_y_blocks - num_total_c_blocks

            blocks_per_row = np.ceil(num_cols / self.block_size).astype(np.int16)
            block_row = (block_idx // blocks_per_row)
            block_col = (block_idx % blocks_per_row)
            i =  block_row * self.block_size
            j = block_col * self.block_size

            # Place the block in the appropriate channel
            end_i = min(i + self.block_size, num_rows)
            end_j = min(j + self.block_size, num_cols)
            decoded_blocks[channel_idx][i:end_i, j:end_j] = block[:end_i - i, :end_j - j]

        print('Entropy decoding completed')
        #print(decoded_blocks[2])
        return decoded_blocks

    def process_blocks_inverse(self, quantized_blocks):
        """
        Reconstructs full image channels (Y, Cb, Cr) by dequantizing and applying 2D IDCT on blocks,
        then reassembling them using the header's dimensions.
        Parameters: quantized_blocks (list): List of quantized 2D DCT blocks.
        Returns: list: [y_channel, cb_channel, cr_channel] as NumPy arrays.
        """
        reconstructed_channels = []
        for ch_num, channel in enumerate(quantized_blocks):
            reconstructed_channels.append(np.zeros(shape=np.shape(channel), dtype=np.uint8))
            for idx in range(0, np.shape(channel)[0], self.block_size):
                for jdx in range(0, np.shape(channel)[1], self.block_size):
                    end_idx = idx + self.block_size if np.shape(channel)[0] - idx > self.block_size else None
                    end_jdx = jdx + self.block_size if np.shape(channel)[1] - jdx > self.block_size else None
                    quantized_block = channel[idx:end_idx, jdx:end_jdx]
                    dequantized_block = self.dequantize_block(quantized_block, ch_num)
                    spatial_block = self.inverse_block_DCT(dequantized_block)
                    reconstructed_channels[ch_num][idx:end_idx, jdx:end_jdx] = spatial_block
        return reconstructed_channels

    def dequantize_block(self, quantized_block, ch_num, **kwargs):
        """
        Multiply each block by the quantization table (reverse of quantize_block).
        :param quantized_block: The quantized block from the compressed data
        :param ch_num: Indicates which of the quantization tables to use
        :param kwargs:
        :return: The dequantized frequency domain block
        """
        original_shape = quantized_block.shape
        # Pad to block_size if needed
        if original_shape != (self.block_size, self.block_size):
            padded_block = np.zeros((self.block_size, self.block_size), dtype=quantized_block.dtype)
            padded_block[:original_shape[0], :original_shape[1]] = quantized_block
            quantized_block = padded_block

        if ch_num == 0:
            dequantized_block = quantized_block * self.luminance_quantization_table
        else:
            dequantized_block = quantized_block * self.chrominance_quantization_table

        if original_shape != (self.block_size, self.block_size):
            dequantized_block = dequantized_block[:original_shape[0], :original_shape[1]]

        return dequantized_block.astype(np.float32)

    def inverse_block_DCT(self, frequency_block, **kwargs):
        """
        Perform inverse DCT on the block and scale pixels back to 0-255 range.
        :param frequency_block: The frequency domain block to be transformed back
        :param kwargs:
        :return: The spatial domain image block
        """
        inverse_DCT = np.clip(idctn(frequency_block, norm='ortho') + 128, 0, 255).astype(np.uint8)
        return inverse_DCT



    def upsample_chrominance(self, channels):
        """
        Upsample chrominance channels to match luminance channel dimensions
        :param channels: List of [Y, Cb, Cr] channels
        :return: YCbCr image with full resolution chrominance channels
        """
        print('upsample factor:', self.upsample_factor)

        Y = channels[0]

        # TODO something other than nearest neighbor?
        # Upsample chrominance channels using nearest-neighbor interpolation
        if self.upsample_factor != 0:
            Cb = np.repeat(np.repeat(channels[1], self.upsample_factor, axis=0),
                       self.upsample_factor, axis=1)
            Cr = np.repeat(np.repeat(channels[2], self.upsample_factor, axis=0),
                       self.upsample_factor, axis=1)
        else:
            Cb = channels[1]
            Cr = channels[2]

        # Ensure upsampled chrominance matches luminance dimensions
        #h, w = Y.shape
        #Cb = Cb[:h, :w]  # Trim if necessary
        #Cr = Cr[:h, :w]
        if Y.shape != Cb.shape != Cr.shape:
            raise ValueError(f"Upsampled Chromiance doesn't have the right dimensions")

        YCbCr_image = np.dstack((Y, Cb, Cr)).astype(np.uint8)
        return YCbCr_image


    def convert_colorspace_inverse(self, ycbcr_image, **kwargs):
        """
        Convert from YCbCr to RGB color space
        This is the inverse of the convert_colorspace method in FlexibleJpeg
        :param ycbcr_image: YCbCr image
        :return: RGB image
        """
        #print(ycbcr_image[:,:,2])
        #print(ycbcr_image[:,:,0])
        rgb_restored = cv2.cvtColor(ycbcr_image, cv2.COLOR_YCrCb2RGB)
        #print(rgb_restored[:,:,0])
        final = np.clip(rgb_restored, 0, 255)
        return final


# Dictionary mapping compression types to their decompressor classes
compression_algorithm_reference = {
    "jpeg_baseline": BaselineJpegDecompress,
    "homemade_jpeg_like": FlexibleJpegDecompress
}

# Example usage - Simple script for decompressing a specific file
if __name__ == '__main__':
    # Create a FlexibleJpegDecompress instance
    test_image_path = os.path.join(os.getcwd(), "tmp", "flex_jpeg_comp.rde")
    #test_image_path = os.path.join(os.getcwd(), "tmp", "flex_jpeg_comp.verbose.rde")

    # Decompress the image from the fixed path
    decompressor = FlexibleJpegDecompress()
    decompressed_image, save_path = decompressor(test_image_path)