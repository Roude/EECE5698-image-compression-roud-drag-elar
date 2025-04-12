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
from scipy.fftpack import idct
import imageio.v3 as imageio
import ast
from abc import ABC, abstractmethod
import re
import json
import struct
from collections import Counter
from src.utilities import parse_huffman_table, make_serializable_table

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

        # Rename downsampling_factor to upsampling_factor
        # TODO Needed?
        self.image_dimensions = None
        self.upsampling_factor = self.downsample_factor

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

        # Update settings from the compressed file
        if isinstance(settings, str):
            settings = ast.literal_eval(settings)

        # Update configuration using the parent class's method
        self.update_configuration(settings)

        # Rename downsampling_factor to upsampling_factor for clarity
        # TODO DOUBLE?
        self.upsampling_factor = settings.get("chrominance_downsample_factor", self.downsample_factor)

        # Set zigzag pattern based on block size
        if self.block_size == 8:
            self.zigzag_pattern = self.default_zigzag_pattern
        else:
            from src.huffman import generate_zigzag_pattern
            self.zigzag_pattern = generate_zigzag_pattern(self.block_size)

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

    def decode_from_file(self, file_path):
        """
        Read compressed file and extract the encoded data, huffman table, and settings
        :param file_path: Path to the compressed file
        :return: Tuple of (bit_data, huffman_table, settings)
        """
        print('File decoding process started')
        print('- Begin extraction')

        with open(file_path, 'r') as file:
            content = file.read()

            dims_start = content.find("image_dimensions :: ") + len("image_dimensions :: ")
            dims_end = content.find(" :: settings_start")
            dims = content[dims_start:dims_end]
            # Extract settings
            settings_start = content.find("settings_start :: ") + len("settings_start :: ")
            settings_end = content.find(" :: settings_end")
            settings_str = content[settings_start:settings_end]

            # Extract Huffman table
            huffman_start = content.find("huffman_table :: ") + len("huffman_table :: ")
            huffman_end = content.find(" :: huffman_table_end")
            huffman_table_str = content[huffman_start:huffman_end]

            # Extract bit data
            bit_data_start = content.find("bit_data :: ") + len("bit_data :: ")
            bit_data_end = content.find(" :: image_end")

            print('- Extracted succesfully')

            if bit_data_end == -1:  # If "image_end" marker not found
                bit_data_str = content[bit_data_start:]
            else:
                bit_data_str = content[bit_data_start:bit_data_end]

            # Process settings to a proper Python dictionary
            settings = ast.literal_eval(settings_str)
            huffman_tables = parse_huffman_table(huffman_table_str)
            bit_data = np.array([bool(int(c)) for c in bit_data_str], dtype=np.bool)
            self.image_dimensions = ast.literal_eval(dims)

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
        #with open("huffman_tables_decomp_reverse.json", "w") as f:
            #json.dump(reverse_huffman, f, indent=2)

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

        #print(reverse_huffman['ac_chrom'])

        # with open("huffman_tables_decomp.json", "w") as f:
        # json.dump(huffman_tables, f, indent=2)

        chrominance_dimensions = (self.image_dimensions[0] // self.downsample_factor,
                             self.image_dimensions[1] // self.downsample_factor)
        num_total_y_blocks = (self.image_dimensions[0] // self.block_size) * (
                    self.image_dimensions[1] // self.block_size)
        num_total_c_blocks = (chrominance_dimensions[0] // self.block_size) * (
                    chrominance_dimensions[1] // self.block_size)
        num_total_blocks = num_total_y_blocks + 2 * num_total_c_blocks

        decoded_blocks = [np.empty(self.image_dimensions, dtype=np.int16),
                          np.empty(chrominance_dimensions, dtype=np.int16),
                          np.empty(chrominance_dimensions, dtype=np.int16)]

        prev_dc_coeff = [0, 0, 0]
        current_pos = 0  # Current bit position in the array
        bit_length = len(compressed_bits)

        for block_num in range(num_total_blocks):
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
                rows, cols = self.image_dimensions
            else:
                rows, cols = chrominance_dimensions

            # Calculate block position in the channel
            if channel_idx == 0:
                block_idx = block_num
            elif channel_idx == 1:
                block_idx = block_num - num_total_y_blocks
                #print(block_idx)
            else:
                block_idx = block_num - num_total_y_blocks - num_total_c_blocks

            #ToDo check if they are not exactly divisible, does this get unnecessarily repeated, init before the loop
            #TODO what happens when image resolution and chromiance resolution are not divisible by block_size
            blocks_per_row = cols // self.block_size
            i = (block_idx // blocks_per_row) * self.block_size
            j = (block_idx % blocks_per_row) * self.block_size

            # Place the block in the appropriate channel
            end_i = min(i + self.block_size, rows)
            end_j = min(j + self.block_size, cols)
            decoded_blocks[channel_idx][i:end_i, j:end_j] = block[:end_i - i, :end_j - j]

        print('Entropy decoding completed')
        print(decoded_blocks[2])
        return decoded_blocks


    #TODO check order, we want the reverse pipeline
    def process_blocks_inverse(self, quantized_blocks):
        """
        Process blocks in reverse order: dequantize and perform inverse DCT
        :param quantized_blocks: List of quantized blocks
        :return: List of channels (Y, Cb, Cr)
        """
        # Determine dimensions for reconstruction
        # This is an estimation, actual dimensions would be stored in the compressed file
        block_count = len(quantized_blocks)
        blocks_per_channel = block_count // 3  # Assuming 3 channels: Y, Cb, Cr

        # Estimate dimensions based on block count and block size
        # This is a simplification; actual implementation would use dimensions from the file
        blocks_per_row = int(np.sqrt(blocks_per_channel))
        height = blocks_per_row * self.block_size
        width = blocks_per_row * self.block_size

        # Initialize channels
        y_channel = np.zeros((height, width), dtype=np.int8)
        cb_channel = np.zeros((height // self.upsampling_factor, width // self.upsampling_factor), dtype=np.int8)
        cr_channel = np.zeros((height // self.upsampling_factor, width // self.upsampling_factor), dtype=np.int8)

        # Process Y channel blocks
        block_idx = 0
        for i in range(0, height, self.block_size):
            for j in range(0, width, self.block_size):
                if block_idx < blocks_per_channel:
                    block = quantized_blocks[block_idx]
                    dequantized_block = self.dequantize_block(block, 0)  # 0 for Y channel
                    idct_block = self.block_idct(dequantized_block)

                    # Handle boundary conditions
                    y_h = min(self.block_size, height - i)
                    y_w = min(self.block_size, width - j)
                    y_channel[i:i + y_h, j:j + y_w] = idct_block[:y_h, :y_w]

                    block_idx += 1

        # Process Cb channel blocks
        cb_height = height // self.upsampling_factor
        cb_width = width // self.upsampling_factor
        for i in range(0, cb_height, self.block_size):
            for j in range(0, cb_width, self.block_size):
                if block_idx < 2 * blocks_per_channel:
                    block = quantized_blocks[block_idx]
                    dequantized_block = self.dequantize_block(block, 1)  # 1 for Cb channel
                    idct_block = self.block_idct(dequantized_block)

                    # Handle boundary conditions
                    cb_h = min(self.block_size, cb_height - i)
                    cb_w = min(self.block_size, cb_width - j)
                    cb_channel[i:i + cb_h, j:j + cb_w] = idct_block[:cb_h, :cb_w]

                    block_idx += 1

        # Process Cr channel blocks
        cr_height = height // self.upsampling_factor
        cr_width = width // self.upsampling_factor
        for i in range(0, cr_height, self.block_size):
            for j in range(0, cr_width, self.block_size):
                if block_idx < 3 * blocks_per_channel:
                    block = quantized_blocks[block_idx]
                    dequantized_block = self.dequantize_block(block, 2)  # 2 for Cr channel
                    idct_block = self.block_idct(dequantized_block)

                    # Handle boundary conditions
                    cr_h = min(self.block_size, cr_height - i)
                    cr_w = min(self.block_size, cr_width - j)
                    cr_channel[i:i + cr_h, j:j + cr_w] = idct_block[:cr_h, :cr_w]

                    block_idx += 1

        return [y_channel, cb_channel, cr_channel]

    def dequantize_block(self, quantized_block, channel_num):
        """
        Dequantize a block by multiplying by the quantization table
        This is the inverse of the quantize_block method in FlexibleJpeg
        :param quantized_block: Quantized DCT block
        :param channel_num: Channel number (0=Y, 1=Cb, 2=Cr)
        :return: Dequantized block
        """

        # Reuse the pad_matrix helper function from the parent class if available
        def pad_matrix(matrix, target_rows, target_cols):
            matrix = np.array(matrix)
            original_rows, original_cols = matrix.shape

            if original_rows > target_rows or original_cols > target_cols:
                raise ValueError("Target dimensions must be greater than or equal to the original dimensions.")

            padded_matrix = np.zeros((target_rows, target_cols), dtype=matrix.dtype)
            padded_matrix[:original_rows, :original_cols] = matrix

            return padded_matrix

        # Select the appropriate quantization table
        if channel_num == 0:  # Luminance channel
            qtable = self.lumiance_quantization_table
        else:  # Chrominance channels
            qtable = self.chrominance_quantization_table

        # Ensure quantization table matches block size
        if qtable.shape[0] < self.block_size or qtable.shape[1] < self.block_size:
            qtable = pad_matrix(qtable, self.block_size, self.block_size)

        # Create output block matching input dimensions
        dequantized_block = np.zeros(quantized_block.shape, dtype=np.float32)

        # Dequantize by multiplying with the quantization table
        for i in range(min(quantized_block.shape[0], self.block_size)):
            for j in range(min(quantized_block.shape[1], self.block_size)):
                dequantized_block[i, j] = quantized_block[i, j] * qtable[i, j]

        return dequantized_block

    def block_idct(self, dct_block):
        """
        Apply inverse DCT to a block
        This is the inverse of the block_dct method in FlexibleJpeg
        :param dct_block: DCT coefficients block
        :return: Spatial domain block
        """
        # Apply inverse DCT
        idct_block = idct(dct_block)

        # Shift back from -128 to 127 range to 0 to 255 range
        # This reverses the shift done in block_dct
        idct_block = np.round(idct_block + 128).astype(np.uint8)

        return idct_block

    def upsample_chrominance(self, channels):
        """
        Upsample chrominance channels to match luminance channel dimensions
        :param channels: List of [Y, Cb, Cr] channels
        :return: YCbCr image with full resolution chrominance channels
        """
        y_channel, cb_channel, cr_channel = channels
        height, width = y_channel.shape

        # Create upsampled channels
        cb_upsampled = np.zeros((height, width), dtype=np.uint8)
        cr_upsampled = np.zeros((height, width), dtype=np.uint8)

        # Simple nearest neighbor upsampling
        for i in range(height):
            for j in range(width):
                cb_upsampled[i, j] = cb_channel[i // self.upsampling_factor, j // self.upsampling_factor]
                cr_upsampled[i, j] = cr_channel[i // self.upsampling_factor, j // self.upsampling_factor]

        # Combine channels
        ycbcr_image = np.stack((y_channel, cb_upsampled, cr_upsampled), axis=-1)

        return ycbcr_image

    def convert_colorspace_inverse(self, ycbcr_image):
        """
        Convert from YCbCr to RGB color space
        This is the inverse of the convert_colorspace method in FlexibleJpeg
        :param ycbcr_image: YCbCr image
        :return: RGB image
        """
        # Define the inverse of the YCbCr conversion matrix
        if self.YCbCr_conversion_matrix is not None:
            # Convert from numpy array if it's a string representation
            if isinstance(self.YCbCr_conversion_matrix, str):
                self.YCbCr_conversion_matrix = np.array(ast.literal_eval(self.YCbCr_conversion_matrix))

            # Calculate inverse matrix
            inverse_matrix = np.linalg.inv(self.YCbCr_conversion_matrix)
        else:
            # Use default values
            inverse_matrix = np.linalg.inv(self.default_YCbCr_conversion_matrix)

        # Get offset
        if self.YCbCr_conversion_offset is not None:
            if isinstance(self.YCbCr_conversion_offset, str):
                self.YCbCr_conversion_offset = np.array(ast.literal_eval(self.YCbCr_conversion_offset))
            offset = self.YCbCr_conversion_offset
        else:
            offset = self.default_YCbCr_conversion_offset

        # Subtract offset and apply inverse matrix
        shifted_ycbcr = ycbcr_image.astype(np.float32) - offset
        rgb_image = np.tensordot(shifted_ycbcr, inverse_matrix, axes=([2], [1]))

        # Clip values to valid range and convert to uint8
        rgb_image = np.clip(rgb_image, 0, 255).astype(np.uint8)

        return rgb_image


# Dictionary mapping compression types to their decompressor classes
compression_algorithm_reference = {
    "jpeg_baseline": BaselineJpegDecompress,
    "homemade_jpeg_like": FlexibleJpegDecompress
}

# Example usage - Simple script for decompressing a specific file
if __name__ == '__main__':
    # Create a FlexibleJpegDecompress instance
    decompressor = FlexibleJpegDecompress()
    test_image_path = os.path.join(os.getcwd(), "tmp", "flex_jpeg_compverbose.rde")

    # Decompress the image from the fixed path
    decompressed_image, save_path = decompressor(test_image_path)