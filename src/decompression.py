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

from src.compression import BaselineJpeg, FlexibleJpeg
from src.huffman import generate_zigzag_pattern


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
        self.upsampling_factor = settings.get("chromiance_downsample_factor", self.downsample_factor)

        # Set zigzag pattern based on block size
        if self.block_size == 8:
            self.zigzag_pattern = self.default_zigzag_pattern
        else:
            from src.huffman import generate_zigzag_pattern
            self.zigzag_pattern = generate_zigzag_pattern(self.block_size)

        # Inverse of compression basically
        quantized_blocks = self.entropy_decode(bit_data, huffman_table)
        unprocessed_blocks = self.process_blocks_inverse(quantized_blocks)
        upsampled_image = self.upsample_chromiance(unprocessed_blocks)
        rgb_image = self.convert_colorspace_inverse(upsampled_image)

        # Save the decompressed image
        if save_location:
            imageio.imwrite(save_location, rgb_image.astype(np.uint8))

        return rgb_image, save_location

    def decode_from_file(self, file_path):
        """
        Read compressed file and extract the encoded data, huffman table, and settings
        :param file_path: Path to the compressed file
        :return: Tuple of (bit_data, huffman_table, settings)
        """
        # Determine which file format to use (text or binary)
        # Todo immediately transfer the arrays? no file reading?
        # TODO try to make it simpler on the compression side
        # get this shit working, perhaps with the txt easier?
        if file_path.endswith('.bin.rde'):
            with open(file_path, 'rb') as file:
                # Read entire file as bytes first
                content = file.read()

                # Find the boundary between header and binary data
                header_end = content.find(b"\n", content.find(b"padding_bits"))

                # Process the header part as text
                header_text = content[:header_end].decode('utf-8', errors='replace')

                # Extract settings
                settings_start = header_text.find("settings_start :: ") + len("settings_start :: ")
                settings_end = header_text.find(" :: settings_end")
                settings = header_text[settings_start:settings_end]

                # Extract Huffman table
                huffman_start = header_text.find("huffman_table :: ") + len("huffman_table :: ")
                huffman_end = header_text.find(" :: huffman_table_end")
                huffman_table_str = header_text[huffman_start:huffman_end]

                try:
                    huffman_table = ast.literal_eval(huffman_table_str)
                except (SyntaxError, ValueError) as e:
                    # Handle potential syntax errors in the Huffman table
                    print(f"Error parsing Huffman table: {e}")
                    # debugging: create a simplified dictionary if parsing fails
                    huffman_table = {}

                # Extract padding information
                padding_start = header_text.find("padding_bits :: ") + len("padding_bits :: ")
                padding_end = header_text.find(" ::", padding_start)
                padding = int(header_text[padding_start:padding_end])

                # Extract binary data (everything after the header)
                binary_data = content[header_end + 1:]

                # Find position of the image_end marker
                image_end_pos = binary_data.find(b" :: image_end")
                if image_end_pos != -1:
                    binary_data = binary_data[:image_end_pos]

                # Convert binary to bit string
                bit_data = ""
                for byte in binary_data:
                    bit_data += format(byte, '08b')

                # Remove padding
                if padding > 0:
                    bit_data = bit_data[:-padding]

        else:
            raise ValueError(f"Unsupported file format: {file_path}")

        return bit_data, huffman_table, settings

    def entropy_decode(self, compressed_bits, huffman_codes):
        """
        Decode the Huffman encoded bit stream
        :param compressed_bits: Compressed bit stream
        :param huffman_codes: Huffman codes dictionary
        :return: List of quantized blocks
        """
        # Inverse the huffman codes for decoding
        reverse_huffman = {code: symbol for symbol, code in huffman_codes.items()}

        # Decode the bit stream
        decoded_blocks = []
        current_block = []

        # Handle bit_data based on whether it's a single string or a list of strings
        if isinstance(compressed_bits, list):
            # Process each encoded block separately
            for block_bits in compressed_bits:
                block_data = self.huffman_decode_block(block_bits, reverse_huffman)
                decoded_blocks.append(block_data)
        else:
            # Assume it's a single bit string containing all blocks
            current_code = ""
            for bit in compressed_bits:
                current_code += bit
                if current_code in reverse_huffman:
                    symbol = reverse_huffman[current_code]
                    current_block.append(symbol)
                    current_code = ""

                    # Check if we've reached the end of a block (EOB marker)
                    if symbol == (0, 0):
                        decoded_blocks.append(current_block)
                        current_block = []

        # Convert RLE blocks back to quantized blocks
        quantized_blocks = self.rle_decode_blocks(decoded_blocks)
        return quantized_blocks

    def huffman_decode_block(self, bits, reverse_huffman):
        """
        Decode a single block of Huffman encoded bits
        :param bits: Bit string for a single block
        :param reverse_huffman: Reversed Huffman code dictionary
        :return: List of decoded symbols
        """
        decoded = []
        current_code = ""
        for bit in bits:
            current_code += bit
            if current_code in reverse_huffman:
                symbol = reverse_huffman[current_code]
                decoded.append(symbol)
                current_code = ""
        return decoded

    def rle_decode_blocks(self, rle_blocks):
        """
        Convert Run-Length Encoded blocks to quantized blocks
        :param rle_blocks: List of RLE encoded blocks
        :return: List of quantized blocks
        """
        quantized_blocks = []
        for rle_block in rle_blocks:
            # Convert RLE to zigzagged block
            zigzag_block = []
            for val, count in rle_block:
                # Add zeros based on the count
                zigzag_block.extend([0] * count)
                # Add the non-zero value (unless it's the EOB marker)
                if val != 0 or (val == 0 and count == 0):  # Special case for EOB
                    zigzag_block.append(val)

            # Ensure we have the right number of elements
            while len(zigzag_block) < self.block_size * self.block_size:
                zigzag_block.append(0)

            # Truncate if we have too many elements (shouldn't happen with proper encoding)
            zigzag_block = zigzag_block[:self.block_size * self.block_size]

            # Convert zigzag to block
            block = self.inverse_zigzag_order(zigzag_block)
            quantized_blocks.append(block)

        return quantized_blocks

    def inverse_zigzag_order(self, zigzag_array):
        """
        Convert a zigzag array back to a 2D block
        This is the reverse of the zigzag_order method in huffman.py
        :param zigzag_array: 1D array in zigzag order
        :return: 2D block
        """
        block = np.zeros((self.block_size, self.block_size), dtype=np.int8)
        for i in range(self.block_size):
            for j in range(self.block_size):
                index = self.zigzag_pattern[i, j]
                if index < len(zigzag_array):
                    block[i, j] = zigzag_array[index]
        return block

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
            qtable = self.chromiance_quantization_table

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

    def upsample_chromiance(self, channels):
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
    try:
        # Create a FlexibleJpegDecompress instance
        decompressor = FlexibleJpegDecompress()
        test_image_path = os.path.join(os.getcwd(), "tmp", "decomp_test.bin.rde")

        # Decompress the image from the fixed path
        decompressed_image, save_path = decompressor(test_image_path)

        print(f"Successfully decompressed image from tmp/decomp_test.bin.rde")
        print(f"Decompressed image saved to: {save_path}")

    except Exception as e:
        print(f"Error decompressing image: {e}")