"""
decompression.py

This module implements the decompression pipeline for the custom JPEG-like compression system.
It contains the FlexibleJpegDecompress class, which inherits from FlexibleJpeg and includes:
    - File decoding and header parsing,
    - Entropy decoding using Huffman tables,
    - Block reconstruction (dequantization and 2D inverse DCT),
    - Chrominance upsampling and conversion from YCbCr to RGB.
    
This implementation relies on NumPy for array operations, SciPy for the inverse DCT, and imageio for
reading/writing image files.
"""

import os
import json
import ast
import numpy as np
from scipy.fftpack import idct
import imageio.v3 as imageio

# Import helper functions and classes from other modules
from src.utilities import parse_huffman_table, bytes_to_bools
from src.compression import FlexibleJpeg
from src.huffman import generate_zigzag_pattern, inverse_zigzag_order

def idct2(block: np.ndarray) -> np.ndarray:
    """
    Applies a 2D inverse discrete cosine transform (IDCT) to a block using orthonormal normalization.
    
    Parameters:
        block (np.ndarray): A 2D NumPy array containing DCT coefficients.
    
    Returns:
        np.ndarray: A 2D NumPy array in the spatial domain after applying the inverse DCT.
    """
    # First apply IDCT to rows, then to columns with orthonormal normalization.
    return idct(idct(block, axis=0, norm='ortho'), axis=1, norm='ortho')

class FlexibleJpegDecompress(FlexibleJpeg):
    """
    Implements the decompression pipeline for a custom JPEG-like compression format.
    
    Inherits configuration and parameters (like block size, quantization tables, and color conversion 
    matrices) from FlexibleJpeg. This class reads a compressed file, decodes header metadata, performs 
    entropy decoding, reconstructs image blocks, upsamples chrominance channels, and converts the result 
    from YCbCr to RGB.
    """

    def __init__(self, config: dict = None) -> None:
        """
        Initializes the FlexibleJpegDecompress object with optional configuration.
        
        Parameters:
            config (dict, optional): Configuration settings. Defaults to None.
        """
        super().__init__(config)
        self.image_dimensions: tuple[int, int] = None  # To be set after header decoding.
        self.upsampling_factor: int = self.downsample_factor  # By default, equal to the downsample factor.

    def __call__(self, compressed_file: str = None, **kwargs) -> tuple[np.ndarray, str]:
        """
        Main decompression method.
        
        Steps performed:
            1. Reads and decodes the compressed file, extracting header metadata, the bitstream, and Huffman tables.
            2. Performs entropy decoding to reconstruct a list of quantized DCT blocks.
            3. For each block, dequantizes and applies a 2D IDCT to revert to the spatial domain.
            4. Reassembles full Y, Cb, and Cr channels, upsamples the chrominance channels, and converts to RGB.
            5. Saves the final RGB image and returns it along with its save path.
        
        Parameters:
            compressed_file (str): Path to the compressed (.rde) file.
            **kwargs: Optional keyword arguments; can override save_location.
            
        Returns:
            tuple: A tuple containing:
                - The decompressed RGB image as a NumPy array.
                - The file path where the decompressed image was saved.
        """
        # Determine and set the save location based on the input file.
        self.set_save_location(compressed_file)
        save_location: str = kwargs.get("save_location", self.save_location)

        # Step 1: Decode the file and extract header, bitstream, and Huffman tables.
        bit_data, huffman_tables, settings = self.decode_from_file(compressed_file)
        if isinstance(settings, str):
            settings = ast.literal_eval(settings)
        self.update_configuration(settings)
        # Update upsampling factor from header settings.
        self.upsampling_factor = settings.get("chrominance_downsample_factor", self.downsample_factor)
        # Set the zigzag pattern based on current block size.
        if self.block_size == 8:
            self.zigzag_pattern = self.default_zigzag_pattern
        else:
            self.zigzag_pattern = generate_zigzag_pattern(self.block_size)

        # Step 2: Entropy decode the bitstream to retrieve quantized blocks.
        quantized_blocks = self.entropy_decode(bit_data, huffman_tables)

        # Step 3: Process each block (dequantization and 2D IDCT) to reassemble image channels.
        channels = self.process_blocks_inverse(quantized_blocks)

        # Step 4: Upsample chrominance channels and convert the image from YCbCr to RGB.
        upsampled = self.upsample_chrominance(channels)
        rgb_image = self.convert_colorspace_inverse(upsampled)

        # Step 5: Save the decompressed image and return it.
        imageio.imwrite(save_location, rgb_image.astype(np.uint8))
        print(f"Decompressed image saved to: {save_location}")
        return rgb_image, save_location

# TODO: Double-check decode_from_file path
    def decode_from_file(self, file_path: str) -> tuple[np.ndarray, dict, dict]:
        """
        Reads a compressed file (expected in .rde format) and extracts the header, bitstream, and Huffman tables.
        
        The header, stored in JSON format, contains metadata such as the original image dimensions, block size,
        and chrominance downsample factor.
        
        Parameters:
            file_path (str): Path to the compressed .rde file.
            
        Returns:
            tuple: A tuple containing:
                - bit_data (np.ndarray): A boolean array representing the bitstream.
                - huffman_tables (dict): Dictionary with Huffman tables for entropy decoding.
                - settings (dict): Configuration and metadata from the header.
        
        Raises:
            ValueError: If the file format is unsupported.
        """
        if file_path.endswith('.rde'):
            with open(file_path, 'rb') as file:
                # Read header length (first 4 bytes in big-endian).
                header_length = int.from_bytes(file.read(4), byteorder='big')
                header_json: str = file.read(header_length).decode('utf-8')
                header: dict = json.loads(header_json)

                # Read the remaining byte data and convert it to a boolean array.
                byte_data = file.read()
                bit_data: np.ndarray = bytes_to_bools(byte_data, header.get('padding_bits', 0))

                # Extract settings and Huffman tables from the header.
                settings = header['settings']
                huffman_tables_raw = header['huffman_tables']
                huffman_tables = parse_huffman_table(huffman_tables_raw)

                # Store metadata needed for block reassembly.
                self.image_dimensions = header['image_dimensions']
                self.block_size = header.get('block_size', 8)
                self.downsample_factor = header.get('chrominance_downsample_factor', 2)
            return bit_data, huffman_tables, settings
        else:
            raise ValueError("Unsupported file format for decompression; expected a .rde file.")

    def entropy_decode(self, compressed_bits: np.ndarray, huffman_tables: dict) -> list:
        """
        Reconstructs the list of quantized DCT blocks by reversing the entropy encoding.
        
        The method decodes the bitstream using inverted Huffman tables to extract the DC (delta decoded)
        and AC (run-length decoded) coefficients for each block. The coefficients are then reordered using
        the previously determined zigzag pattern.
        
        Parameters:
            compressed_bits (np.ndarray): A boolean array representing the encoded bitstream.
            huffman_tables (dict): Dictionary of Huffman tables used during compression.
            
        Returns:
            list: A list where each element is a quantized DCT block (2D NumPy array).
        
        Raises:
            ValueError: If decoding fails for DC or AC coefficients.
        """
        # Invert the Huffman tables for easier look-up: {code: symbol}.
        reverse_huffman = {
            name: {v: k for k, v in table.items()}
            for name, table in huffman_tables.items()
        }

        def get_min_max(table: dict) -> tuple[int, int]:
            """Returns the minimum and maximum lengths of the Huffman codes in the table."""
            lengths = [len(code) for code in table.values()]
            return min(lengths), max(lengths)

        dc_lum_min, dc_lum_max = get_min_max(reverse_huffman['dc_lum'])
        dc_chrom_min, dc_chrom_max = get_min_max(reverse_huffman['dc_chrom'])
        ac_lum_min, ac_lum_max = get_min_max(reverse_huffman['ac_lum'])
        ac_chrom_min, ac_chrom_max = get_min_max(reverse_huffman['ac_chrom'])

        # Calculate number of blocks for the luminance (Y) channel.
        image_h, image_w = self.image_dimensions
        num_y_blocks_horizontal = image_w // self.block_size
        num_y_blocks_vertical = image_h // self.block_size
        num_y_blocks = num_y_blocks_horizontal * num_y_blocks_vertical

        # Calculate number of blocks for each chrominance channel (Cb, Cr).
        chrom_h = image_h // self.downsample_factor
        chrom_w = image_w // self.downsample_factor
        num_c_blocks_horizontal = chrom_w // self.block_size
        num_c_blocks_vertical = chrom_h // self.block_size
        num_c_blocks = num_c_blocks_horizontal * num_c_blocks_vertical

        total_blocks = num_y_blocks + 2 * num_c_blocks  # Y + Cb + Cr

        decoded_blocks = []
        prev_dc = [0, 0, 0]  # Previous DC coefficients for Y, Cb, Cr.
        current_pos = 0  # Bitstream pointer.
        bit_length = len(compressed_bits)

        # Process each block sequentially.
        for block_idx in range(total_blocks):
            # Determine the channel index: 0 for Y, 1 for Cb, 2 for Cr.
            if block_idx < num_y_blocks:
                channel = 0
                dc_min, dc_max, ac_min, ac_max = dc_lum_min, dc_lum_max, ac_lum_min, ac_lum_max
            else:
                channel = 1 if block_idx < num_y_blocks + num_c_blocks else 2
                dc_min, dc_max, ac_min, ac_max = dc_chrom_min, dc_chrom_max, ac_chrom_min, ac_chrom_max

            # Decode DC coefficient by reading bits until a valid Huffman code is found.
            delta_dc = None
            for l in range(dc_min, min(dc_max + 1, bit_length - current_pos + 1)):
                code = "".join(str(int(b)) for b in compressed_bits[current_pos:current_pos + l])
                table = reverse_huffman['dc_lum'] if channel == 0 else reverse_huffman['dc_chrom']
                if code in table:
                    delta_dc = int(table[code])
                    current_pos += l
                    dc_value = prev_dc[channel] + delta_dc
                    prev_dc[channel] = dc_value
                    break
            if delta_dc is None:
                raise ValueError(f"DC decoding failed at bit position {current_pos}")

            # Decode AC coefficients using run-length decoding.
            ac_coeffs = []
            while len(ac_coeffs) < (self.block_size * self.block_size - 1):
                ac_symbol = None
                for l in range(ac_min, min(ac_max + 1, bit_length - current_pos + 1)):
                    code = "".join(str(int(b)) for b in compressed_bits[current_pos:current_pos + l])
                    table = reverse_huffman['ac_lum'] if channel == 0 else reverse_huffman['ac_chrom']
                    if code in table:
                        ac_symbol = table[code]
                        current_pos += l
                        break
                if ac_symbol is None:
                    raise ValueError(f"AC decoding failed at bit position {current_pos}")
                # Check for the End-of-Block marker.
                if ac_symbol == (0, 0):
                    break
                else:
                    value, run = ac_symbol
                    ac_coeffs.extend([0] * run)
                    ac_coeffs.append(value)
            # Pad with zeros if block is incomplete.
            if len(ac_coeffs) < (self.block_size * self.block_size - 1):
                ac_coeffs.extend([0] * (self.block_size * self.block_size - 1 - len(ac_coeffs)))
            block_coeffs = [dc_value] + ac_coeffs
            # Reorder coefficients back into a 2D block using the zigzag pattern.
            block = inverse_zigzag_order(block_coeffs, self.zigzag_pattern, self.block_size)
            decoded_blocks.append(block)
        return decoded_blocks

    def process_blocks_inverse(self, quantized_blocks: list) -> list:
        """
        Reconstructs image channels (Y, Cb, Cr) from the list of quantized DCT blocks by:
            - Dequantizing each block,
            - Applying a 2D inverse DCT,
            - Reassembling the blocks into full channel matrices.
        
        Parameters:
            quantized_blocks (list): List of 2D NumPy arrays representing the quantized blocks.
            
        Returns:
            list: A list containing three NumPy arrays [y_channel, cb_channel, cr_channel].
        """
        image_h, image_w = self.image_dimensions
        block_size = self.block_size

        # Compute block counts for the Y channel.
        num_y_blocks_horizontal = image_w // block_size
        num_y_blocks_vertical = image_h // block_size

        # Compute block counts for chrominance channels (Cb and Cr), which are downsampled.
        chrom_h = image_h // self.downsample_factor
        chrom_w = image_w // self.downsample_factor
        num_c_blocks_horizontal = chrom_w // block_size
        num_c_blocks_vertical = chrom_h // block_size

        # Initialize empty arrays for channels.
        y_channel = np.zeros((image_h, image_w), dtype=np.float32)
        cb_channel = np.zeros((chrom_h, chrom_w), dtype=np.float32)
        cr_channel = np.zeros((chrom_h, chrom_w), dtype=np.float32)

        index = 0
        # Process Y channel blocks.
        for i in range(num_y_blocks_vertical):
            for j in range(num_y_blocks_horizontal):
                block = quantized_blocks[index]
                # Dequantize the block.
                deq = self.dequantize_block(block, 0)
                # Apply the 2D IDCT and add the offset to revert the initial centering.
                block_spatial = idct2(deq) + 128
                # Place the block in the correct region of the Y channel.
                r = slice(i * block_size, i * block_size + block_size)
                c = slice(j * block_size, j * block_size + block_size)
                y_channel[r, c] = block_spatial[:y_channel[r, c].shape[0], :y_channel[r, c].shape[1]]
                index += 1

        # Process Cb channel blocks.
        for i in range(num_c_blocks_vertical):
            for j in range(num_c_blocks_horizontal):
                block = quantized_blocks[index]
                deq = self.dequantize_block(block, 1)
                block_spatial = idct2(deq) + 128
                r = slice(i * block_size, i * block_size + block_size)
                c = slice(j * block_size, j * block_size + block_size)
                cb_channel[r, c] = block_spatial[:cb_channel[r, c].shape[0], :cb_channel[r, c].shape[1]]
                index += 1

        # Process Cr channel blocks.
        for i in range(num_c_blocks_vertical):
            for j in range(num_c_blocks_horizontal):
                block = quantized_blocks[index]
                deq = self.dequantize_block(block, 2)
                block_spatial = idct2(deq) + 128
                r = slice(i * block_size, i * block_size + block_size)
                c = slice(j * block_size, j * block_size + block_size)
                cr_channel[r, c] = block_spatial[:cr_channel[r, c].shape[0], :cr_channel[r, c].shape[1]]
                index += 1

        return [y_channel, cb_channel, cr_channel]

    def dequantize_block(self, quantized_block: np.ndarray, channel_num: int) -> np.ndarray:
        """
        Dequantizes a DCT block by multiplying it elementwise with its corresponding quantization table.
        
        Parameters:
            quantized_block (np.ndarray): The quantized DCT block.
            channel_num (int): Channel index (0 for luminance, 1 or 2 for chrominance).
            
        Returns:
            np.ndarray: The dequantized block as a float32 NumPy array.
        """
        # Select appropriate quantization table.
        if channel_num == 0:
            qtable = self.lumiance_quantization_table
        else:
            qtable = self.chrominance_quantization_table

        # If the block size is smaller, slice the qtable.
        h, w = quantized_block.shape
        qtable_used = qtable[:h, :w]
        return quantized_block.astype(np.float32) * qtable_used.astype(np.float32)

    def upsample_chrominance(self, channels: list) -> np.ndarray:
        """
        Upsamples the chrominance channels (Cb and Cr) from reduced dimensions back to full image size.
        Uses simple nearest-neighbor interpolation.
        
        Parameters:
            channels (list): List containing three NumPy arrays [y_channel, cb_channel, cr_channel].
            
        Returns:
            np.ndarray: A combined YCbCr image (with full resolution) as a NumPy array.
        """
        y_channel, cb_channel, cr_channel = channels
        image_h, image_w = self.image_dimensions
        up_cb = np.zeros((image_h, image_w), dtype=np.uint8)
        up_cr = np.zeros((image_h, image_w), dtype=np.uint8)
        factor = self.upsampling_factor

        # Upsample using nearest-neighbor interpolation.
        for i in range(image_h):
            for j in range(image_w):
                up_cb[i, j] = cb_channel[i // factor, j // factor]
                up_cr[i, j] = cr_channel[i // factor, j // factor]

        # Stack the channels along the last dimension to form a full YCbCr image.
        return np.stack((y_channel.astype(np.uint8), up_cb, up_cr), axis=-1)

    def convert_colorspace_inverse(self, ycbcr_image: np.ndarray) -> np.ndarray:
        """
        Converts an image from YCbCr color space back to RGB.
        
        The function subtracts the offset and applies the inverse of the color conversion matrix.
        
        Parameters:
            ycbcr_image (np.ndarray): Image in YCbCr color space.
            
        Returns:
            np.ndarray: Image converted to RGB color space.
        """
        # Use custom conversion matrix if available; otherwise use defaults.
        conv_matrix = self.YCbCr_conversion_matrix if self.YCbCr_conversion_matrix is not None else self.default_YCbCr_conversion_matrix
        offset = self.YCbCr_conversion_offset if self.YCbCr_conversion_offset is not None else self.default_YCbCr_conversion_offset

        # Compute inverse of the conversion matrix.
        inv_matrix = np.linalg.inv(conv_matrix)
        # Subtract the offset.
        shifted = ycbcr_image.astype(np.float32) - offset
        # Apply the inverse transform using tensordot.
        rgb = np.tensordot(shifted, inv_matrix, axes=([2], [1]))
        rgb = np.clip(rgb, 0, 255)
        return rgb.astype(np.uint8)
