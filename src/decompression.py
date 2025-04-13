"""
decompression.py

This module implements the decompression pipeline for the custom JPEG-like compression system.
It contains the FlexibleJpegDecompress class, which inherits configuration parameters from FlexibleJpeg.
The pipeline includes:
    - Header parsing,
    - Entropy decoding,
    - Block reconstruction (dequantization and 2D IDCT),
    - Upsampling the chrominance channels,
    - Converting from YCbCr back to RGB.
"""

import os
import json
import ast
import numpy as np
from scipy.fftpack import idct
import imageio.v3 as imageio

# Import helper functions and classes from other modules.
from src.utilities import parse_huffman_table, bytes_to_bools
from src.compression import FlexibleJpeg
from src.huffman import generate_zigzag_pattern, inverse_zigzag_order

def idct2(block: np.ndarray) -> np.ndarray:
    """
    Applies a 2D inverse discrete cosine transform (IDCT) to a block using orthonormal normalization.
    
    Parameters:
        block (np.ndarray): A 2D array of DCT coefficients.
        
    Returns:
        np.ndarray: The spatial domain representation after inverse DCT.
    """
    return idct(idct(block, axis=0, norm='ortho'), axis=1, norm='ortho')

class FlexibleJpegDecompress(FlexibleJpeg):
    """
    Implements the decompression pipeline for the custom JPEG-like compression format.
    
    Inherits settings (like block size, quantization tables, and color conversion matrices) from FlexibleJpeg.
    This class:
        - Reads the compressed file header,
        - Reconstructs the quantized blocks via entropy decoding,
        - Dequantizes and applies a 2D IDCT,
        - Reassembles channels and upsamples chrominance,
        - Converts back to RGB.
    """

    def __init__(self, config: dict = None) -> None:
        """
        Initializes the FlexibleJpegDecompress object with an optional configuration.
        
        Parameters:
            config (dict, optional): Compression configuration. Defaults to None.
        """
        super().__init__(config)
        self.image_dimensions: tuple[int, int] = None
        self.upsampling_factor: int = self.downsample_factor
        self.save_location: str = None

    def set_save_location(self, file_path: str) -> None:
        """
        Sets a default save location based on the input file name.
        
        For example, for "foo.rde", it generates "foo_decompressed.png".
        
        Parameters:
            file_path (str): The input compressed file path.
        """
        input_path, _ = os.path.splitext(file_path)
        self.save_location = f"{input_path}_decompressed.png"

    def __call__(self, compressed_file: str = None, **kwargs) -> tuple[np.ndarray, str]:
        """
        Main decompression method.
        
        This method:
            1. Decodes the compressed file and extracts header metadata, bitstream, and Huffman tables.
            2. Reverses the entropy encoding to reconstruct quantized DCT blocks.
            3. Processes each block via dequantization and a 2D IDCT.
            4. Reassembles the image channels, upsamples chrominance, and converts from YCbCr to RGB.
            5. Saves and returns the decompressed image along with the save path.
        
        Parameters:
            compressed_file (str): Path to the compressed (.rde) file.
            **kwargs: Optional keyword arguments; can override the save location.
            
        Returns:
            tuple: (decompressed RGB image as a NumPy array, save path as a string)
        """
        # Set default save location.
        self.set_save_location(compressed_file)
        save_location: str = kwargs.get("save_location", self.save_location)

        # Step 1: Decode the file.
        bit_data, huffman_tables, settings = self.decode_from_file(compressed_file)
        if isinstance(settings, str):
            settings = ast.literal_eval(settings)
        self.update_configuration(settings)
        self.upsampling_factor = settings.get("chrominance_downsample_factor", self.downsample_factor)
        if self.block_size == 8:
            self.zigzag_pattern = self.default_zigzag_pattern
        else:
            self.zigzag_pattern = generate_zigzag_pattern(self.block_size)

        # Step 2: Perform entropy decoding.
        quantized_blocks = self.entropy_decode(bit_data, huffman_tables)

        # Step 3: Process blocks (dequantize + 2D IDCT) to reconstruct channels.
        channels = self.process_blocks_inverse(quantized_blocks)

        # Step 4: Upsample chrominance and convert back to RGB.
        upsampled = self.upsample_chrominance(channels)
        rgb_image = self.convert_colorspace_inverse(upsampled)

        # Step 5: Save the output image.
        imageio.imwrite(save_location, rgb_image.astype(np.uint8))
        print(f"Decompressed image saved to: {save_location}")
        return rgb_image, save_location

    def decode_from_file(self, file_path: str) -> tuple[np.ndarray, dict, dict]:
        """
        Reads the compressed file (.rde), extracts the header, bitstream, and Huffman tables.
        
        Parameters:
            file_path (str): Path to the compressed file.
            
        Returns:
            tuple: (bitstream (np.ndarray), Huffman tables (dict), settings (dict))
            
        Raises:
            ValueError: If the file format is not supported.
        """
        if file_path.endswith('.rde'):
            with open(file_path, 'rb') as file:
                header_length = int.from_bytes(file.read(4), byteorder='big')
                header_json: str = file.read(header_length).decode('utf-8')
                header: dict = json.loads(header_json)

                byte_data = file.read()
                bit_data: np.ndarray = bytes_to_bools(byte_data, header.get('padding_bits', 0))

                settings = header['settings']
                huffman_tables_raw = header['huffman_tables']
                huffman_tables = parse_huffman_table(huffman_tables_raw)

                self.image_dimensions = header['image_dimensions']
                self.block_size = header.get('block_size', 8)
                self.downsample_factor = header.get('chrominance_downsample_factor', 2)
            return bit_data, huffman_tables, settings
        else:
            raise ValueError("Unsupported file format; expected a .rde file.")

    def entropy_decode(self, compressed_bits: np.ndarray, huffman_tables: dict) -> list:
        """
        Reconstructs a list of quantized DCT blocks by reversing the entropy encoding.
        
        Parameters:
            compressed_bits (np.ndarray): Boolean array representing the encoded bitstream.
            huffman_tables (dict): Huffman tables used during compression.
            
        Returns:
            list: List of quantized 2D DCT blocks.
            
        Raises:
            ValueError: If decoding fails at any stage.
        """
        reverse_huffman = {name: {v: k for k, v in table.items()} for name, table in huffman_tables.items()}

        def get_min_max(table: dict) -> tuple[int, int]:
            lengths = [len(code) for code in table.values()]
            return min(lengths), max(lengths)

        dc_lum_min, dc_lum_max = get_min_max(reverse_huffman['dc_lum'])
        dc_chrom_min, dc_chrom_max = get_min_max(reverse_huffman['dc_chrom'])
        ac_lum_min, ac_lum_max = get_min_max(reverse_huffman['ac_lum'])
        ac_chrom_min, ac_chrom_max = get_min_max(reverse_huffman['ac_chrom'])

        image_h, image_w = self.image_dimensions
        num_y_blocks_horizontal = image_w // self.block_size
        num_y_blocks_vertical = image_h // self.block_size
        num_y_blocks = num_y_blocks_horizontal * num_y_blocks_vertical

        chrom_h = image_h // self.downsample_factor
        chrom_w = image_w // self.downsample_factor
        num_c_blocks_horizontal = chrom_w // self.block_size
        num_c_blocks_vertical = chrom_h // self.block_size
        num_c_blocks = num_c_blocks_horizontal * num_c_blocks_vertical

        total_blocks = num_y_blocks + 2 * num_c_blocks

        decoded_blocks = []
        prev_dc = [0, 0, 0]
        current_pos = 0
        bit_length = len(compressed_bits)

        for block_idx in range(total_blocks):
            if block_idx < num_y_blocks:
                channel = 0
                dc_min, dc_max, ac_min, ac_max = dc_lum_min, dc_lum_max, ac_lum_min, ac_lum_max
            else:
                channel = 1 if block_idx < num_y_blocks + num_c_blocks else 2
                dc_min, dc_max, ac_min, ac_max = dc_chrom_min, dc_chrom_max, ac_chrom_min, ac_chrom_max

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
                if ac_symbol == (0, 0):
                    break
                else:
                    value, run = ac_symbol
                    ac_coeffs.extend([0] * run)
                    ac_coeffs.append(value)
            if len(ac_coeffs) < (self.block_size * self.block_size - 1):
                ac_coeffs.extend([0] * (self.block_size * self.block_size - 1 - len(ac_coeffs)))
            block_coeffs = [dc_value] + ac_coeffs
            block = inverse_zigzag_order(block_coeffs, self.zigzag_pattern, self.block_size)
            decoded_blocks.append(block)
        return decoded_blocks

    def process_blocks_inverse(self, quantized_blocks: list) -> list:
        """
        Reconstructs full image channels (Y, Cb, Cr) by dequantizing and applying 2D IDCT on blocks,
        then reassembling them using the header's dimensions.
        
        Parameters:
            quantized_blocks (list): List of quantized 2D DCT blocks.
            
        Returns:
            list: [y_channel, cb_channel, cr_channel] as NumPy arrays.
        """
        image_h, image_w = self.image_dimensions
        block_size = self.block_size

        num_y_blocks_horizontal = image_w // block_size
        num_y_blocks_vertical = image_h // block_size

        chrom_h = image_h // self.downsample_factor
        chrom_w = image_w // self.downsample_factor
        num_c_blocks_horizontal = chrom_w // block_size
        num_c_blocks_vertical = chrom_h // block_size

        y_channel = np.zeros((image_h, image_w), dtype=np.float32)
        cb_channel = np.zeros((chrom_h, chrom_w), dtype=np.float32)
        cr_channel = np.zeros((chrom_h, chrom_w), dtype=np.float32)

        index = 0
        for i in range(num_y_blocks_vertical):
            for j in range(num_y_blocks_horizontal):
                block = quantized_blocks[index]
                deq = self.dequantize_block(block, 0)
                block_spatial = idct2(deq) + 128
                r = slice(i * block_size, i * block_size + block_size)
                c = slice(j * block_size, j * block_size + block_size)
                y_channel[r, c] = block_spatial[:y_channel[r, c].shape[0], :y_channel[r, c].shape[1]]
                index += 1

        for i in range(num_c_blocks_vertical):
            for j in range(num_c_blocks_horizontal):
                block = quantized_blocks[index]
                deq = self.dequantize_block(block, 1)
                block_spatial = idct2(deq) + 128
                r = slice(i * block_size, i * block_size + block_size)
                c = slice(j * block_size, j * block_size + block_size)
                cb_channel[r, c] = block_spatial[:cb_channel[r, c].shape[0], :cb_channel[r, c].shape[1]]
                index += 1

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
        Dequantizes a block by elementwise multiplying it with its corresponding quantization table.
        
        Parameters:
            quantized_block (np.ndarray): The quantized block.
            channel_num (int): Channel index (0 for luminance, 1 or 2 for chrominance).
            
        Returns:
            np.ndarray: The dequantized block.
        """
        qtable = self.lumiance_quantization_table if channel_num == 0 else self.chrominance_quantization_table
        h, w = quantized_block.shape
        qtable_used = qtable[:h, :w]
        return quantized_block.astype(np.float32) * qtable_used.astype(np.float32)

    def upsample_chrominance(self, channels: list) -> np.ndarray:
        """
        Upsamples the chrominance channels (Cb and Cr) using nearest-neighbor interpolation,
        and stacks them with the Y channel to form a full-resolution YCbCr image.
        
        Parameters:
            channels (list): [y_channel, cb_channel, cr_channel]
            
        Returns:
            np.ndarray: The upsampled YCbCr image.
        """
        y_channel, cb_channel, cr_channel = channels
        image_h, image_w = self.image_dimensions
        up_cb = np.zeros((image_h, image_w), dtype=np.uint8)
        up_cr = np.zeros((image_h, image_w), dtype=np.uint8)
        factor = self.upsampling_factor

        for i in range(image_h):
            for j in range(image_w):
                up_cb[i, j] = cb_channel[i // factor, j // factor]
                up_cr[i, j] = cr_channel[i // factor, j // factor]

        return np.stack((y_channel.astype(np.uint8), up_cb, up_cr), axis=-1)

    def convert_colorspace_inverse(self, ycbcr_image: np.ndarray) -> np.ndarray:
        """
        Converts the image from YCbCr to RGB using the inverse conversion matrix.
        
        Parameters:
            ycbcr_image (np.ndarray): Image in YCbCr.
            
        Returns:
            np.ndarray: Converted image in RGB.
        """
        conv_matrix = self.YCbCr_conversion_matrix if self.YCbCr_conversion_matrix is not None else self.default_YCbCr_conversion_matrix
        offset = self.YCbCr_conversion_offset if self.YCbCr_conversion_offset is not None else self.default_YCbCr_conversion_offset

        inv_matrix = np.linalg.inv(conv_matrix)
        shifted = ycbcr_image.astype(np.float32) - offset
        rgb = np.tensordot(shifted, inv_matrix, axes=([2], [1]))
        rgb = np.clip(rgb, 0, 255)
        return rgb.astype(np.uint8)
