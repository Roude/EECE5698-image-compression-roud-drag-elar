import os
import numpy as np
from skimage import io, color
import imageio.v3 as imageio
import yaml
from scipy.fftpack import dctn, idctn


class CompressImage:
    def __init__(self, config=None):
        """Base class for image compression with configuration support."""
        self.config = {}
        if config:
            with open(config, 'r') as config_file:
                self.config = yaml.load(config_file, yaml.SafeLoader)
            self.update_configuration(self.config)

    def update_configuration(self, config):
        """Update the compression configuration."""
        if isinstance(config, str):
            with open(config, 'r') as config_file:
                self.config = yaml.load(config_file, yaml.SafeLoader)
        else:
            self.config = config

    def set_datatype_and_channels(self, image_uncompressed):
        """Ensure image has correct datatype and channels."""
        #TODO see if needed for regular compression algo, makes it unnecessarily bright
        #if image_uncompressed.dtype != np.uint8:
            #image_uncompressed = (255 * (image_uncompressed / np.max(image_uncompressed))).astype(np.uint8)
        if image_uncompressed.shape[-1] == 4:  # Remove alpha channel if present
            image_uncompressed = image_uncompressed[:, :, :3]
        return image_uncompressed

    def set_save_location(self, file_path, suffix="_decompressed"):
        """Set save location to the tmp directory with fixed name."""
        tmp_dir = os.path.join(os.getcwd(), "tmp")
        os.makedirs(tmp_dir, exist_ok=True)  # Ensure directory exists
        self.save_location = os.path.join(tmp_dir, "flex_jpeg_comp.png")
        return self.save_location


class JPEGLikeCompressor(CompressImage):
    def __init__(self, config=None):
        super().__init__(config)

        # Default compression parameters
        self.block_size = 8
        self.downsample_factor = 2
        self.upsample_factor = self.downsample_factor

        # Color conversion defaults
        self.YCbCr_conversion_matrix = np.array([
            [65.738, 129.057, 25.064],
            [-37.945, -74.494, 112.439],
            [112.439, -94.154, -18.285]
        ], dtype=np.float32) / 256

        self.YCbCr_conversion_offset = np.array([16, 128, 128]).transpose()

        # Quantization tables
        self.luminance_quantization_table = np.array([
            [6, 4, 4, 6, 10, 16, 20, 24],
            [5, 5, 6, 8, 10, 23, 24, 22],
            [6, 5, 6, 10, 16, 23, 28, 22],
            [6, 7, 9, 12, 20, 35, 32, 25],
            [7, 9, 15, 22, 27, 44, 41, 31],
            [10, 14, 22, 26, 32, 42, 45, 37],
            [20, 26, 31, 35, 41, 48, 48, 40],
            [29, 37, 38, 39, 45, 40, 41, 40]
        ], dtype=np.uint8)

        self.chrominance_quantization_table = np.array([
            [10, 8, 9, 9, 9, 8, 10, 9],
            [9, 9, 10, 10, 10, 11, 12, 17],
            [13, 12, 12, 12, 12, 20, 16, 16],
            [14, 17, 18, 20, 23, 23, 22, 20],
            [25, 25, 25, 25, 25, 25, 25, 25],
            [25, 25, 25, 25, 25, 25, 25, 25],
            [25, 25, 25, 25, 25, 25, 25, 25],
            [25, 25, 25, 25, 25, 25, 25, 25]
        ], dtype=np.uint8)

    def __call__(self, image, settings):
        """Main compression/decompression pipeline."""
        # Load and process input
        if isinstance(image, str):
            image_uncompressed = io.imread(image)
            self.original_path = image
        else:
            image_uncompressed = image

        if isinstance(settings, str):
            with open(settings, 'r') as settings_file:
                settings = yaml.safe_load(settings_file)

        # Update parameters from settings
        self._update_parameters_from_settings(settings)

        # Validate image dimensions
        self._validate_image_dimensions(image_uncompressed)

        # Process image through pipeline
        image_uncompressed = self.set_datatype_and_channels(image_uncompressed)
        YCbCrImage = self._convert_to_YCbCr(image_uncompressed)
        downsampled_image = self._downsample_chrominance(YCbCrImage)
        block_processed_channels = self._process_blocks(downsampled_image)
        unprocessed_blocks = self._process_blocks_inverse(block_processed_channels)
        upsampled_image = self._upsample_chrominance(unprocessed_blocks)
        rgb_image = self._convert_to_RGB(upsampled_image)

        # Save and return result
        if hasattr(self, 'original_path'):
            self.set_save_location(self.original_path)
        if hasattr(self, 'save_location'):
            imageio.imwrite(self.save_location, rgb_image.astype(np.uint8))
            print(f"Decompressed image saved to: {self.save_location}")

        return rgb_image, getattr(self, 'save_location', None)

    def _update_parameters_from_settings(self, settings):
        """Update compression parameters from settings dictionary."""
        self.YCbCr_conversion_matrix = np.array(
            settings.get("YCbCr_conversion_matrix"),
            dtype=np.float32
        ) / 256

        self.YCbCr_conversion_offset = np.array(
            settings.get("YCbCr_conversion_offset"),
            dtype=np.uint8
        )

        # Simplified quantization tables for testing
        self.luminance_quantization_table = np.array([
            [1, 1, 1, 1, 2, 2, 4, 4],
            [1, 1, 1, 1, 2, 2, 4, 4],
            [1, 1, 1, 2, 2, 4, 4, 4],
            [1, 1, 2, 2, 2, 4, 4, 4],
            [2, 2, 2, 2, 4, 4, 4, 4],
            [2, 2, 2, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4],
            [4, 4, 4, 4, 4, 4, 4, 4]
        ], dtype=np.float32)

        self.chrominance_quantization_table = self.luminance_quantization_table.copy()

    def _validate_image_dimensions(self, image):
        """Check if image dimensions are compatible with block size."""
        self.image_dimensions = image.shape[:2]
        self.channel_amount = image.shape[2]

        self.chrominance_dimensions = (
            self.image_dimensions[0] // self.downsample_factor,
            self.image_dimensions[1] // self.downsample_factor
        )

        # Check divisibility by block size
        for dim in [self.image_dimensions, self.chrominance_dimensions]:
            if any(d % self.block_size != 0 for d in dim):
                print(f'Warning! Dimensions {dim} not divisible by {self.block_size}')

    def _convert_to_YCbCr(self, rgb_image):
        """Convert RGB image to YCbCr color space."""
        return color.rgb2ycbcr(rgb_image).astype(np.uint8)

    def _downsample_chrominance(self, YCbCr_image):
        """Downsample chrominance channels using averaging."""
        luminance = YCbCr_image[:, :, 0]
        ch_CbCr = YCbCr_image[:, :, 1:]

        # Average blocks of downsampled_factor x downsampled_factor pixels
        running_average = np.zeros_like(
            ch_CbCr[::self.downsample_factor, ::self.downsample_factor, :],
            dtype=np.float32
        )

        for idx in range(self.downsample_factor):
            for jdx in range(self.downsample_factor):
                running_average += (
                        ch_CbCr[idx::self.downsample_factor, jdx::self.downsample_factor, :] /
                        (self.downsample_factor ** 2)
                )

        return [
            luminance,
            running_average[:, :, 0].astype(np.uint8),
            running_average[:, :, 1].astype(np.uint8)
        ]

    def _process_blocks(self, downsampled_image):
        """Process each channel in blocks through DCT and quantization."""
        processed_channels = []

        for ch_num, channel in enumerate(downsampled_image):
            processed_channel = np.zeros_like(channel, dtype=np.int16)

            for idx in range(0, channel.shape[0], self.block_size):
                for jdx in range(0, channel.shape[1], self.block_size):
                    # Get current block
                    block = self._get_image_block(channel, idx, jdx)

                    # Process block
                    frequency_block = self._block_DCT(block)
                    quantized_block = self._quantize_block(frequency_block, ch_num)

                    # Store result
                    end_i = idx + min(self.block_size, channel.shape[0] - idx)
                    end_j = jdx + min(self.block_size, channel.shape[1] - jdx)
                    processed_channel[idx:end_i, jdx:end_j] = quantized_block[:end_i - idx, :end_j - jdx]

            processed_channels.append(processed_channel)

        return processed_channels

    def _get_image_block(self, channel, idx, jdx):
        """Extract a block from the image with boundary handling."""
        end_i = idx + self.block_size
        end_j = jdx + self.block_size
        return channel[idx:end_i, jdx:end_j]

    def _block_DCT(self, image_block):
        """Perform 2D DCT on image block."""
        return dctn(image_block.astype(np.float32) - 128, norm='ortho') * 2

    def _quantize_block(self, frequency_block, ch_num):
        """Quantize frequency domain block using appropriate quantization table."""
        # Pad block to standard size if needed
        padded_block = self._pad_matrix(frequency_block, self.block_size, self.block_size)

        # Select quantization table
        q_table = (
            self.luminance_quantization_table if ch_num == 0
            else self.chrominance_quantization_table
        )

        # Quantize and ensure DC coefficient is not zero
        quantized = np.round(padded_block / (q_table / 4)).astype(np.int16)
        quantized[-1, -1] = 0  # Ensure last coefficient is zero for RLE

        return quantized[:frequency_block.shape[0], :frequency_block.shape[1]]

    def _pad_matrix(self, matrix, target_rows, target_cols):
        """Pad matrix with zeros to reach target dimensions."""
        if matrix.shape[0] > target_rows or matrix.shape[1] > target_cols:
            raise ValueError("Target dimensions must be >= original dimensions")

        padded = np.zeros((target_rows, target_cols), dtype=matrix.dtype)
        padded[:matrix.shape[0], :matrix.shape[1]] = matrix
        return padded

    def _process_blocks_inverse(self, quantized_blocks):
        """Reverse the block processing (dequantization + IDCT)."""
        reconstructed_channels = []

        for ch_num, channel in enumerate(quantized_blocks):
            reconstructed = np.zeros_like(channel, dtype=np.uint8)

            for idx in range(0, channel.shape[0], self.block_size):
                for jdx in range(0, channel.shape[1], self.block_size):
                    # Get current block
                    block = self._get_image_block(channel, idx, jdx)

                    # Reverse processing
                    dequantized = self._dequantize_block(block, ch_num)
                    spatial_block = self._inverse_block_DCT(dequantized)

                    # Store result
                    end_i = idx + min(self.block_size, channel.shape[0] - idx)
                    end_j = jdx + min(self.block_size, channel.shape[1] - jdx)
                    reconstructed[idx:end_i, jdx:end_j] = spatial_block[:end_i - idx, :end_j - jdx]

            reconstructed_channels.append(reconstructed)

        return reconstructed_channels

    def _dequantize_block(self, quantized_block, ch_num):
        """Reverse the quantization process."""
        padded = self._pad_matrix(quantized_block, self.block_size, self.block_size)
        q_table = (
            self.luminance_quantization_table if ch_num == 0
            else self.chrominance_quantization_table
        )
        return (padded * q_table).astype(np.int16)[
               :quantized_block.shape[0],
               :quantized_block.shape[1]
               ]

    def _inverse_block_DCT(self, frequency_block):
        """Perform inverse DCT and clip to valid range."""
        return np.clip(idctn(frequency_block, norm='ortho') + 128,0, 255).astype(np.uint8)

    def _upsample_chrominance(self, channels):
        """Upsample chrominance channels to match luminance dimensions."""
        Y = channels[0]
        Cb = np.repeat(
            np.repeat(channels[1], self.upsample_factor, axis=0),
            self.upsample_factor, axis=1
        )
        Cr = np.repeat(
            np.repeat(channels[2], self.upsample_factor, axis=0),
            self.upsample_factor, axis=1
        )

        if Y.shape != Cb.shape or Y.shape != Cr.shape:
            raise ValueError("Upsampled chrominance dimensions don't match luminance")

        return np.dstack((Y, Cb, Cr)).astype(np.uint8)

    def _convert_to_RGB(self, ycbcr_image):
        """Convert YCbCr image back to RGB color space."""
        ycbcr_scaled = ycbcr_image.astype(np.float32)
        ycbcr_scaled[..., 0] = ycbcr_scaled[..., 0] * (235 / 255) + 16  # Y
        ycbcr_scaled[..., 1:] = ycbcr_scaled[..., 1:] * (240 / 255) + 16  # Cb/Cr

        rgb = color.ycbcr2rgb(ycbcr_scaled)
        final = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        print(final)
        return final


if __name__ == '__main__':
    compressor = JPEGLikeCompressor()

    #test_image_path = os.path.join(os.getcwd(), "assets", "unit_test_images", "white_16x16.tif")
    test_image_path = os.path.join(os.getcwd(), "assets", "test_images", "landscape.png")

    compression_config = os.path.join(
        os.getcwd(),
        "compression_configurations",
        "homemade_compression_jpeg_like.yaml"
    )

    compressor(test_image_path, compression_config)