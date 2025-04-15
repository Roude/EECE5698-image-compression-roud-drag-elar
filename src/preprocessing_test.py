import cv2
import yaml
from skimage import io, color
import imageio.v3 as imageio
import os
import numpy as np
from skimage.color import convert_colorspace
from skimage.io import imread
from scipy.fftpack import dct, idct, dctn,idctn

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

class TestingAlgo(CompressImage):
    def __init__(self, config=None):
        super().__init__(config)
        self.zigzag_pattern = None
        self.block_size = 8
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




    def __call__(self, image, settings):
        """
        Implementation of JPEG-like compression with individual functions broken out.
        :param image_uncompressed:
        :param kwargs:
        :return:
        """

        #TODO implement quality factor
        if isinstance(image, str):
            image_uncompressed = io.imread(image)
            self.original_path = image
        else:
            image_uncompressed = image
        if isinstance(settings, str):
            with open(settings, 'r') as settings_file:
                settings = yaml.safe_load(settings_file)

        image_uncompressed = self.set_datatype_and_channels(image_uncompressed)

        self.YCbCr_conversion_matrix = np.array(settings.get("YCbCr_conversion_matrix"), dtype=np.float32) / 256
        self.YCbCr_conversion_offset = np.array(settings.get("YCbCr_conversion_offset"), dtype=np.uint8)

        #TODO are we going to change the quantization table with different quality factors? if so just multiply by a scalar or this?
        def create_emphasis_matrix(self, emphasis_factor):
            matrix = np.zeros((self.block_size, self.block_size), dtype=np.float32)
            for i in range(self.block_size):
                for j in range(self.block_size):
                    # Calculate distance from top-left corner (0,0)
                    distance = np.sqrt(i * i + j * j)
                    # Base value that increases with distance
                    matrix[i, j] = max(1, np.floor(1/(self.block_size * self.block_size) * emphasis_factor * distance))
            return matrix

        #what is this even supposed to mean
        #self.chrominance_quantization_table = self.chrominance_quantization_table
        #self.luminance_quantization_table = self.luminance_quantization_table
        '''
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
        
        '''
        #print(self.luminance_quantization_table)

        #TODO how are these quantization tables working?
        self.image_dimensions = image_uncompressed.shape[:2]
        self.channel_amount = image_uncompressed.shape[2]

        #TODO implement this for compression and decomp
        self.downsample_factor = settings.get("chrominance_downsample_factor", 2)
        self.upsample_factor = self.downsample_factor

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
        unprocessed_blocks = self.process_blocks_inverse(block_processed_channels)
        upsampled_image = self.upsample_chrominance(unprocessed_blocks)
        rgb_image = self.convert_colorspace_inverse(upsampled_image)
        #rgb_image = self.convert_colorspace_inverse(YCbCrImage)
        print(f"Successfully decompressed image")

        tmp_dir = os.path.join(os.getcwd(), "tmp")
        os.makedirs(tmp_dir, exist_ok=True)  # Ensure directory exists
        self.save_location = os.path.join(tmp_dir, "flex_jpeg_testing.png")

        # Save the decompressed image
        if self.save_location:
            imageio.imwrite(self.save_location, rgb_image.astype(np.uint8))

        print(f"Decompressed image saved to: {self.save_location}")

        return rgb_image, self.save_location

    def convert_colorspace(self, image_uncompressed, **kwargs):
        """
        Apply YCbCr color conversion to image
        :param image_uncompressed:
        :param kwargs:
        :return:
        """
        ycbcr_image = cv2.cvtColor(image_uncompressed, cv2.COLOR_RGB2YCrCb)  # Note: OpenCV uses YCrCb order
        return ycbcr_image
    #TODO perhaps establish vertical and horizontal downsampling instead of all as one
    def downsample_chrominance(self, YCbCr_image, **kwargs):
        """
        Apply downsampling factor to YCbCr formatted image and save the image as a tuple of 3 matricies with the
        luminance and two chrominance channels.
        :param YCbCr_image: The image already converted to YCbCr format
        :return: A Tuple, element 0 is the lumiance channel, element 1 is the chrominance channels
        """

        print(self.downsample_factor)

        # TODO understand what exactly is happening here
        luminance = YCbCr_image[:, :, 0]
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

        return [luminance,
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
            #print(range(0, np.shape(channel)[0], self.block_size))
            for idx in range(0,np.shape(channel)[0], self.block_size):
                for jdx in range(0, np.shape(channel)[1], self.block_size):
                    end_idx = idx + self.block_size if np.shape(channel)[0] - idx > self.block_size else None
                    end_jdx = jdx + self.block_size if np.shape(channel)[1] - jdx > self.block_size else None
                    image_block = channel[idx:end_idx, jdx:end_jdx]
                    frequency_block = self.block_DCT(image_block, **kwargs)
                    quantized_block = self.quantize_block(frequency_block, ch_num, **kwargs)
                    #if idx == 0 and jdx == 8 and ch_num == 1:
                        #print(quantized_block)
                    block_processed_channels[ch_num][idx:end_idx, jdx:end_jdx] = quantized_block
        return block_processed_channels

    # might be cleaner to have them as submethods
    def block_DCT(self, image_block, **kwargs):
        """
        First scale pixels from 0 to 255 to -128 to 128, then perform DCT on the block
        :param image_block: An image block of a flexible size, for the DCT to be performed on.
        :param kwargs:
        :return: the cosine transformed block
        """
        dct = dctn(image_block.astype(np.float32) - 128, norm='ortho')
        return dct


    def quantize_block(self, frequency_domain_block, ch_num, **kwargs):
        """
        Divide each block by the quantization table (passed as a keyword argument or the default).
        Round to the nearest integer
        :param frequency_domain_block: The output of block_dct
        :param ch_num: automatically passed in by process_blocks. Indicates which of the quantization tables to use.
        :param kwargs:
        :return: the quantized block
        """
        if ch_num == 0:
            #TODO reset downsample, quantization tables, etc, put it into the comp and decomp algos
            quantized_block = np.round(frequency_domain_block / self.luminance_quantization_table)
        else:
            quantized_block = np.round(frequency_domain_block / self.chrominance_quantization_table)
        # needs to be zero for RLE to be successful
        quantized_block[-1, -1] = 0
        return quantized_block.astype(np.int16)

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
        #print(reconstructed_channels[1])
        return reconstructed_channels

    def dequantize_block(self, quantized_block, ch_num, **kwargs):
        """
        Multiply each block by the quantization table (reverse of quantize_block).
        :param quantized_block: The quantized block from the compressed data
        :param ch_num: Indicates which of the quantization tables to use
        :param kwargs:
        :return: The dequantized frequency domain block
        """
        if ch_num == 0:
            dequantized_block = quantized_block * self.luminance_quantization_table
        else:
            dequantized_block = quantized_block * self.chrominance_quantization_table
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
        Y = channels[0]
        #print(Y)
        # Upsample chrominance channels using nearest-neighbor interpolation
        Cb = np.repeat(np.repeat(channels[1], self.upsample_factor, axis=0),
                       self.upsample_factor, axis=1)
        Cr = np.repeat(np.repeat(channels[2], self.upsample_factor, axis=0),
                       self.upsample_factor, axis=1)

        print(self.upsample_factor)
        print(self.downsample_factor)

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
        print(ycbcr_image[:,:,0])
        rgb_restored = cv2.cvtColor(ycbcr_image, cv2.COLOR_YCrCb2RGB)
        print(rgb_restored[:,:,0])
        final = np.clip(rgb_restored, 0, 255)
        return final


if __name__ == '__main__':

    #baseline_jpeg = BaselineJpeg(os.path.join(os.getcwd(),"compression_configurations", "baseline_jpeg_q100.yaml"))
    #start_time = time.time()

    testin = TestingAlgo()

    #test_image_path = os.path.join(os.getcwd(), "assets", "unit_test_images", "white_16x16.tif")
    test_image_path = os.path.join(os.getcwd(), "assets", "test_images", "landscape.png")

    compression_config = os.path.join(os.getcwd(),
                                              "compression_configurations",
                                              "homemade_compression_jpeg_like.yaml")
    testin(test_image_path, compression_config)
