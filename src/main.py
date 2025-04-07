import os

from compression import CompressImage

jpeg_compressor = CompressImage(os.path.join(os.getcwd(),
                                             "compression_configurations",
                                             "baseline_jpeg_q100.yaml"))

if __name__ == '__main__':

    path_to_compressed_image = jpeg_compressor(os.path.join(os.getcwd(), "assets", "landscape.png"))
