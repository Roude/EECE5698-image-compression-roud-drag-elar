# This File is for individual sub-routines of our compression and decompression pipelines.
import numpy as np
from collections import Counter, namedtuple
import heapq
import numpy as np
from plotly import graph_objects as go
#from torch.distributed.rpc.internal import serialize


def make_serializable(obj):
    if isinstance(obj, np.array):
        return obj.to_list()
    else:
        return obj


def parse_huffman_table(raw_huffman_tables):
    """Parse Huffman table string and ensure proper tuple types"""
    import ast
    from collections.abc import Mapping

    def convert_value(v):
        if isinstance(v, str) and v.startswith('(') and v.endswith(')'):
            try:
                return ast.literal_eval(v)  # Convert string tuple to real tuple
            except (ValueError, SyntaxError):
                return v
        return v

    if not isinstance(raw_huffman_tables, Mapping):
        raise ValueError("Huffman table must be a dictionary")

    # Process each component table
    fixed_tables = {}
    for table_name, table in raw_huffman_tables.items():
        if not isinstance(table, Mapping):
            raise ValueError(f"Table {table_name} must be a dictionary")

        fixed_table = {}
        for symbol, code in table.items():
            fixed_table[convert_value(symbol)] = code

        fixed_tables[table_name] = fixed_table

    return fixed_tables

def make_serializable_table(tables):
    serializable_tables = {}
    for table_name, table in tables.items():
        serializable_tables[table_name] = {}
        for key, value in table.items():
            # Convert numpy types to native Python types
            if isinstance(key, tuple):
                # Option 1: Convert tuple to a string (e.g., "(0,1)")
                serialized_key = str(
                    tuple(int(x) if isinstance(x, (np.integer, np.int16, np.int32)) else x for x in key))

                # Option 2: Convert tuple to a list (JSON-compatible)
                # serialized_key = [int(x) if isinstance(x, (np.integer, np.int16, np.int32)) else x for x in key]
            elif isinstance(key, (np.integer, np.int16, np.int32)):
                serialized_key = int(key)
            else:
                serialized_key = key  # Keep as-is (int, str, etc.)

            # Ensure the value is serializable
            if isinstance(value, (np.integer, np.int16, np.int32)):
                serialized_value = int(value)
            else:
                serialized_value = value

            serializable_tables[table_name][serialized_key] = serialized_value
    return serializable_tables
'''
def make_serializable(obj):
    if isinstance(obj, (np.integer, np.int32, np.int64, np.int8, np.int16)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, tuple):
        return tuple(make_serializable(item) for item in obj)
    elif isinstance(obj, list):
        return [make_serializable(item) for item in list]
    elif isinstance(obj, dict):
        return {make_serializable(k): make_serializable(v) for k, v in obj.items()}
    else:
        return obj
'''


def bytes_to_bools(byte_data, padding_bits=0):
    """
    Directly convert bytes to a numpy bool array without string conversion
    :param byte_data: bytes object containing packed bits
    :param padding_bits: number of padding bits to remove from end
    :return: numpy array of bools (1 bit per element)
    """
    # Convert bytes to uint8 array (no copy, just view)
    byte_array = np.frombuffer(byte_data, dtype=np.uint8)

    # Unpack bits directly to bool array
    bool_array = np.unpackbits(byte_array)

    # Remove padding if needed
    if padding_bits > 0:
        bool_array = bool_array[:-padding_bits]

    return bool_array

def display_greyscale_image(fig, image_array, **kwargs):
    fig.add_trace(go.Heatmap(z=image_array, colorscale='gray'), **kwargs)

    fig.update_yaxes(autorange='reversed', scaleanchor='x', constrain='domain')
    fig.update_xaxes(constrain='domain')
    fig.update_layout(coloraxis_showscale=False)  # Remove color scale


def gaussian_matrix(shape, max_value, std_dev):
    """
    Generate a 2D Gaussian matrix with the peak at the bottom-right corner.

    Parameters:
        shape (tuple): Shape of the matrix as (rows, cols).
        max_value (float): Maximum value of the Gaussian (at bottom-right).
        std_dev (float): Standard deviation of the Gaussian.

    Returns:
        numpy.ndarray: 2D matrix with Gaussian distribution.
    """
    rows, cols = shape
    x = np.arange(cols)
    y = np.arange(rows)
    x_grid, y_grid = np.meshgrid(x, y)

    # Bottom-right corner is (cols-1, rows-1)
    x0, y0 = cols - 1, rows - 1

    # Calculate squared distance from the peak for each point
    dist_sq = (x_grid - x0)**2 + (y_grid - y0)**2

    # Apply 2D Gaussian function
    gaussian = np.exp(-dist_sq / (2 * std_dev**2))

    # Normalize so the max is 1, then scale to max_value
    gaussian *= max_value / gaussian[y0, x0]

    return gaussian

def ln_norm(shape, max_value, min_value, norm_order):
    """
    Generate a 2D matrix with a Gaussian-like decay based on the L-N norm distance
    from the bottom-right corner.

    Parameters:
        shape (tuple): Shape of the matrix as (rows, cols).
        max_value (float): Maximum value at the bottom-right corner.
        std_dev (float): Spread (controls decay rate).
        norm_order (float): Order of the norm (e.g., 1 for Manhattan, 2 for Euclidean, np.inf for Chebyshev).

    Returns:
        numpy.ndarray: Matrix with generalized Gaussian decay.
    """
    rows, cols = shape
    x = np.arange(cols)
    y = np.arange(rows)
    x_grid, y_grid = np.meshgrid(x, y)

    # Coordinates of the peak (bottom-right)
    x0, y0 = cols - 1, rows - 1

    # Compute L-N distance from each point to the bottom-right
    dx = np.abs(x_grid - x0)
    dy = np.abs(y_grid - y0)
    distance = (dx**norm_order + dy**norm_order)**(1.0 / norm_order)
    slope = (max_value - min_value)/np.max(distance)

    # Exponential decay
    matrix = max_value - slope*distance

    return matrix

def match_dimensions_by_clipping(arr1: np.ndarray, arr2: np.ndarray):
    """
    Clips the larger array along each axis so that both arrays match the shape of the smaller one.
    Returns the two clipped arrays.
    """
    # Get the minimum shape along each dimension
    min_shape = tuple(min(s1, s2) for s1, s2 in zip(arr1.shape, arr2.shape))

    # Define slicing helper
    def crop_to_shape(arr, shape):
        slices = tuple(slice(0, s) for s in shape)
        return arr[slices]

    arr1_cropped = crop_to_shape(arr1, min_shape)
    arr2_cropped = crop_to_shape(arr2, min_shape)

    return arr1_cropped, arr2_cropped

if __name__ == '__main__':
    matrix = gaussian_matrix(shape=(5, 5), max_value=1.0, std_dev=4.0)
    print(matrix)