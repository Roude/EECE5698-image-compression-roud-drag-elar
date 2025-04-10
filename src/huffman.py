import numpy as np
from collections import Counter, namedtuple
import heapq

def generate_zigzag_pattern(size):
    matrix = np.zeros((size, size), dtype=int)
    current = 0
    for diag in range(2 * size - 1):
        if diag % 2 == 0:  # Even diagonals
            col = max(0, diag - size + 1)
            row = min(diag, size - 1)
            while col < size and row >= 0:
                matrix[row, col] = current
                current += 1
                col += 1
                row -= 1
        else:  # Odd diagonals
            col = min(diag, size - 1)
            row = max(0, diag - size + 1)
            while col >= 0 and row < size:
                matrix[row, col] = current
                current += 1
                col -= 1
                row += 1

    return matrix

def zigzag_order(matrix, pattern):
    h, w = matrix.shape
    result = np.zeros(h * w, dtype=matrix.dtype)
    for i in range(h):
        for j in range(w):
            result[pattern[i, j]] = matrix[i, j]
    return result


def inverse_zigzag_order(flattened_array, pattern):
    h, w = pattern.shape
    original_matrix = np.zeros((h, w), dtype=flattened_array.dtype)
    for i in range(h):
        for j in range(w):
            original_matrix[i, j] = flattened_array[pattern[i, j]]
    return original_matrix

# not as fast as numpy
def diagonal_traversal(matrix):
    """
    Traverses a matrix in a zig-zag diagonal pattern starting from the top-left corner.
    Args:
        matrix (list of list of int): The input matrix.
    Returns:
        list of int: The elements of the matrix in zig-zag diagonal order.
    """
    if not matrix.any():
        return []

    rows, cols = len(matrix), len(matrix[0])
    result = []

    diagonals = []

    # Collect diagonals starting from first column
    for r in range(rows):
        i, j = r, 0
        diagonal = []
        while i >= 0 and j < cols:
            diagonal.append(matrix[i][j])
            i -= 1
            j += 1
        diagonals.append(diagonal)

    # Collect diagonals starting from top row (excluding first element to avoid duplication)
    for c in range(1, cols):
        i, j = rows - 1, c
        diagonal = []
        while i >= 0 and j < cols:
            diagonal.append(matrix[i][j])
            i -= 1
            j += 1
        diagonals.append(diagonal)

    # Flatten the diagonals in a zig-zag pattern
    for idx, diagonal in enumerate(diagonals):
        if idx % 2 == 1:
            diagonal.reverse()
        result.extend(diagonal)

    return result



#only for zeros, can be extended to all numbers - alternatively LZ77
def run_length_encoding(zigzag_array):
    """
    Run-length encoding specific to JPEG standard
    :param zigzag_array: 1D array of coefficients in zigzag order
    :return: List of (value, zero_count) tuples
    """
    encoded = []
    zero_count = 0
    for val in zigzag_array:
        if val == 0:
            zero_count += 1
        else:
            encoded.append((val, zero_count))
            zero_count = 0
    encoded.append((0, 0))  # End-of-Block (EOB)
    return encoded


def run_length_decoding(encoded, expected_length=64):
    zigzag_array = []
    for val, zero_count in encoded[:-1]:
        zigzag_array.extend([0] * zero_count)
        zigzag_array.append(val)

    # Pad remaining with zeros (if needed)
    remaining_zeros = expected_length - len(zigzag_array)
    if remaining_zeros > 0:
        zigzag_array.extend([0] * remaining_zeros)

    return np.array(zigzag_array, dtype=int)



#frequency = absolute Häufigkeit
#huffman code is prefixed and has the lowest mean codewordlength, still better algorithms out there
class HuffmanNode(namedtuple("Node", ["char", "freq", "left", "right"])):
    #less than
    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(freq_dict):
    heap = [HuffmanNode(char, freq, None, None) for char, freq in freq_dict.items()]
    #making a min-heap so we can easily pop the two least frequent symbols, dw about same values heapq handles that
    heapq.heapify(heap)
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = HuffmanNode(None, left.freq + right.freq, left, right)
        heapq.heappush(heap, merged)
        #the following node contains the entire tree
    return heap[0]

def generate_huffman_codes(node, prefix="", code_dict=None):
    if code_dict is None:
        code_dict = {}
    if node.char is not None:
        code_dict[node.char] = prefix
    else:
        generate_huffman_codes(node.left, prefix + "0", code_dict)
        generate_huffman_codes(node.right, prefix + "1", code_dict)
    return code_dict

# ToDo are they all just one string? they should be as we have the ends saved with the EOB
def huffman_encode(rle_data, huffman_codes):
    return "".join(huffman_codes[(val, count)] for val, count in rle_data)

if __name__ == '__main__':
    testmatrix = generate_zigzag_pattern(16)
    print(testmatrix)
    #how to use huffman encoding
    #runlength encoding data
    rle_data = [(16, 0), (11, 0), (16, 0), (11, 5), (16, 0), (3, 2), (0, 0)]
    #create a dictionary by counting the frequencies of values
    freq_dict = Counter([val for val, _ in rle_data])
    print(freq_dict)
    huffman_tree = build_huffman_tree(freq_dict)
    print(huffman_tree)
    huffman_codes = generate_huffman_codes(huffman_tree)
    print(huffman_codes)
    encoded_bits = huffman_encode(rle_data, huffman_codes)
    print(encoded_bits)
