import numpy as np
from collections import Counter, namedtuple
import heapq


zigzag_pattern = np.array([
    [ 0,  1,  5,  6, 14, 15, 27, 28],
    [ 2,  4,  7, 13, 16, 26, 29, 42],
    [ 3,  8, 12, 17, 25, 30, 41, 43],
    [ 9, 11, 18, 24, 31, 40, 44, 53],
    [10, 19, 23, 32, 39, 45, 52, 54],
    [20, 22, 33, 38, 46, 51, 55, 60],
    [21, 34, 37, 47, 50, 56, 59, 61],
    [35, 36, 48, 49, 57, 58, 62, 63]
])

def zigzag_order(matrix):
    h, w = matrix.shape
    result = np.zeros(h * w, dtype=np.int32)
    for i in range(h):
        for j in range(w):
            result[zigzag_pattern[i, j]] = matrix[i, j]
    return result


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


#block = np.array([
#    [16, 11, 10, 16, 24, 40, 51, 61],
#    [12, 12, 14, 19, 26, 58, 60, 55],
#    [14, 13, 16, 24, 40, 57, 69, 56],
#    [14, 17, 22, 29, 51, 87, 80, 62],
#    [18, 22, 37, 56, 68, 109, 103, 77],
#    [24, 35, 55, 64, 81, 104, 113, 92],
#    [49, 64, 78, 87, 103, 121, 120, 101],
#    [72, 92, 95, 98, 112, 100, 103, 99]
#])

#zigzag_result = zigzag_order(block)
#print(zigzag_result)

#only for zeros, can be extended to all numbers - alternatively LZ77
def run_length_encoding(zigzag_array):
    encoded = []
    count = 0
    for val in zigzag_array:
        if val == 0:
            count += 1
        else:
            encoded.append((val, count))
            count = 0
    encoded.append((0, 0))  # End-of-Block (EOB)
    return encoded

#zigzag_data = [16, 11, 10, 0, 0, 0, 0, 0, 1, 2, 0, 0, 3, 0, 0, 0]
#rle_result = run_length_encoding(zigzag_data)
#print(rle_result)  # [(16, 0), (11, 0), (10, 0), (1, 5), (2, 0), (3, 2), (0, 0)]


#frequency = absolute Häufigkeit
#huffman code is prefixed and has the lowest mean codewordlength
#TODO check questions below
#what if chars share the same freq, only nonneg integers right? huffman code to change for each block?
#online it was proposed to only use the first values of the rle_data - how does that work? doesn't that remove the zeros
class HuffmanNode(namedtuple("Node", ["char", "freq", "left", "right"])):
    #less than
    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(freq_dict):
    heap = [HuffmanNode(char, freq, None, None) for char, freq in freq_dict.items()]
    #making a min-heap so we can easily pop the two least frequent symbols
    heapq.heapify(heap)
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = HuffmanNode(None, left.freq + right.freq, left, right)
        heapq.heappush(heap, merged)
        #the following node contains the entire tree
    return heap[0]

def generate_huffman_codes(node, prefix="", code_dict={}):
    if node.char is not None:
        code_dict[node.char] = prefix
    else:
        generate_huffman_codes(node.left, prefix + "0", code_dict)
        generate_huffman_codes(node.right, prefix + "1", code_dict)
    return code_dict

def huffman_encode(rle_data, huffman_codes):
    return " ".join(huffman_codes[val] for val, _ in rle_data)


if __name__ == '__main__':
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
