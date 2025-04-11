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


def parse_huffman_table(huffman_table_str):
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

    # Safely evaluate the main dictionary
    raw_tables = ast.literal_eval(huffman_table_str)

    if not isinstance(raw_tables, Mapping):
        raise ValueError("Huffman table must be a dictionary")

    # Process each component table
    fixed_tables = {}
    for table_name, table in raw_tables.items():
        if not isinstance(table, Mapping):
            raise ValueError(f"Table {table_name} must be a dictionary")

        fixed_table = {}
        for symbol, code in table.items():
            fixed_table[convert_value(symbol)] = code

        fixed_tables[table_name] = fixed_table

    return fixed_tables


import numpy as np
import json


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


def display_greyscale_image(fig, image_array, **kwargs):
    fig.add_trace(go.Heatmap(z=image_array, colorscale='gray'), **kwargs)

    fig.update_yaxes(autorange='reversed', scaleanchor='x', constrain='domain')
    fig.update_xaxes(constrain='domain')
    fig.update_layout(coloraxis_showscale=False)  # Remove color scale