# This File is for individual sub-routines of our compression and decompression pipelines.
import numpy as np
from collections import Counter, namedtuple
import heapq
import numpy as np
from plotly import graph_objects as go

def make_serializable(obj):
    if isinstance(obj, np.array):
        return obj.to_list()
    else:
        return obj

def display_greyscale_image(fig, image_array, **kwargs):
    fig.add_trace(go.Heatmap(z=image_array, colorscale='gray'), **kwargs)

    fig.update_yaxes(autorange='reversed', scaleanchor='x', constrain='domain')
    fig.update_xaxes(constrain='domain')
    fig.update_layout(coloraxis_showscale=False)  # Remove color scale