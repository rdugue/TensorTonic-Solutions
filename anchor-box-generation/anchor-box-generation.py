import numpy as np
def generate_anchors(feature_size, image_size, scales, aspect_ratios):
    """
    Generate anchor boxes for object detection.
    """
    # Write code here
    scales = (np.asarray(scales)).flatten()
    aspect_ratios = (np.asarray(aspect_ratios)).flatten()
    
    stride = image_size  / feature_size
    w = []
    h = []
    for s in scales:
        for r  in aspect_ratios:
            w.append(s * np.sqrt(r))
            h.append(s / np.sqrt(r))

    anchor = []
    for i in range(feature_size):
        for j in range(feature_size):
            cx = (j + 0.5) * stride
            cy = (i + 0.5) * stride
            for w_i, h_i in zip(w, h):
                anchor.append([
                    cx - w_i/2,
                    cy - h_i/2,
                    cx + w_i/2,
                    cy + h_i/2
                ])

    return anchor