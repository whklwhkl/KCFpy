import numpy as np


def parse_box(det_dict, condition_fn):
    boxes = []
    for x in det_dict['dets']:
        if condition_fn(x['label']):
            l,t,r,b = x['x1y1x2y2']
            if l>=0 and t>=0 and r>l+1 and b>t+1:
                boxes.append([l, t, r - l, b - t])
    return np.array(boxes)
