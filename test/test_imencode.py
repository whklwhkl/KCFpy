import cv2
import numpy as np


if __name__ == '__main__':
    img = np.random.rand(224,224,3).astype(np.uint8)
    for i in range(100):
        cv2.imencode('.jpg', img)
