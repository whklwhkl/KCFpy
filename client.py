from track import Track
from colors import colors

import numpy as np
import requests
import asyncio  # todo
import cv2
from io import BytesIO


DET_URL = 'http://192.168.20.122:6666/det'
EXT_URL = 'http://192.168.20.122:6667/ext'
CMP_URL = 'http://192.168.20.122:6668/{}'


def det(img_file):
    response = requests.post(DET_URL, files={'img': img_file})
    return response.json()


def ext(img_file):
    response = requests.post(EXT_URL, files={'img': img_file})
    return response.json()


def up(identity, feature):
    response = requests.post(CMP_URL.format('update'), json={'id': identity, 'feature': feature})
    response.json()


def query(feature):
    response = requests.post(CMP_URL.format('query'), json={'id': '', 'feature': feature})
    return response.json()


def _nd2file(img_nd):
    return BytesIO(cv2.imencode('.jpg', img_nd)[1])


def _cvt_ltrb2ltwh(boxes):
    boxes = np.array(boxes)
    boxes[:, 2: 4] -= boxes[:, :2]
    return boxes[:, :4]


def _crop(frame, trk_box):
    H, W, _ = frame.shape
    l, t, w, h = map(int, trk_box)
    l = max(l, 0)
    t = max(t, 0)
    r = min(l + w, W)
    b = min(t + h, H)
    crop = frame[t: b, l: r, :]
    return cv2.resize(crop, (128, 384))


if __name__ == '__main__':
    # cap = cv2.VideoCapture('/home/wanghao/Videos/CVPR19-02.mp4')
    cap = cv2.VideoCapture(0)
    frame_count = 0
    INTEVAL = 24

    while cap.isOpened():
        ret, frame = cap.read()
        frame_ = frame.copy()       # no drawing
        if not ret:
            break
        Track.step(frame_)
        if frame_count % INTEVAL == 0:
            boxes = det(_nd2file(frame_))
            if len(boxes):
                boxes = _cvt_ltrb2ltwh(boxes)
                Track.update(frame_, boxes)
            for t in Track.ALL:
                if t.is_valid():
                    t.color = colors[t.id % 256]
                    img_roi = _crop(frame, t.box)
                    t.feature = ext(_nd2file(img_roi))
                    up(t.id, t.feature)
                    i = query(t.feature)
                    assert i == t.id
        Track.render(frame)
        cv2.imshow('tracking', frame)
        c = cv2.waitKey(1) & 0xFF
        if c == 27 or c == ord('q'):
            break
        frame_count += 1
