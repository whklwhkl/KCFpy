from track import Track
from colors import colors

import numpy as np
import requests
import tkinter
import hashlib
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


class Entry:
    def __init__(self):
        self.window = tkinter.Tk()
        self.label = tkinter.Label(self.window, text='name:')
        self.label.pack()
        self.entry = tkinter.Entry(self.window)
        self.entry.pack()
        self.content = None

        def set_name():
            self.content = self.entry.get()
            self.destroy()

        self.buttonY = tkinter.Button(self.window, text='confirm', command=set_name)
        self.buttonY.pack()
        self.buttonN = tkinter.Button(self.window, text='cancle', command=self.destroy)
        self.buttonN.pack()

    def show(self):
        self.window.mainloop()

    def destroy(self):
        self.window.destroy()


frame = None


def get_click_point(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        for trk in Track.ALL:
            l, t, w, h = trk.box
            if l < x < l + w and t < y < t + h:
                img_roi = _crop(frame, trk.box)
                trk.feature = ext(_nd2file(img_roi))
                entry = Entry()
                entry.show()
                up(entry.content, trk.feature)
                break


if __name__ == '__main__':
    # cap = cv2.VideoCapture('/home/wanghao/Videos/CVPR19-02.mp4')
    cap = cv2.VideoCapture(0)
    frame_count = 0
    INTEVAL = 24
    win = cv2.namedWindow('tracking')
    cv2.setMouseCallback('tracking', get_click_point)

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
                    img_roi = _crop(frame, t.box)
                    t.feature = ext(_nd2file(img_roi))
                    id_idx = query(t.feature)
                    i = id_idx.get('id')
                    if i is not None and i != -1:
                        t.id = i
                        t.color = colors[id_idx.get('idx')]
        Track.render(frame)
        cv2.imshow('tracking', frame)
        c = cv2.waitKey(1) & 0xFF
        if c == 27 or c == ord('q'):
            break
        frame_count += 1
