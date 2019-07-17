from track import Track
from colors import colors

import numpy as np
import requests
import tkinter
import asyncio  # todo
import cv2
from time import time
from io import BytesIO
from threading import Thread
from queue import Queue
from termcolor import colored


DET_URL = 'http://192.168.20.122:6666/det'
EXT_URL = 'http://192.168.20.122:6667/ext'
CMP_URL = 'http://192.168.20.122:6668/{}'

api_calls = {'register': 0, 'detection': 0, 'feature': 0, 'query': 0, }


def det(img_file):
    api_calls['detection'] += 1
    response = requests.post(DET_URL, files={'img': img_file})
    return response.json()


def ext(img_file):
    api_calls['feature'] += 1
    response = requests.post(EXT_URL, files={'img': img_file})
    return response.json()


def up(identity, feature):
    api_calls['register'] += 1
    response = requests.post(CMP_URL.format('update'), json={'id': identity, 'feature': feature})
    response.json()


def query(feature):
    api_calls['query'] += 1
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
        target = self.window.mainloop()

    def destroy(self):
        self.window.destroy()


frame = None

q_reg = Queue(1)
def get_click_point(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        for trk in Track.ALL:
            l, t, w, h = trk.box
            if l < x < l + w and t < y < t + h:

                def work():
                    img_roi = _crop(frame, trk.box)
                    trk.feature = ext(_nd2file(img_roi))
                    entry = Entry()
                    entry.show()
                    up(entry.content, trk.feature)
                    q_reg.put(1)

                th = Thread(target=work)
                th.start()
                break


class Worker:
    def __init__(self, func):
        self.q = Queue(maxsize=32)  # in
        self.p = Queue(maxsize=32)  # out
        self.running = True

        def loop():
            while self.running:
                try:
                    i = self.q.get()
                    o = func(*i)
                    self.p.put(o)
                except Exception:
                    continue

        self.th = Thread(target=loop, daemon=True)
        self.th.start()

    def put(self, *args):
        self.q.put(args)

    def get(self):
        return self.p.get()


if __name__ == '__main__':
    import sys
    cam_id = int(sys.argv[1])
    # cap = cv2.VideoCapture('/home/wanghao/Videos/CVPR19-02.mp4')
    cap = cv2.VideoCapture(cam_id)
    frame_count = 0
    INTEVAL = 24
    win = cv2.namedWindow('tracking')
    cv2.setMouseCallback('tracking', get_click_point)
    w_det = Worker(lambda x: (x, det(_nd2file(x))))
    w_ext = Worker(lambda i, x: (i, ext(_nd2file(x))))
    w_cmp = Worker(lambda i, x: (i, query(x)))
    MATCHES = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        frame_ = frame.copy()       # no drawing
        Track.step(frame_)
        if frame_count % INTEVAL == 0:
            w_det.put(frame_)
            for t in MATCHES.values():
                if not t.visible:
                    t.similarity *= 0.99    # forgetting

        if not w_det.p.empty():
            frame_, boxes = w_det.get()
            if len(boxes):
                boxes = _cvt_ltrb2ltwh(boxes)
                Track.update(frame_, boxes)
                for t in Track.ALL:
                    if t.visible and t.feature is None:
                        img_roi = _crop(frame_, t.box)
                        w_ext.put(t, img_roi)
            Track.decay()

        if not q_reg.empty():
            q_reg.get()
            for t in Track.ALL:
                if t.feature is not None:
                    w_cmp.put(t, t.feature)

        if not w_ext.p.empty():
            t, feature = w_ext.get()
            t.feature = feature
            w_cmp.put(t, feature)

        if not w_cmp.p.empty():
            t, ret = w_cmp.get()
            i = ret.get('id')
            c = colors[ret.get('idx')]
            if i is not None and i != -1:
                t.similarity = ret.get('similarity')
                if i in MATCHES and MATCHES[i] < t:
                    f = MATCHES[i]
                    f.color = Track.color
                    f.id = -1
                    f.similarity = 0
                    t.color = c
                    t.id = i
                    MATCHES[i] = t
                elif i not in MATCHES:
                    t.color = c
                    t.id = i
                    MATCHES[i] = t
            # print(colored('%d'%len(MATCHES), 'green'))

        Track.render(frame)
        for i, kv in enumerate(api_calls.items()):
            cv2.putText(frame, '{:<10}'.format(kv[0]), (10, i*20 + 60), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0), 2)
            cv2.putText(frame, '{:>6}'.format(kv[1]), (100, i*20 + 60), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0), 2)
        cv2.imshow('tracking', frame)
        c = cv2.waitKey(1) & 0xFF
        if c == 27 or c == ord('q'):
            break
        frame_count += 1

    cv2.destroyAllWindows()
    for w in [w_ext, w_det, w_cmp]:
        w.running = False
        w.th.join(0.5)
