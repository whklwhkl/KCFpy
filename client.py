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
# from termcolor import colored

HOST = '192.168.1.253'  # 192.168.20.122
# HOST = '192.168.20.122'  # 192.168.20.122

DET_URL = 'http://%s:6666/det' % HOST
EXT_URL = 'http://%s:6667/ext' % HOST
CMP_URL = 'http://%s:6668/{}' % HOST
PAR_URL = 'http://192.168.1.104:1234/par'

ATTRIBUTES = ['Female', 'Front', 'Side', 'Back', 'Hat', 'Glasses', 'HandBag', 'ShoulderBag', 'Backpack', 'HoldObjectsInFront', 'ShortSleeve', 'LongSleeve', 'LongCoat', 'Trousers', 'Skirt&Dress']

api_calls = {k: 0 for k in ['register', 'detection', 'feature', 'query', 'refresh', 'attributes']}


def det(img_file):
    api_calls['detection'] += 1
    response = requests.post(DET_URL, files={'img': img_file})
    return response.json()


def ext(img_file):
    api_calls['feature'] += 1
    response = requests.post(EXT_URL, files={'img': img_file})
    return response.json()


def par(img_file):
    api_calls['attributes'] += 1
    response = requests.post(PAR_URL, files={'img': img_file})
    return np.fromstring(response.json()['predictions'], dtype=np.uint8)


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
    cam_id = sys.argv[1]
    try:
        cam_id = int(cam_id)
    except ValueError:
        print('using other source')
        if 'udp' in cam_id:
            import os
            print('udp')
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport:udp"

    # cap = cv2.VideoCapture('/home/wanghao/Videos/CVPR19-02.mp4')
    cap = cv2.VideoCapture(cam_id)
    frame_count = 0
    INTEVAL = 24
    FORGETTING = .99
    win = cv2.namedWindow('tracking')
    cv2.setMouseCallback('tracking', get_click_point)
    w_det = Worker(lambda x: (x, det(_nd2file(x))))
    w_ext = Worker(lambda i, x: (i, ext(_nd2file(x))))
    w_cmp = Worker(lambda i, x: (i, query(x)))
    w_par = Worker(lambda i, x: (i, par(_nd2file(x))))
    MATCHES = {}
    sim2ema = None
    sim_ema = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        frame_ = frame.copy()       # no drawing
        Track.step(frame_)
        if frame_count % INTEVAL == 0:
            w_det.put(frame_)
            # for t in MATCHES.values():
            #     if not t.visible:
            #         # and t.health > 0
            #         t.similarity *= FORGETTING    # forgetting
            Track.decay()
            # refresh features when abnormal aspect ratio is detected

        if not w_det.p.empty():
            frame_, boxes = w_det.get()
            if len(boxes):
                boxes = _cvt_ltrb2ltwh(boxes)
                Track.update(frame_, boxes)
                for t in Track.ALL:
                    if t.visible and t.feature is None or t.distorted:
                        if t.distorted:
                            api_calls['refresh'] += 1
                        img_roi = _crop(frame_, t.box)
                        w_ext.put(t, img_roi)
            else:
                for t in Track.ALL:
                    t.visible = False
                    t.health -= 1 if t.age > Track.PROBATION else 9999

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
            if i == 'evelyn':
                print('eve', ret.get('similarity'))
            c = colors[ret.get('idx')]
            if i is not None and i != -1:
                t.similarity = ret.get('similarity')
                # change colors
                if t.similarity > .93:
                    if i in MATCHES and MATCHES[i] < t:
                        f = MATCHES[i]
                        f.color = Track.color
                        f.id = -1
                        f.similarity = 0
                        t.color = c
                        t.id = i
                        MATCHES[i] = t
                        w_par.put(t, _crop(frame_, t.box))
                    elif i not in MATCHES:
                        t.color = c
                        t.id = i
                        MATCHES[i] = t

        if not w_par.p.empty():
            t, att = w_par.get()            # person attributes
            setattr(t, 'par', att)

        Track.render(frame)
        for t in Track.ALL:
            if hasattr(t, 'par') and t.visible:
                x, y, w, h = map(int, t.box)
                y += h//4
                for a, m in zip(ATTRIBUTES, t.par):
                    if a == 'Female' and not m:
                        a = 'Male'
                        m = True
                    if m:
                        cv2.putText(frame, a, (x + w + 3, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, t.color, 2)
                        y += h//8
        for i, kv in enumerate(api_calls.items()):
            cv2.putText(frame, '{:<10}'.format(kv[0]), (10, i*20 + 60), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0), 1)
            cv2.putText(frame, '{:>6}'.format(kv[1]), (100, i*20 + 60), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0), 1)
        cv2.imshow('tracking', frame)
        c = cv2.waitKey(1) & 0xFF
        if c == 27 or c == ord('q'):
            break
        frame_count += 1

    cv2.destroyAllWindows()
    for w in [w_ext, w_det, w_cmp]:
        w.running = False
        w.th.join(0.5)
