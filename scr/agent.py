from .track import Track
from .colors import colors
from .worker import Worker
from .abnormal_det import MovingAverage

from io import BytesIO
from threading import Thread
from queue import Queue

import numpy as np
import requests
import cv2
import os

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport:udp"
# from termcolor import colored


INTEVAL = 24    # det every $INTEVAL frames

# HOST = '192.168.1.253'  # 192.168.20.122
HOST = '192.168.20.191'  # 192.168.20.122

DET_URL = 'http://%s:6666/det' % HOST
EXT_URL = 'http://%s:6667/ext' % HOST
CMP_URL = 'http://%s:6669/{}' % HOST
PAR_URL = 'http://192.168.1.104:1234/par'

ATTRIBUTES = ['Female', 'Front', 'Side', 'Back', 'Hat',
              'Glasses', 'Hand Bag', 'Shoulder Bag', 'Backpack',
              'Hold Objects in Front', 'Short Sleeve', 'Long Sleeve',
              'Long Coat', 'Trousers', 'Skirt & Dress']


def det(img_file, api_calls):
    api_calls['detection'] += 1
    response = requests.post(DET_URL, files={'img': img_file})
    return response.json()


def ext(img_file, api_calls):
    api_calls['feature'] += 1
    response = requests.post(EXT_URL, files={'img': img_file})
    return response.json()


def par(img_file, api_calls):
    api_calls['attributes'] += 1
    response = requests.post(PAR_URL, files={'img': img_file})
    return np.fromstring(response.json()['predictions'], dtype=np.uint8)


def up(identity, feature, api_calls):
    api_calls['register'] += 1
    response = requests.post(CMP_URL.format('update'),
                             json={'id': identity, 'feature': feature})
    response.json()


def query(feature, api_calls):
    api_calls['query'] += 1
    response = requests.post(CMP_URL.format('query'),
                             json={'id': '', 'feature': feature})
    return response.json()


def _nd2file(img_nd):
    return BytesIO(cv2.imencode('.jpg', img_nd)[1])


def _cvt_ltrb2ltwh(boxes):
    boxes = np.array(boxes)
    boxes[:, 2: 4] -= boxes[:, :2]
    return boxes[:, :4]


def _crop(frame, trk_box):
    H, W, _ = frame.shape
    left, t, w, h = map(int, trk_box)
    left = max(left, 0)
    t = max(t, 0)
    r = min(left + w, W)
    b = min(t + h, H)
    crop = frame[t: b, left: r, :]
    return cv2.resize(crop, (128, 384))


class Agent:

    """
    interact with `display_queue` and `control_queue`
    display_queue: rendered images as output
    control_queue: (x, y) coordinate pairs as input
    """
    def __init__(self, source):
        self.source = source
        try:
            source = int(source)
        except ValueError:
            pass
        self.cap = cv2.VideoCapture(source)
        self.display_queue = Queue(32)
        self.control_queue = Queue(1)
        self.q_reg = Queue(32)  # register queue
        self.frame_count = 0
        self.api_calls = {k: 0 for k in ['register', 'detection', 'feature',
                                         'query', 'refresh', 'attributes']}
        self.w_det = Worker(lambda x: (x, det(_nd2file(x), self.api_calls)))
        self.w_ext = Worker(lambda i, x: (i, ext(_nd2file(x), self.api_calls)))
        self.w_cmp = Worker(lambda i, x: (i, query(x, self.api_calls)))
        self.w_par = Worker(lambda i, x: (i, par(_nd2file(x), self.api_calls)))
        self.matches = {}
        self.sim_ema = {}

        class _Track(Track):
            ALL = set()
            current_id = 0

        self.Track = _Track
        self.running = True
        self.th = Thread(target=self.loop)
        self.th.start()

    def loop(self):
        while self.running:
            ret, frame = self.cap.read()
            # frame = cv2.resize(frame, (0, 0), fx=.5, fy=.5)  # down-sampling

            if not ret or frame is None:
                break
            frame_ = frame.copy()
            self.Track.step(frame_)
            if self.frame_count % INTEVAL == 0:
                self.w_det.put(frame_)
                self.Track.decay()
            self._post_det_procedure()
            self._post_ext_procedure()
            self._post_cmp_procedure()
            self._post_reg_procedure()
            self._post_par_procedure()
            if not self.control_queue.empty():
                x, y = self.control_queue.get()
                self.click_handle(frame_, x, y)
            self._render(frame)
            # print(self.display_queue.qsize())
            # print(self.w_cmp.p.qsize(), self.w_cmp.q.qsize())
            self.display_queue.put(frame[...,::-1])  # give RGB
            self.frame_count += 1
        self._kill_workers()

    def stop(self):
        self.running = False
        self.th.join(.1)

    def click_handle(self, frame, x, y):
        H, W, _ = frame.shape
        x *= W
        y *= H
        # print(x, y, 'agent', self.source)
        for trk in self.Track.ALL:
            l, t, w, h = trk.box
            if l < x < l + w and t < y < t + h:
                def work():
                    img_roi = _crop(frame, trk.box)
                    trk.feature = ext(_nd2file(img_roi), self.api_calls)
                    # entry = Entry()
                    # entry.show()
                    # up(entry.content, trk.feature)
                    up(str(trk.id), trk.feature, self.api_calls)
                    self.q_reg.put(1)

                th = Thread(target=work)
                th.start()
                th.join()
                break   # only match once

    def _post_det_procedure(self):
        if self.w_det.has_feedback():
            frame_, boxes = self.w_det.get()
            if len(boxes):
                boxes = _cvt_ltrb2ltwh(boxes)
                self.Track.update(frame_, boxes)
                for t in self.Track.ALL:
                    if t.visible and t.feature is None or t.distorted:
                        if t.distorted:
                            self.api_calls['refresh'] += 1
                        img_roi = _crop(frame_, t.box)
                        self.w_ext.put(t, img_roi)
            else:
                for t in self.Track.ALL:
                    t.visible = False
                    t.health -= 1 if t.age > self.Track.PROBATION else 9999

    def _post_ext_procedure(self):
        if not self.w_ext.p.empty():
            t, feature = self.w_ext.get()
            t.feature = feature
            self.w_cmp.put(t, feature)

    def _post_cmp_procedure(self):
        if not self.w_cmp.p.empty():
            t, ret = self.w_cmp.get()
            i = ret.get('id')
            c = colors[ret.get('idx')]
            if i is not None and i != -1:
                t.similarity = ret.get('similarity')

                if i not in self.matches:
                    self.matches[i] = t
                    self.sim_ema[i] = MovingAverage(t.similarity, conf_band=2)
                    t.color = c
                    t.id = i
                    self.matches[i] = t
                    # self.w_par.put(t, _crop(frame_, t.box))
                else:
                    if t.similarity > self.sim_ema[i].x and \
                            self.sim_ema[i](t.similarity):
                        f = self.matches[i]
                        f.color = Track.color
                        f.id = int(f.id)
                        f.similarity = 0
                        t.color = c
                        t.id = i
                        self.matches[i] = t

    def _post_reg_procedure(self):
        if not self.q_reg.empty():
            self.q_reg.get()
            for t in self.Track.ALL:
                if t.feature is not None:
                    self.w_cmp.put(t, t.feature)

    def _post_par_procedure(self):
        if not self.w_par.p.empty():
            # t, att = self.w_par.get()            # person attributes
            setattr(t, 'par', att)

    def _render(self, frame):
        self.Track.render(frame)
        for t in self.Track.ALL:
            if hasattr(t, 'par') and t.visible:
                x, y, w, h = map(int, t.box)
                y += h//4
                for a, m in zip(ATTRIBUTES, t.par):
                    if a == 'Female' and not m:
                        a = 'Male'
                        m = True
                    if m:
                        cv2.putText(frame, a, (x + w + 3, y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, t.color, 3)
                        y += h//8
        for i, kv in enumerate(self.api_calls.items()):
            cv2.putText(frame, '{:<10}'.format(kv[0]), (10, i*20 + 60),
                        cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0), 1)
            cv2.putText(frame, '{:>6}'.format(kv[1]), (100, i*20 + 60),
                        cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0), 1)

    def _kill_workers(self):
        for w in [self.w_ext, self.w_det, self.w_cmp, self.w_par]:
            w.suicide()
