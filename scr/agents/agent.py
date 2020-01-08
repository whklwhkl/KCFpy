from .track import Track
from .colors import colors
from .worker import Worker
from .abnormal_det import MovingAverage

from io import BytesIO
from threading import Thread
from queue import Queue
from time import sleep

import numpy as np
import requests
import cv2
import os

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport:udp"
# from termcolor import colored


INTEVAL = 5    # det every $INTEVAL frames
REFRESH_INTEVAL = 3


def _nd2file(img_nd):
    return BytesIO(cv2.imencode('.jpg', img_nd)[1])


def _cvt_ltrb2ltwh(boxes):
    boxes_ = []
    for b in boxes['dets']:
        boxes_.append(b['x1y1x2y2'])
    boxes = np.array(boxes_)
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
    return cv2.resize(crop, (128, 256))


class Agent:

    """
    interact with `display_queue` and `control_queue`
    display_queue: rendered images as output
    control_queue: (x, y) coordinate pairs as input
    """

    def __init__(self, source, host='localhost'):
        try:
            source = int(source)
        except ValueError:
            pass
        self.source = os.path.expanduser(source)
        self.cap = cv2.VideoCapture(self.source)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FPS, 12)
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 704)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.display_queue = Queue(32)
        self.control_queue = Queue(1)
        self.api_calls = {k: 0 for k in ['detection', 'refresh']}
        self.frame_count = 0

        # HOST = '192.168.1.100'  # 192.168.20.122
        # HOST = '192.168.1.253'  # 192.168.20.122
        # HOST = '192.168.20.191'  # 192.168.20.122

        DET_URL = 'http://%s:6666/det' % host


        def det(img_file, api_calls):
            api_calls['detection'] += 1
            response = requests.post(DET_URL, files={'img': img_file})
            return response.json()

        self.w_det = Worker(lambda x: (x, det(_nd2file(x), self.api_calls)))
        self.workers = [self.w_det]

        class _Track(Track):
            ALL = set()
            current_id = 0

        self.Track = _Track
        self.running = True
        self.suspend = False
        self.on_det_funcs = []
        self.th = Thread(target=self.loop, daemon=True)
        self.th.start()

    def loop(self):
        while self.running:
            # sleep(0.1)
            if self.suspend == True:
                sleep(0.5)
            ret, frame = self.cap.read()
            # ret, frame = self.cap.read()
            # ret, frame = self.cap.read()

            if not ret or frame is None:
                self.cap = cv2.VideoCapture(self.source)
                # print('renewed', self.source)
                continue
            # frame = cv2.resize(frame, (0, 0), fx=.5, fy=.5)  # down-sampling
            frame_ = frame.copy()
            self.Track.step(frame_)
            if self.frame_count % INTEVAL == 0:
                self.w_det.put(frame_)
                self.Track.decay()
            self._post_det_procedure()
            if not self.control_queue.empty():
                x, y = self.control_queue.get()
                self.click_handle(frame_, x, y)
            self._render(frame)
            # print(self.display_queue.qsize())
            # print(self.w_cmp.p.qsize(), self.w_cmp.q.qsize())
            self.display_queue.put(frame[...,::-1])  # give RGB
            self.frame_count += 1
        self._kill_workers()

    def reset(self):
        pass

    def save(self):
        pass

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
                    trk.feature = self.ext(_nd2file(img_roi), self.api_calls)
                    self.up(str(trk.id), trk.feature, self.api_calls)
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
                    if t.visible:
                        if isinstance(t.id, int):
                            if t.age % REFRESH_INTEVAL == 0:
                                if t.age // REFRESH_INTEVAL:
                                    self.api_calls['refresh'] += 1
                                img_roi = _crop(frame_, t.box)
                                self.on_new_det(t, img_roi)
            else:
                for t in self.Track.ALL:
                    t.visible = False
                    t.health -= 1 if t.age > self.Track.PROBATION else 9999

    def on_new_det(self, t:Track, img_roi):
        return

    def _render(self, frame):
        self.Track.render(frame)
        for i, kv in enumerate(self.api_calls.items()):
            cv2.putText(frame, '{:<10}'.format(kv[0]), (10, i*20 + 60),
                        cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0), 1)
            cv2.putText(frame, '{:>6}'.format(kv[1]), (100, i*20 + 60),
                        cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0), 1)

    def _kill_workers(self):
        for w in self.workers:
            w.suicide()
