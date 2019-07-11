import numpy as np
import kcftracker
import requests
import asyncio  # todo
import cv2
from io import BytesIO


DET_URL = 'http://0.0.0.0:6666/det'
EXT_URL = 'http://0.0.0.0:6667/ext'
CMP_URL = 'http://0.0.0.0:6668/{}'


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


def color(idx):
    return (0, 255, 0)


if __name__ == '__main__':
    cap = cv2.VideoCapture('/home/wanghao/Videos/CVPR19-02.mp4')
    frame_count = 0
    INTEVAL = 24
    trackers = {}
    while cap.isOpened():
        ret, frame = cap.read()
        frame_ = frame.copy()
        if not ret:
            break
        if frame_count % INTEVAL:
            # tracking
            for i, t in trackers.items():
                boundingbox = t.update(frame_)  # l,t, w,h
                l, t, w, h = map(int, boundingbox)
                cv2.rectangle(frame, (l, t), (l+w, t+h), color(i), 1)
        else:
            # detect & extract
            boxes = det(BytesIO(cv2.imencode('.jpg', frame_)[1]))
            for i, (l, t, r, b, c) in enumerate(boxes):
                l, t, r, b = map(int, [l, t, r, b])
                cv2.rectangle(frame, (l, t), (r, b), color(i), 2)
                cv2.putText(frame, '%.2f' % c, (l, t), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color(i), 2)
                tracker = kcftracker.KCFTracker(False, True, True)
                tracker.init([l, t, r-l, b-t], frame_)
                # assign box to tracker
                trackers[i] = tracker
        cv2.imshow('tracking', frame)
        c = cv2.waitKey(1) & 0xFF
        if c == 27 or c == ord('q'):
            break
        frame_count += 1
