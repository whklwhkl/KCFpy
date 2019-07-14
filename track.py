# all boxes here are of the format LTWH

import cv2
import numpy as np

from kcftracker import KCFTracker


class Track:
    # DEFAULT
    age = 0     # age < PROBATION means is_candidate
    color = (128, 128, 128)  # gray
    current_id = 0
    health = 2  # positive means visible
    is_occluded = False
    momentum = .9
    momentum_ = 1 - momentum
    velocity = np.zeros([4])
    # STATIC
    ALL = set()
    BIRTH_IOU = .5
    CANDIDATE_IOU = .55
    OCCLUSION_IOU = .4
    PROBATION = 1
    MINIMUM_CONFIDENCE = .6

    def __init__(self, frame, init_box, feature=None):
        self.box = init_box
        # todo: estimate v3d using 2d info
        self.descriptor = feature           # appearance
        self.tracker = KCFTracker(False, True, True)
        self.tracker.init(init_box, frame)
        self.id = Track.current_id
        Track.ALL.add(self)
        Track.current_id += 1

    def step1(self, frame):
        new_box = self.tracker.update(frame)
        new_box = np.array(new_box)
        self.velocity = (new_box - self.box) * Track.momentum_ + self.velocity * Track.momentum
        self.box = new_box

    def step0(self):
        self.box += self.velocity * np.array([1, 1, 0, 0])

    def is_valid(self):
        return self.age >= Track.PROBATION

    @classmethod
    def update(cls, frame, det_boxes):
        tracks, trk_boxes = cls._gather()
        if len(trk_boxes):
            iou_mtx = iou(trk_boxes, det_boxes)
            match_trk = np.any(iou_mtx > cls.CANDIDATE_IOU, 1)
            dead_trks = []
            for iou_det, m, t in zip(iou_mtx, match_trk, tracks):
                if m:
                    t.age += 1
                    t.health = cls.health
                    t.tracker = KCFTracker(False, True, True)
                    det_idx = np.argmax(iou_det)
                    t.tracker.init(det_boxes[det_idx], frame)
                elif t.age < cls.PROBATION:
                    t.health -= 9999
                else:
                    t.health -= 1
                if t.health < 0:
                    dead_trks += [t]
            for t in dead_trks:
                cls.ALL.remove(t)
            no_match_det = np.all(iou_mtx < cls.BIRTH_IOU, 0)
            for box in det_boxes[no_match_det]:
                cls(frame, box)
        else:
            for box in det_boxes:
                cls(frame, box)

    @classmethod
    def step(cls, frame):
        if len(cls.ALL) == 0:
            return
        tracks, trk_boxes = cls._gather()
        iou_mtx = iou(trk_boxes, trk_boxes)
        iou_mtx -= np.eye(len(tracks))
        occlusion_mask = np.any(iou_mtx > cls.OCCLUSION_IOU, -1)
        dead_trks = []
        for occ, t in zip(occlusion_mask, tracks):
            if occ:
                t.step0()
            else:
                t.step1(frame)

    @classmethod
    def render(cls, frame):
        for trk in cls.ALL:
            l, t, w, h = map(int, trk.box)
            r = l + w
            b = t + h
            cv2.rectangle(frame, (l, t), (r, b), trk.color, 2)
            cv2.putText(frame, f'{trk.age}', (l, t), cv2.FONT_HERSHEY_SIMPLEX, 0.6, trk.color, 2)
            cv2.putText(frame, f'v: {np.linalg.norm(trk.velocity[:2])}', (r, b), cv2.FONT_HERSHEY_SIMPLEX, 0.6, trk.color, 2)

    @classmethod
    def _gather(cls):
        tracks = list(cls.ALL)
        trk_boxes = list(map(lambda t: t.box, tracks))
        return tracks, np.array(trk_boxes)


def iou(boxes1, boxes2):
    area1 = boxes1[:, 3] * boxes1[:, 2]
    area2 = boxes2[:, 3] * boxes2[:, 2]
    left = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    top = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    right1 = boxes1[:, 2] + boxes1[:, 0]
    bottom1 = boxes1[:, 3] + boxes1[:, 1]
    right2 = boxes2[:, 2] + boxes2[:, 0]
    bottom2 = boxes2[:, 3] + boxes2[:, 1]
    right = np.minimum(right1[:, None], right2[None])
    bottom = np.minimum(bottom1[:, None], bottom2[None])
    intersection = np.maximum(right - left, 0) * np.maximum(bottom - top, 0)
    union = area1[:, None] + area2[None] - intersection
    return intersection / union