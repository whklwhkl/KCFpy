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
    momentum = .95
    momentum_ = 1 - momentum
    similarity = 0
    velocity = np.zeros([4])
    # STATIC
    ALL = set()
    BIRTH_IOU = .5
    CANDIDATE_IOU = .55
    OCCLUSION_IOU = .45
    PROBATION = 2
    MINIMUM_CONFIDENCE = .6

    def __init__(self, frame, init_box, feature=None):
        self.box = init_box
        self.visible = True
        # todo: estimate v3d using 2d info
        self.feature = feature           # appearance
        self.tracker = KCFTracker(False, True, True)
        self.tracker.init(init_box, frame)
        self.id = Track.current_id
        type(self).ALL.add(self)
        type(self).current_id += 1

    def step1(self, frame):
        new_box = self.tracker.update(frame)
        new_box = np.array(new_box)
        ds = new_box - self.box
        ds_ = ds if self.velocity is None else self.velocity
        self.velocity = ds * Track.momentum_ + ds_ * Track.momentum
        self.box = new_box
        self.visible = True

    def step0(self):
        self.box += self.velocity * np.array([1, 1, 0, 0])
        self.visible = False
        # self.health -= 1

    def is_valid(self):
        return self.age >= Track.PROBATION

    def _render(self, frame):
        l, t, w, h = map(int, self.box)
        r = l + w
        b = t + h
        cv2.rectangle(frame, (l, t), (r, b), self.color, 2)

        def text(t, x, y, size=.9):
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, t, (x, y), font, size, self.color, 2)

        text('HP:%d' % self.health, l, t - 3, .6)
        text('LV:%d' % self.age, l + 3, b - 3, .6)
        text(self.id if isinstance(self.id, str) else '?', r, t)
        text('{:.2f}'.format(self.similarity), r, b, .6)

    def __gt__(self, other):
        return self.similarity > other.similarity

    @classmethod
    def update(cls, frame, det_boxes):
        tracks, trk_boxes = cls._gather()
        if len(trk_boxes):
            iou_mtx = iou(trk_boxes, det_boxes)
            match_trk = np.any(iou_mtx > cls.CANDIDATE_IOU, 1)
            for iou_det, m, t in zip(iou_mtx, match_trk, tracks):
                if m:
                    t.tracker = KCFTracker(False, True, True)
                    det_idx = np.argmax(iou_det)
                    t.tracker.init(det_boxes[det_idx], frame)
                else:
                    t.visible = False
            no_match_det = np.all(iou_mtx < cls.BIRTH_IOU, 0)
            for box in det_boxes[no_match_det]:
                cls(frame, box)
        else:
            for box in det_boxes:
                cls(frame, box)

    @classmethod
    def decay(cls):
        dead_trks = []
        for t in cls.ALL:
            if t.visible:
                t.age += 1
                t.health = cls.health
            else:
                t.health -= 1 if t.age >= cls.PROBATION else 9999
            if t.health < 0:
                dead_trks += [t]
        for t in dead_trks:
            cls.ALL.remove(t)

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
    def _gather(cls):
        tracks = list(cls.ALL)
        trk_boxes = list(map(lambda t: t.box, tracks))
        return tracks, np.array(trk_boxes)

    @classmethod
    def render(cls, frame):
        cv2.putText(frame, 'Tracks:%d' % len(cls.ALL), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        trk_lst = []
        for trk in cls.ALL:
            if isinstance(trk.id, str):
                trk_lst += [trk]
            else:
                trk._render(frame)  # unmatched tracks
        for trk in trk_lst:
            if trk.visible:
                trk._render(frame)  # tracks with matched ids


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
