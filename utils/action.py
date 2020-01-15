import torch
import cv2
import numpy as np


def cut(img, bbox):
    x1, y1, w, h = map(int, bbox)
    height, width, _ = img.shape

    xc = x1 + w //2
    yc = y1 + h // 2

    xlength = ylength = min(max(w, h), width, height) // 2

    if xc - xlength < 0 and xc + xlength < width - 1:
        xx1 = 0
        xx2 = xlength*2
    elif xc - xlength > 0 and xc + xlength > width - 1:
        xx1 = width - 1 - xlength*2
        xx2 = width - 1
    elif xc - xlength < 0 and xc + xlength > width -1:
        xx1 = 0
        xx2 = width - 1
    else:
        xx1 = xc - xlength
        xx2 = xc + xlength

    if yc - ylength < 0 and yc + ylength < height - 1:
        yy1 = 0
        yy2 = ylength*2
    elif yc - ylength > 0 and yc + ylength > height - 1:
        yy1 = height - 1 - ylength*2
        yy2 = height - 1
    elif yc - ylength < 0 and yc + ylength > height - 1:
        yy1 = 0
        yy2 = height - 1
    else:
        yy1 = yc - ylength
        yy2 = yc + ylength

    return img[yy1:yy2, xx1:xx2, :]


def frames2batch(f_imgs):
    ids = []
    i = 0
    for k, v in f_imgs.items():
        ids.append(k)
        if i == 0:
            simg = frames2data(v)
        else:
            simg = torch.cat((simg, frames2data(v)), 0)
        i += 1
    return ids, simg.half().cuda()


mean = np.array([104, 117, 128], np.uint8)[None, None]
def frames2data(frames):
    simg = []
    for i, frame in enumerate(frames):
        h, w = frame.shape[:2]
        scale_factor = 256 / h

        frame = cv2.resize(frame, (int(w * scale_factor + 0.5), int(h * scale_factor + 0.5)))
        h1, w1 = frame.shape[:2]
        tw = 224
        th = 224
        x1 = (w1 - tw) // 2
        y1 = (h1 - th) // 2
        box = np.array([x1, y1, x1 + tw, y1 + th])
        frame = frame[y1: y1 + th, x1: x1 + tw, :]
        # frame_ = []
        # for b in box:
        #     l,t,r,b = b
        #     frame.append(frame[t:b, l:r, :])
        # print('???')
        # frame = np.stack(frame_) if len(frame_) else np.zeros([0, th, tw, 3], np.uint8)
        # frame = mmcv.imcrop(frame, box)

        frame -= mean
        # frame = mmcv.imnormalize(frame, mean=[104, 117, 128], std=[1, 1, 1], to_rgb=False)
        img = frame.astype(np.float16).transpose([2, 0, 1])
        simg.append(img)
        # if i == 0:
        #     simg = torch.from_numpy(frame.transpose((2, 0, 1))).float()
        #     simg = torch.unsqueeze(simg, 0)
        # else:
        #     img = torch.from_numpy(frame.transpose((2, 0, 1))).float()
        #     img = torch.unsqueeze(img, 0)
        #     simg = torch.cat((simg, img), 0)

    # ssimg = torch.unsqueeze(simg, 0)
    ssimg = np.stack(simg)[None]
    return ssimg
