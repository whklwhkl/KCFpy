from track import Track
from PIL import Image, ImageTk
from colors import colors

import numpy as np
import requests
import tkinter as tk
import cv2
from io import BytesIO
from threading import Thread
from queue import Queue
from scr.worker import Worker
from scr.abnormal_det import MovingAverage
# from termcolor import colored

# HOST = '192.168.1.253'  # 192.168.20.122
HOST = '192.168.20.122'  # 192.168.20.122

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


# def par(img_file):
#     api_calls['attributes'] += 1
#     response = requests.post(PAR_URL, files={'img': img_file})
#     return np.fromstring(response.json()['predictions'], dtype=np.uint8)


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


frame = None

q_reg = Queue(32)


def get_click_point(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        for trk in Track.ALL:
            l, t, w, h = trk.box
            if l < x < l + w and t < y < t + h:
                def work():
                    img_roi = _crop(frame, trk.box)
                    trk.feature = ext(_nd2file(img_roi))
                    # entry = Entry()
                    # entry.show()
                    # up(entry.content, trk.feature)
                    up(str(trk.id), trk.feature)
                    q_reg.put(1)

                th = Thread(target=work)
                th.start()
                th.join()
                break


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

    # window = tk.Tk()
    # window.overrideredirect(True)
    # window.wm_attributes('-topmost', True)
    # window.geometry('+960+540')
    # display1 = tk.Label(window)
    # display1.grid(row=1, colunm=0, padx=0, pady=0)

    cap = cv2.VideoCapture(cam_id)
    frame_count = 0
    INTEVAL = 24
    FORGETTING = .99
    win = cv2.namedWindow(sys.argv[1])
    cv2.setMouseCallback(sys.argv[1], get_click_point)
    w_det = Worker(lambda x: (x, det(_nd2file(x))))
    w_ext = Worker(lambda i, x: (i, ext(_nd2file(x))))
    w_cmp = Worker(lambda i, x: (i, query(x)))
    w_par = Worker(lambda i, x: (i, par(_nd2file(x))))
    MATCHES = {}
    sim_ema = {}

    while cap.isOpened():
    # def show_frame():
        ret, frame = cap.read()
        if not ret or frame is None:
            break
            # return
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
            c = colors[ret.get('idx')]
            if i is not None and i != -1:
                t.similarity = ret.get('similarity')

                if i not in MATCHES:
                    MATCHES[i] = t
                    sim_ema[i] = MovingAverage(t.similarity, conf_band=2)
                    t.color = c
                    t.id = i
                    MATCHES[i] = t
                else:
                    # print(sim_ema[i].x, (sim_ema[i].x2 - sim_ema[i].x ** 2) ** .5 * sim_ema[i].k)
                    if t.similarity > sim_ema[i].x and sim_ema[i](t.similarity):
                        f = MATCHES[i]
                        f.color = Track.color
                        f.id = int(f.id)
                        f.similarity = 0
                        t.color = c
                        t.id = i
                        MATCHES[i] = t
                # change colors
                # if t.similarity > .93:
                #     if i in MATCHES and MATCHES[i] < t:
                #
                #         # w_par.put(t, _crop(frame_, t.box))
                #     elif i not in MATCHES:
                #         t.color = c
                #         t.id = i
                #         MATCHES[i] = t

        if not w_par.p.empty():
            # t, att = w_par.get()            # person attributes
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
                        cv2.putText(frame, a, (x + w + 3, y), cv2.FONT_HERSHEY_SIMPLEX, 2.0, t.color, 3)
                        y += h//8
        for i, kv in enumerate(api_calls.items()):
            cv2.putText(frame, '{:<10}'.format(kv[0]), (10, i*20 + 60), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0), 1)
            cv2.putText(frame, '{:>6}'.format(kv[1]), (100, i*20 + 60), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0), 1)
        # frame = cv2.resize(frame, (900, 460))
        # imgtk = ImageTk.PhotoImage(master=display1, image=Image.fromarray(frame[...,::-1]))
        # display1.imgtk = imgtk
        # display1.configure(image=imgtk)
        #
        # window.after(10, show_frame)

        cv2.imshow(sys.argv[1], frame)
        c = cv2.waitKey(1) & 0xFF
        if c == 27 or c == ord('q'):
            break
        frame_count += 1

    cv2.destroyAllWindows()
    # show_frame()
    # window.mainloop()
    for w in [w_ext, w_det, w_cmp]:
        w.running = False
        w.th.join(0.5)
