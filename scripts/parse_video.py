import cv2
import sys
import numpy as np
import requests
import argparse
from io import BytesIO
from os import mkdir
from PIL import Image
from os.path import join, exists, basename, splitext

from scr.track import Track
from utils.box import parse_box


ap = argparse.ArgumentParser()
ap.add_argument('--video_path', '-v')
ap.add_argument('--output_folder', '-o', default=None)
ap.add_argument('--ip', '-i', default='localhost')
ap.add_argument('--port', '-p')
ap.add_argument('--classes', '-c', default='person', help='names of classes of interest')
ap.add_argument('--frequency', '-f', default=24, type=int,help='frequency of performing detection')
ap.add_argument('--see', '-s', action='store_true', help='if chosen, play the video while tracking')
args = ap.parse_args()


cap = cv2.VideoCapture(args.video_path)
address = 'http://%s:%s/det' % (args.ip, args.port)
args.output_folder = args.output_folder or splitext(args.video_path)[0]
if not exists(args.output_folder):
    mkdir(args.output_folder)

from tqdm import tqdm
count = cap.get(cv2.CAP_PROP_POS_FRAMES)
print(count, 'fram')
while True:
    count = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    ret, frame = cap.read()
    if not ret: break
    Track.step(frame)
    if count % args.frequency:
        pass
    else:
        H, W, _ = frame.shape
        frame_square = np.pad(frame, [[0, max(0, W - H)], [0, max(0, H - W)], [0,0]], mode='constant')
        file = BytesIO(cv2.imencode('.jpg', frame_square)[1])
        det = requests.post(address, files={'img':file}).json()
        if 'dets' not in det:
            continue
        box = parse_box(det, condition_fn=lambda x:x in args.classes.split(','))
        if len(box):
            try:
                Track.update(frame, box)
            except:
                pass
        Track.decay()
        img = Image.fromarray(frame[...,::-1])
        for d in Track.ALL:
            if d.is_valid():
                l, t, w, h = d.box
                r, b = l + w, t + h
                if not exists(join(args.output_folder, str(d.id))):
                    mkdir(join(args.output_folder, str(d.id)))
                img.crop([l,t,r,b]).save(
                    join(args.output_folder,str(d.id),
                         '{id}_{frame_count}.jpg'.format(id=d.id,
                                                         frame_count=count)))
    if args.see:
        Track.render(frame)
        cv2.imshow('foo', frame)
        if cv2.waitKey(1)==27:
            break
    print('\033[F'+str(count), 'frames', 'processed')
