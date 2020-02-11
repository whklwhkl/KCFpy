from .agent import *
from .agent import _crop, _nd2file
from .kcftracker import KCFTracker
from .record import Record
from .worker import Consumer
from utils.action import cut, frames2data
from time import time
from datetime import datetime


PAR = True
DEBUG = False
SIMILARITY_THRESHOLD = .85
DISTANCE_THRESHOLD = 200
FEATURE_MOMENTUM = .9
IMAGE_LIST_SIZE = 10
MIN_OVERSTAY = 120


def make_object_type():
    class Person:

        current_id = 0

        def __init__(self, feature, box):
            cls = type(self)
            self.id = str(cls.current_id)
            self.feature = feature
            self.box = box
            self.first_seen = self.last_seen = time()
            self.last_save = 0
            self.imgs = []  # video clips of action recognition for each person id
            cls.current_id += 1

        # def add_img(self, frame):
        #     self.imgs.append(cut(frame, self.box))
        #     if len(self.imgs) > IMAGE_LIST_SIZE:
        #         self.imgs.pop(0)

        def __str__(self):
            return self.id

        @staticmethod
        def is_overstay(seconds):
            return seconds > MIN_OVERSTAY

        @staticmethod
        def is_alike(similarity):
            return similarity > SIMILARITY_THRESHOLD

        @staticmethod
        def is_far(delta):
            return delta > DISTANCE_THRESHOLD

    return Person


def make_track_type():
    from .async_kcftracker import update
    class _Track(Track):
        ALL = set()
        current_id = 0
        health = 5
        CANDIDATE_IOU = .5

        def step1(self, frame):
            coro = update(self.tracker, frame)
            try:
                coro.send(None)
            except StopIteration as e:
                new_box = e.value
            # new_box = self.tracker.update(frame)
            new_box = np.array(new_box)
            ds = new_box - self.box
            ds_ = ds if self.velocity is None else self.velocity
            self.velocity = ds * Track.momentum_ + ds_ * Track.momentum
            self.box = new_box
            H, W = frame.shape[:2]
            l, t, h, w = self.box
            if 0 < (l+w/2) < W and 0 < (t+h/2) < H:
                pass
            else:
                self.visible = False

    return _Track
    # return _Track


class Storage:

    def __init__(self, object_type, memory=15):
        self.id_map = {}
        self.memory = memory
        self.object_type = object_type

    def add(self, feature, box):
        p = self.object_type(feature, box)
        self.id_map[p.id] = p

    def reg(self, feature, box):
        feature = np.array(feature, np.float32)
        if len(self.id_map):
            q = self.query(feature)
            p = self.id_map[q['id']]
            if self.object_type.is_far(abs(box - p.box).sum()):
                # diff
                # print('new', abs(box - p.box).sum())
                self.add(feature, box)
            elif self.object_type.is_alike(q['similarity']):
                # same
                p.feature = p.feature * FEATURE_MOMENTUM + feature * (1-FEATURE_MOMENTUM)
                p.box = p.box * FEATURE_MOMENTUM + box * (1-FEATURE_MOMENTUM)
            else:
                # occlusion
                pass # ignore
        else:
            self.add(feature, box)

    def query(self, feature):
        assert len(self.id_map), 'no id in storage, register first!'
        similarity_lst = []
        id_lst = []
        for v in self.id_map.values():
            similarity_lst += [self.compare(v.feature, feature)]
            id_lst += [v]
        idx = np.argmax(similarity_lst)
        id = id_lst[idx]
        sim = similarity_lst[idx]
        return {'id': str(id),
                'similarity': similarity_lst[idx]}

    def forget(self):
        now = time()
        delete_keys = []
        for k in self.id_map:
            if now - self.id_map[k].last_seen > self.memory:
                delete_keys.append(k)
        for k in delete_keys:
            del self.id_map[k]

    @staticmethod
    def compare(feature1, feature2):
        f1 = np.array(feature1)
        f2 = np.array(feature2)
        cos = np.dot(f1, f2)/np.linalg.norm(f1)/np.linalg.norm(f2)
        return cos
        # return np.exp(cos - 1)

class PersonAgent(Agent):

    def __init__(self, source, host='localhost'):
        super().__init__(source, host)
        self.current_date = datetime.now().date() # - timedelta(days=1)
        source_dir = source[source.find('@')+1:source.find('/cam')]
        source_dir = os.path.basename(source_dir)

        self.source_dir = source_dir

        self.output_dir = os.path.join('output', source_dir, str(self.current_date))

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.output_log = os.path.join(self.output_dir + '/log.txt')

        self.Track = make_track_type()
        self.api_calls = {k: 0 for k in ['register',
                                         'detection',
                                         'feature',
                                         'query',
                                         'refresh',
                                         'attributes',
                                         # 'action',
                                         ]}
        PAR_URL = 'http://%s:6666/att' % host
        EXT_URL = 'http://%s:6666/fea' % host
        # ACT_URL = 'http://%s:6671/act' % host
        self.storage = Storage(make_object_type())
        # self.bag_storage = BagStorage()

        def par(img_file, api_calls):
            api_calls['attributes'] += 1
            # print(img_file)
            response = requests.post(PAR_URL, files={'img': img_file})
            att = response.json()
            return att
            # return np.array(response.json()['predictions'], dtype=np.uint8)

        def ext(img_file, api_calls):
            api_calls['feature'] += 1
            response = requests.post(EXT_URL, files={'img': img_file})
            return response.json()

        def query(feature, api_calls):
            api_calls['query'] += 1
            try:
                return self.storage.query(feature)
            except AssertionError:
                return {}

        #Function to put frame into Record object and push to queue
        def output(record, frame):
            record.add_frame(frame)
            return record, frame

        # def act(img_list, api_calls):
        #     api_calls['action'] += 1
        #     a = frames2data(img_list)
        #     response = requests.post(ACT_URL, pickle.dumps(a))
        #     return response.json()[0]

        self.w_par = Worker(lambda i, x: (i, par(_nd2file(x), self.api_calls)), debug=DEBUG)
        self.w_ext = Worker(lambda i, x: (i, ext(_nd2file(x), self.api_calls)))
        self.w_cmp = Worker(lambda i, x: (i, query(x, self.api_calls)))
        #Worker containing the Record objects of overstayed objects
        self.w_record = Worker(lambda x, i: (x, output(x, i)))
        self.w_tracking = Consumer(lambda x: self.Track.step(x), debug=DEBUG)
        # self.w_act = Worker(lambda i, x: (i, act(x, self.api_calls)))
        self.workers.extend([self.w_ext, self.w_cmp, self.w_par, self.w_record,
                             self.w_tracking,
                             ])
        # memory
        self.reported = set()

        self.th = Thread(target=self.loop, daemon=True)
        self.th.start()

    def on_new_det(self, t:Track, img_roi):
        self.w_ext.put(t, img_roi)

    #Function to check current system date
    def check_date(self):
        if datetime.now().date() > self.current_date:
            print('Creating new directory for {}'.format(datetime.now().date()))

            #Update date and create new directory
            self.current_date = datetime.now().date()
            new_dir = os.path.join(os.path.join('output', str(self.source_dir)), str(self.current_date))
            os.makedirs(new_dir)

            #Change output directory and output log file paths
            self.output_dir = new_dir
            self.output_log = new_dir + '/log.txt'

    def loop(self):
        track_list = None
        while self.running:

            #Check date every 600 frames
            if self.frame_count % 600 == 0:
                self.check_date()
                self.Track.ALL = set()

            ret, frame = self.cap.read()

            if not ret or frame is None:
                self.cap = cv2.VideoCapture(self.source)
                # print('renewed', self.source)
                continue
            # frame = cv2.resize(frame, (0, 0), fx=.5, fy=.5)  # down-sampling
            frame_ = frame.copy()
            self.w_tracking.put(frame_)
                # self.Track.step(frame_)

            if self.frame_count % INTEVAL == 0:
                self.w_det.put(frame_)
                self.storage.forget()
                self.Track.decay()
            now = time()
            for t in self.Track.ALL:
                p = self.storage.id_map.get(t.id)
                if p is not None:
                    seconds = now - p.first_seen
                    t.stay = seconds
                    t.par = getattr(p, 'attributes', [])
                    # p.add_img(frame_) # type Person
                    # if len(p.imgs) >= IMAGE_LIST_SIZE and self.frame_count % TIMEWALL == 0:
                    #     self.w_act.put(p.id, p.imgs[-IMAGE_LIST_SIZE:])
                    if getattr(p, 'not_register', True):
                        self.w_par.put(p, _crop(frame_, t.box))
                        p.not_register = False
                    if self.storage.object_type.is_overstay(seconds):
                        t.overstay = True
                        if p.id not in self.reported and hasattr(p, 'attributes'):
                            self.reported.add(p.id)
                            output_path = os.path.join(self.output_dir, '{}_{}_{}.%s'.format(p.id, '_'.join(p.attributes), datetime.now()))
                            self.w_record.put(Record(output_path % 'avi'), frame_)
                            p.example = self.crop(frame_, t.box)
                            cv2.imwrite(output_path % 'jpg', p.example)
                            p.color = t.color
                                # logging = ' '.join(['[overstay] id:', p.id,
                                #                     'attr:', ' '.join(p.attributes),
                                #                     'loc:', self.source])
                                # print(logging)
                                # print(logging, file=open(self.output_log, 'a'))

                    p.last_seen = now
            self._post_det_procedure()
            self._post_ext_procedure()
            self._post_cmp_procedure(frame_)
            # self._post_act_procedure()
            self._post_par_procedure()
            if not self.control_queue.empty():
                x, y = self.control_queue.get()
                H, W, _ = frame.shape
                self.click_handle(int(x * W), int(y * H))
            self._render(frame)

            #Perform post output procedure
            self._post_output_procedure(frame)

            x_offset = 200
            for p in list(self.storage.id_map.values()):
                if hasattr(p, 'example'):
                    example = p.example
                    h, w, _ = example.shape
                    x_offset_ = x_offset + w
                    frame[0:h, x_offset:x_offset_] = example
                    cv2.rectangle(frame, (x_offset, 0), (x_offset_, h), p.color, 1)
                    cv2.putText(frame, p.id, (x_offset, h//2),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, p.color, 2)
                    y_offset = 20
                    for a in p.attributes:
                        cv2.putText(frame, a, (x_offset, h + y_offset),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, p.color, 2)
                        y_offset += 20
                    x_offset = x_offset_
            # print(self.display_queue.qsize())
            # print(self.w_cmp.p.qsize(), self.w_cmp.q.qsize())
            self.display_queue.put(frame)  # give RGB
            self.frame_count += 1
        self._kill_workers()

    def _post_det_procedure(self):
        if self.w_det.has_feedback():
            frame_, boxes = self.w_det.get()
            if len(boxes):
                boxes, labels = _cvt_ltrb2ltwh(boxes)
                # TODO: seperate person and bags into boxes1 and boxes2
                # memory bags in boxes2
                # for b in boxes2:
                #     self.bag_storage.reg(b)
                del labels
                self.Track.update(frame_, boxes)
                for t in self.Track.ALL:
                    # t.visible=True
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

    def _post_ext_procedure(self):
        if not self.w_ext.p.empty():
            t, feature = self.w_ext.get()
            t.feature = feature
            self.w_cmp.put(t, feature)
            self.storage.reg(feature, t.box)
            self.api_calls['register'] += 1

    def _post_cmp_procedure(self, frame_):
        if not self.w_cmp.p.empty():
            t, ret = self.w_cmp.get()
            i = ret.get('id')
            if i is not None:
                t.similarity = ret.get('similarity')
                if self.storage.object_type.is_alike(t.similarity):
                    c = colors[hash(i or 0) % 256]
                    t.color = c
                    t.id = i

    def _post_par_procedure(self):
        if not self.w_par.p.empty():
            p, att = self.w_par.get()            # person attributes
            p.attributes = att

    #Function to perform post output procedure
    def _post_output_procedure(self, frame):
        if self.w_record.has_feedback():
            current_record, _ = self.w_record.get()
            #If True, save video
            if current_record.check_save():
                current_record.save_video()
            else:
                self.w_record.put(current_record, frame)

    # def _post_act_procedure(self):
    #     if not self.w_act.p.empty():
    #         i, ret = self.w_act.get()
    #         if i in self.storage.id_map:
    #             self.storage.id_map[i].action = ','.join(ret)  # take the first action
    #             # self.storage.id_map[i].action = ret[0]  # take the first action

    def _render(self, frame):
        if len(self.points):
            for p in self.points:
                cv2.drawMarker(frame, p, (255, 0, 255))
        if self.contour is not None:
            frame_ = frame.copy()
            cv2.drawContours(frame_, [self.contour], 0, (0, 255, 0), thickness=-1)
            opacity = .7
            cv2.addWeighted(frame_, 1 - opacity, frame, opacity, 0., frame)

        cv2.rectangle(frame, (0,0), (200, 175), (128,128,128), -1)
        cv2.putText(frame, 'Tracks:%d' % len(self.Track.ALL), (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        for i, kv in enumerate(self.api_calls.items()):
            cv2.putText(frame, '{:<10}'.format(kv[0]), (10, i*20 + 60),
                        cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 0), 1)
            cv2.putText(frame, '{:>6}'.format(kv[1]), (100, i*20 + 60),
                        cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 0), 1)
        # trk_lst = []
        # for trk in cls.ALL:
        #     if isinstance(trk.id, str):
        #         trk_lst += [trk]
        #     else:
        #         trk._render(frame)  # unmatched tracks
        # for trk in trk_lst:
        #     if trk.visible:
        #         trk._render(frame)  # tracks with matched ids
        for t in self.Track.ALL:
            if t.visible and getattr(t, 'overstay', False) or DEBUG:
                t._render(frame)
                x, y, w, h = map(int, t.box)
                if hasattr(t, 'overstay'):
                    t.text(frame, 'overstay', int(x), int(y + h + 20)) # type Track
                if hasattr(t, 'stay'):
                    t.text(frame, '%d' % int(t.stay), x + 3, y + h - 3, .6, 2)
                p = self.storage.id_map.get(t.id)
                if p is not None and hasattr(p, 'action'):
                    y += 20
                    cv2.putText(frame, p.action, (x + w + 3, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 1., t.color, 2)
                if hasattr(t, 'par'):
                    # docker image: par2
                    for a in t.par:
                        y += 16
                        cv2.putText(frame, a, (x + w + 3, y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, t.color, 2)

    @staticmethod
    def crop(frame, trk_box):
        H, W, _ = frame.shape
        left, t, w, h = map(int, trk_box)
        left = max(left, 0)
        t = max(t, 0)
        r = min(left + w, W)
        b = min(t + h, H)
        crop = frame[t: b, left: r, :]
        return crop


def _cvt_ltrb2ltwh(boxes, contour=None):
    boxes_ = []
    labels = []
    if contour is None:
        for b in boxes:
            labels.append(b['label'])
            b = b['box']
            boxes_.append([b['left'], b['top'], b['right'], b['bottom']])
    else:
        for b in boxes:
            l = b['labels']
            b = b['box']
            left = b['left']
            top = b['top']
            right = b['right']
            bottom = b['bottom']
            point = ((left+right)/2, bottom)
            if cv2.pointPolygonTest(contour, point, False) > 0:
                # -1:out, 1: in, 0:on
                labels.append(l)
                boxes_.append([left, top, right, bottom])
            else:
                print('excluded', b)
    boxes = np.array(boxes_)
    boxes[:, 2: 4] -= boxes[:, :2]
    return boxes[:, :4], labels
