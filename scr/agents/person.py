from .agent import *
from .agent import _crop, _nd2file
from time import time


PAR = True
ATTRIBUTES = ['Female', 'Front', 'Side', 'Back', 'Hat',
              'Glasses', 'Hand Bag', 'Shoulder Bag', 'Backpack',
              'Hold Objects in Front', 'Short Sleeve', 'Long Sleeve',
              'Long Coat', 'Trousers', 'Skirt & Dress']
SIMILARITY_THRESHOLD = .8
FEATURE_MOMENTUM = .9


def make_object_type():
    class Person:

        current_id = 0

        def __init__(self, feature):
            cls = type(self)
            self.id = str(cls.current_id)
            self.feature = feature
            self.first_seen = self.last_seen = time()
            self.imgs = []
            cls.current_id += 1

        def __str__(self):
            return self.id

        @staticmethod
        def is_overstay(seconds):
            return seconds > 30
    return Person


def cut(img, bbox):
    x1, y1, w, h = map(int, bbox)
    height, width, _ = img.shape

    xc = x1 + w //2
    yc = y1 + h // 2

    xlength = min(max(w, h), width, height) // 2
    ylength = min(max(w, h), width, height) // 2

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


class Storage:

    def __init__(self, object_type , memory=30):
        self.id_map = {}
        self.memory = memory
        self.frames = []
        self.object_type = object_type

    def add(self, feature):
        p = self.object_type(feature)
        self.id_map[p.id] = p

    def reg(self, feature):
        feature = np.array(feature, np.float32)
        if len(self.id_map):
            q = self.query(feature)
            if q['similarity'] > SIMILARITY_THRESHOLD:
                id = q['id']
                p = self.id_map[id]
                p.feature = p.feature * FEATURE_MOMENTUM + feature * (1-FEATURE_MOMENTUM)
            else:
                self.add(feature)
        else:
            self.add(feature)

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
        if sim > SIMILARITY_THRESHOLD:
            self.id_map[str(id)].last_seen = time()
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

    def add_video(frame):
        self.frames.append(frame)

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
        self.q_reg = Queue(32)  # register queue
        self.api_calls = {k: 0 for k in ['register',
                                         'detection',
                                         'feature',
                                         'query',
                                         'refresh',
                                         'attributes',
                                         'counts']}
        PAR_URL = 'http://%s:6668/par' % host
        EXT_URL = 'http://%s:6667/ext' % host
        self.storage = Storage(make_object_type())

        def par(img_file, api_calls):
            api_calls['attributes'] += 1
            # print(img_file)
            response = requests.post(PAR_URL, files={'img': img_file})
            # print(response)
            return np.array(response.json()['predictions'], dtype=np.uint8)

        def ext(img_file, api_calls):
            api_calls['feature'] += 1
            response = requests.post(EXT_URL, files={'img': img_file})
            return response.json()

        def up(feature, api_calls):
            api_calls['register'] += 1
            self.storage.reg(feature)

        def query(feature, api_calls):
            api_calls['query'] += 1
            try:
                return self.storage.query(feature)
            except AssertionError:
                return {}

        self.w_par = Worker(lambda i, x: (i, par(_nd2file(x), self.api_calls)))
        self.w_ext = Worker(lambda i, x: (i, ext(_nd2file(x), self.api_calls)))
        self.w_cmp = Worker(lambda i, x: (i, query(x, self.api_calls)))
        self.workers.extend([self.w_ext, self.w_cmp, self.w_par])
        self.matches = {}
        self.time0 = {}

    def on_new_det(self, t:Track, img_roi):
        self.w_ext.put(t, img_roi)

    def loop(self):
        while self.running:
            # sleep(0.1)
            if self.suspend == True:
                sleep(0.5)
            ret, frame = self.cap.read()

            if not ret or frame is None:
                self.cap = cv2.VideoCapture(self.source)
                # print('renewed', self.source)
                continue
            # frame = cv2.resize(frame, (0, 0), fx=.5, fy=.5)  # down-sampling
            frame_ = frame.copy()
            self.Track.step(frame_)
            if self.frame_count % INTEVAL == 0:
                self.w_det.put(frame_)
                self.storage.forget()
                self.Track.decay()
            self.api_calls['counts'] = len(self.Track.ALL)
            for t in self.Track.ALL:
                i = t.id
                now = time()
                if i in self.time0:
                    seconds = now - self.time0[i]
                else:
                    self.time0[i] = now
                    seconds = 0
                t.stay = seconds
                p = self.storage.id_map.get(i)
                if p is not None:
                    p.last_seen = now
                    if self.storage.object_type.is_overstay(seconds):
                        if not hasattr(t, 'par'):
                            self.w_par.put(t, _crop(frame_, t.box))
                            print('[overstay] id:', i, '@', self.source)
                            # TODO: save alerts
            self._post_det_procedure()
            self._post_ext_procedure()
            self._post_cmp_procedure(frame_)
            self._post_reg_procedure()
            if PAR:
                self._post_par_procedure()
            # if not self.control_queue.empty():
            #     x, y = self.control_queue.get()
            #     self.click_handle(frame_, x, y)
            self._render(frame)
            # print(self.display_queue.qsize())
            # print(self.w_cmp.p.qsize(), self.w_cmp.q.qsize())
            self.display_queue.put(frame[...,::-1])  # give RGB
            self.frame_count += 1
        self._kill_workers()

    def _post_ext_procedure(self):
        if not self.w_ext.p.empty():
            t, feature = self.w_ext.get()
            t.feature = feature
            self.w_cmp.put(t, feature)
            self.storage.reg(feature)
            self.api_calls['register'] += 1

    def _post_cmp_procedure(self, frame_):
        if not self.w_cmp.p.empty():
            t, ret = self.w_cmp.get()
            i = ret.get('id')
            if i is not None:
                t.similarity = ret.get('similarity')
                if t.similarity > SIMILARITY_THRESHOLD:
                    c = colors[hash(i or 0) % 256]
                    # print(t.id, 'color', c)
                    if i in self.matches:
                        f = self.matches[i]
                        if t > f:
                            f.color = Track.color
                            f.id = int(f.id)
                            f.similarity = 0
                            self.matches[i] = t
                    else:
                        self.matches[i] = t
                    self.matches[i].color = c
                    self.matches[i].id = i

    def _post_reg_procedure(self):
        if not self.q_reg.empty():
            self.q_reg.get()
            for t in self.Track.ALL:
                if t.feature is not None:
                    self.w_cmp.put(t, t.feature)

    def _post_par_procedure(self):
        if not self.w_par.p.empty():
            t, att = self.w_par.get()            # person attributes
            setattr(t, 'par', att)

    def _render(self, frame):
        super()._render(frame)
        for t in self.Track.ALL:
            x, y, w, h = map(int, t.box)
            if t.visible:
                if hasattr(t, 'stay'):
                    t.text(frame, 'sec:%d' % int(t.stay), x + 3, y + h - 3, .6, 2)
                if hasattr(t, 'par'):
                    y += h//4
                    for a, m in zip(ATTRIBUTES, t.par):
                        if a == 'Female' and not m:
                            a = 'Male'
                            m = True
                        if m:
                            cv2.putText(frame, a, (x + w + 3, y),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, t.color, 2)
                            y += h//8
