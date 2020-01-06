from .agent import Agent


PAR = True
ATTRIBUTES = ['Female', 'Front', 'Side', 'Back', 'Hat',
              'Glasses', 'Hand Bag', 'Shoulder Bag', 'Backpack',
              'Hold Objects in Front', 'Short Sleeve', 'Long Sleeve',
              'Long Coat', 'Trousers', 'Skirt & Dress']


class PersonAgent(Agent):

    def __init__(self, source):
        super().__init__(source)
        self.q_reg = Queue(32)  # register queue
        self.api_calls = {k: 0 for k in ['register', 'detection', 'feature',
                                         'query', 'refresh', 'attributes']}
        PAR_URL = 'http://%s:6668/par' % host
        EXT_URL = 'http://%s:6667/ext' % host
        CMP_URL = 'http://%s:6669/{}' % host

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

        self.CMP_URL = CMP_URL
        self.ext = ext
        self.up = up
        self.w_par = Worker(lambda i, x: (i, par(_nd2file(x), self.api_calls)))
        self.w_ext = Worker(lambda i, x: (i, ext(_nd2file(x), self.api_calls)))
        self.w_cmp = Worker(lambda i, x: (i, query(x, self.api_calls)))
        self.workers.extend([self.w_ext, self.w_cmp, self.w_par])
        self.on_det_funcs = self.w_ext.put

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
            self._post_ext_procedure()
            self._post_cmp_procedure(frame_)
            self._post_reg_procedure()
            if PAR:
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

        def _post_ext_procedure(self):
            if not self.w_ext.p.empty():
                t, feature = self.w_ext.get()
                t.feature = feature
                self.w_cmp.put(t, feature)

    def _post_cmp_procedure(self, frame_):
        if not self.w_cmp.p.empty():
            t, ret = self.w_cmp.get()
            i = ret.get('id')
            # assert isinstance(i, str)
            c = colors[ret.get('idx')]
            if i is not None and i != -1:
                t.similarity = ret.get('similarity')
                # sim_ema = self.sim_ema.setdefault(i, MovingAverage(
                #     t.similarity, conf_band=2.5))
                if PAR:
                    self.w_par.put(t, _crop(frame_, t.box))
                if t.similarity > .94:
                # if sim_ema(t.similarity):
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
