from .agent import *
from time import time
from datetime import datetime
import os

PAR = True
ATTRIBUTES = ['Female', 'Front', 'Side', 'Back', 'Hat',
              'Glasses', 'Hand Bag', 'Shoulder Bag', 'Backpack',
              'Hold Objects in Front', 'Short Sleeve', 'Long Sleeve',
              'Long Coat', 'Trousers', 'Skirt & Dress']

#Output
base_dir = './output'

#Storage Memory Parameter, forget vehicle if not seen > STORAGE_MEMORY
STORAGE_MEMORY = 8

#Car x1y1 Distance Parameter
DISTANCE_THRESHOLD = 10

#Feature Matching Parameters
COSINE_UPPER_THRESH = 0.9
COSINE_LOWER_THRESH = 0.85

#Feature Smoothing Parameter
FEATURE_MOMENTUM = 1

#Overstay Parameter in Seconds
OVERSTAY_THRESH = 60


def make_object_type():
    class Vehicle:

        current_id = 0

        def __init__(self, feature, box):
            cls = type(self)
            self.id = str(cls.current_id)
            self.feature = feature
            self.box = box
            self.first_seen = self.last_seen = time()
            self.imgs = []
            cls.current_id += 1

        def __str__(self):
            return self.id

        @staticmethod
        def is_overstay(seconds):
            return seconds > OVERSTAY_THRESH

        @staticmethod
        def is_alike(similarity):
            return similarity >= COSINE_UPPER_THRESH

        @staticmethod
        def is_far(delta):
            return delta > DISTANCE_THRESHOLD

        #Function to compare old and new box
        @staticmethod
        def within_limits(old_box, new_box):

            #x1 +- Thresh and y1 +- thresh
            if (old_box[0] + DISTANCE_THRESHOLD >= new_box[0] and old_box[0] - DISTANCE_THRESHOLD <= new_box[0] ) and (old_box[1] + DISTANCE_THRESHOLD >= new_box[1] and old_box[1] - DISTANCE_THRESHOLD <= new_box[1] ):
                return True
            else:
                return False

    return Vehicle


class Storage:
    def __init__(self, object_type, memory=STORAGE_MEMORY):
        self.id_map = {}
        #Default memory = 180, else 60
        self.memory = memory
        self.frames = []
        self.object_type = object_type

    def add(self, feature, box):
        p = self.object_type(feature, box)
        self.id_map[p.id] = p

    def reg(self, feature, box):
        feature = np.array(feature, np.float32)

        #print(self.id_map.keys())

        if len(self.id_map):
            #Query feature and get best match
            q = self.query(feature)
            p = self.id_map[q['id']]

            #If x1, y1 is within limits
            if self.object_type.within_limits(p.box, box) and q['similarity'] < COSINE_UPPER_THRESH and q['similarity'] >= COSINE_LOWER_THRESH:
                p.feature = p.feature * FEATURE_MOMENTUM + feature * (1-FEATURE_MOMENTUM)
                p.box = p.box * FEATURE_MOMENTUM + box * (1-FEATURE_MOMENTUM)

            #Update feature if similarity is high
            elif self.object_type.is_alike(q['similarity']):
                p.feature = p.feature * FEATURE_MOMENTUM + feature * (1-FEATURE_MOMENTUM)
                p.box = p.box * FEATURE_MOMENTUM + box * (1-FEATURE_MOMENTUM)

            #Register new vehicle if lower than thresh
            elif q['similarity'] < COSINE_LOWER_THRESH:
                self.add(feature, box)
            #Possible occlusion or too many similar vehicles in Storage
            else:
                pass

        #First case, if storage is empty add new vehicle
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

    def add_video(frame):
        self.frames.append(frame)

    @staticmethod
    def compare(feature1, feature2):
        f1 = np.array(feature1)
        f2 = np.array(feature2)
        cos = np.dot(f1, f2)/np.linalg.norm(f1)/np.linalg.norm(f2)
        return cos
        # return np.exp(cos - 1)

#Function to perform cropping based on tracking boxes
def _crop(frame, trk_box):
    H, W, _ = frame.shape
    left, t, w, h = map(int, trk_box)
    left = max(left, 0)
    t = max(t, 0)
    r = min(left + w, W)
    b = min(t + h, H)
    crop = frame[t: b, left: r, :]

    return crop

class VehicleAgent(Agent):
    #Scene 0 = Drop off point, scene 1 = barrier
    def __init__(self, source, detector_opt, host='localhost', scene = 0):
        super().__init__(source, host)

        #Current date
        self.current_date = datetime.now().date()

        #Create directories
        source_dir = source.split('/')
        source_dir = source_dir[len(source_dir) - 1]

        #Output directory
        self.output_dir = os.path.join(os.path.join(base_dir, str(source_dir)), str(self.current_date))

        #Output log .txt
        self.output_log = os.path.join(os.path.join(base_dir, str(source_dir)), str(self.current_date) + '/log.txt')

        #Create directory if it does not exist
        if not os.path.exists(os.path.join(os.path.join(base_dir, str(source_dir)), str(self.current_date))):
            os.makedirs(os.path.join(os.path.join(base_dir, str(source_dir)), str(self.current_date)))

        #self.q_reg = Queue(32)  # register queue
        self.api_calls = {k: 0 for k in ['register',
                                         'detection',
                                         'feature',
                                         'query',
                                         'refresh',
                                         'attributes',
                                         'counts']}
        #If drop off point
        if scene == 0:                              
            self.storage = Storage(make_object_type())
        #If carpark
        else:
            self.storage = Storage(make_object_type(), memory = 60)

        from .vehicle_agent.vehicle_detector import Vehicle_Detector
        #from .vehicle_agent.attribute import VehicleAttributes
        from .vehicle_agent.vehicle_attributes import Vehicle_Attributes
        from .vehicle_agent.vehicle_feature_extractor import Vehicle_Feature_Extractor

        #Initialise Detector Model
        #Increase conf_thresh if not carpark barrier scene
        if scene != 0:
            detector_opt.conf_thres = 0.7

        self.vehicle_detector = Vehicle_Detector(detector_opt, scene)

        #Initialise Vehicle feature extractor
        self.vehicle_feature_extractor = Vehicle_Feature_Extractor()

        #Initialise Vehicle Attribute model
        #public github model
        self.vehicle_attribute = Vehicle_Attributes()

        #XZ Model
        #self.vehicle_attribute = VehicleAttributes()

        #Perform detection and push to output queue
        def det(img):
            detections = self.vehicle_detector.detect(img)
            return detections

        #Perform feature extraction and push to queue
        def ext(img):
            feature = self.vehicle_feature_extractor.get_features(img)
            return feature

        #Perform feature query on storage
        def query(feature):
            try:
                return self.storage.query(feature)
            except AssertionError:
                return {}

        #Perform Vehicle Attributes
        def var(t, img):
            ret = self.vehicle_attribute.predict(img)
            return t, ret

        #Worker for detetion
        self.w_det = Worker(lambda x : (x, det(x)))
        
        self.w_var = Worker(lambda i, x: (i, var(i, x)), debug = True)

        #Takes in Tracker Object and cropped image in process queue
        self.w_ext = Worker(lambda i, x : (i, ext(x)))

        #Takes in Tracker Object and Extracted features
        self.w_cmp = Worker(lambda i, x: (i, query(x)))

        #Worker for Vehicle Attributes, takes in tracker object and cropped image
        #self.w_par = Worker(lambda i, x : (i, var(x)))

        self.workers = [self.w_det, self.w_ext, self.w_cmp, self.w_var]

        self.matches = {}
        self.time0 = {}
        self.reported = set()

        #Create Thread object for each class instance and begin thread
        self.th = Thread(target=self.loop, daemon=True)
        self.th.start()

    #Function to push Track object and cropped image ROI for feature extraction
    def on_new_det(self, t:Track, img_roi):
        self.w_ext.put(t, img_roi)

    #Function to check current system date
    def check_date(self):
        if datetime.now().date() > self.current_date:
            os.makedirectory(str(datetime.now().date()))
            self.current_date = datetime.now().date()

    #Main loop
    def loop(self):
        while self.running:

            #Check date every 100 frames
            if self.frame_count % 100 == 0:
                self.check_date()

            if self.suspend == True:
                sleep(0.5)
            ret, frame = self.cap.read()

            if not ret or frame is None:
                self.cap = cv2.VideoCapture(self.source)
                continue

            frame_ = frame.copy()

            self.Track.step(frame_)

            if self.frame_count % INTEVAL == 0:
                #Push new frame into detection worker queue
                self.w_det.put(frame_)

                #Remove IDs that have not been seen for too long
                self.storage.forget()

                #Remove dead trackers
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
                            self.w_var.put(t, _crop(frame_, t.box))
    
                        if hasattr(t, 'par'):
                            #Output into folder
                            cv2.imwrite(self.output_dir + '/' + str(i) + '_' + str(t.par[1][0]) + str(t.par[1][1]) + '.jpg', _crop(frame_, t.box))
                            #End of output

                            if i not in self.reported:
                                self.reported.add(i)
                                print('[overstay] id:', i, '@', self.source)

                                with open(self.output_log, 'a') as f:
                                    message = 'Overstay id: {} Frame no: {} @ {}'.format(i, self.frame_count, self.source)
                                    f.write(message)
                                
            #Perform post detection (KCF tracker) procedures and push to w_ext
            self._post_det_procedure()

            #Perform post feature extraction, push to w_cmp queue for feature comparison and register in storage
            self._post_ext_procedure()

            #Perform post comparison procedures, set matches
            self._post_cmp_procedure(frame_)

            #Perform post par procedure
            self._post_par_procedure()

            #Render frame and add it to display worker queue
            self._render(frame)
            self.display_queue.put(frame[...,::-1])  # give RGB
            self.frame_count += 1

        self._kill_workers()

    #Function to convert YOLO bboxes
    def convert(self, boxes):
        boxes_ = []

        for b in boxes:
            boxes_.append(b)

        boxes = np.array(boxes_)
        boxes[:, 2: 4] -= boxes[:, :2]

        return boxes[:, :4]

    #Function to perform post detection procedure
    def _post_det_procedure(self):
        #If worker queue is not empty
        if self.w_det.has_feedback():
            frame_, boxes = self.w_det.get()

            #if len(boxes):
            #If detection boxes are not empty, update track boxes
            if boxes is not None and boxes != []:
                #boxes = _cvt_ltrb2ltwh(boxes)
                boxes = self.convert(boxes)
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

    #Function to perform post FE procedure
    def _post_ext_procedure(self):

        #If worker queue is not empty
        if self.w_ext.has_feedback():
            t, feature = self.w_ext.get()
            #print(feature[0])
            t.feature = feature[0]
            self.w_cmp.put(t, feature[0])
            self.storage.reg(feature[0], t.box)
            self.api_calls['register'] += 1

    #Function perform feature similarity comparison
    def _post_cmp_procedure(self, frame_):
        if self.w_cmp.has_feedback():
            t, ret = self.w_cmp.get()
            i = ret.get('id')

            if i is not None:
                t.similarity = ret.get('similarity')

                if t.similarity > COSINE_UPPER_THRESH:
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

    def _post_par_procedure(self):
        if not self.w_var.p.empty():
            t, att = self.w_var.get()
            setattr(t, 'par', att)

    #Render frame
    def _render(self, frame):
        #super()._render(frame)
        for t in self.Track.ALL:
            x, y, w, h = map(int, t.box)
            r = x + w
            b = y + h
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

                #Show only overstay
                if self.storage.object_type.is_overstay(t.stay):
                    cv2.rectangle(frame, (x, y), (r, b), t.color, 2)
                #End

                    if t.visible:
                        if hasattr(t, 'stay'):
                            t.text(frame, 'sec:%d' % int(t.stay), x + 3, y + h - 3, .6, 2)
                        if hasattr(t, 'par'):
                            for item in t.par[1]:
                                y += 20
                                cv2.putText(frame, str(item), (x + w + 3, y),
                                                cv2.FONT_HERSHEY_SIMPLEX, 1., t.color, 2)
                                cv2.putText(frame, str(item), (x + w + 3, y),
                                                cv2.FONT_HERSHEY_SIMPLEX, 1., t.color, 2)
