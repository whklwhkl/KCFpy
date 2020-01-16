from ..agent import *
#from .agent import _crop, _nd2file
from time import time
import torch

#Custom Classes
from .vehicle_tracker import Vehicle_Tracker
from .vehicle_attributes import Vehicle_Attributes
from .vehicle_detector import Vehicle_Detector
from .vehicle_feature_extractor import Vehicle_Feature_Extractor


#Configuration Values
COSINE_UPPER_THRES = .9 #Match if higher
COSINE_LOWER_THRES = .5 #Create new vehicle instance if lower 
OVERSTAY_THRES = 1800 #3 mins for 10 FPS

#Vehicle Agent Class
class VehicleAgent:
    def __init__(self, source, detector_opt, thread_no, host='localhost'):
        #super().__init__(source, host)

        self.source = os.path.expanduser(source)
        self.cap = cv2.VideoCapture(self.source)

        self.thread_no = thread_no

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FPS, 12)

        self.display_queue = Queue(32)
        self.control_queue = Queue(1)
        self.frame_count = 0

        self.running = True
        self.suspend = False

        self.q_reg = Queue(32)  # register queue
        #self.storage = Storage(make_object_type())

        #Queue for GUI display
        self.display_queue = Queue(32)

        #Init Vehicle Tracker Class
        self.vehicle_tracker = Vehicle_Tracker()

        #Initialise Detector Model
        self.vehicle_detector = Vehicle_Detector(detector_opt)

        #Initialise Vehicle feature extractor
        self.vehicle_feature_extractor = Vehicle_Feature_Extractor()
        
        #Initialise Vehicle Attribute model
        self.vehicle_attribute = Vehicle_Attributes(model_path = '/home/jeff/development/capitaland_poc_v2/model/vehicle_attributes.pth')
        
        #Create Worker instances
        self.worker_vehicle_tracker = Worker(lambda x,y: self.search_detections(x, y))
        self.worker_vehicle_detector = Worker(lambda x: self.vehicle_detector.detect(x))
        self.worker_vehicle_attribute = Worker(lambda x: self.vehicle_attribute.predict(x))
        #self.worker_check_overstay = Worker(self.check_overstay())
        
        print('--- Vehicle Agent Initialised ---')

        self.th = Thread(target=self.loop, daemon=True)
        self.th.start()

    def loop(self):
        while self.running:
            self.frame_count += 1
            print('Thread no: {}    Frame no: {}'.format(self.thread_no, self.frame_count))
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

            #Perform detection (pre, detect and post)
            detections = self.vehicle_detector.detect(frame_)

            if detections is not None:
                overstay_vehicles, id_list = self.search_detections(detections, frame_)


            #Function to check for overstaying vehicles
            self.check_overstay()


            self.display_queue.put(frame_[...,::-1])
        self._kill_workers()

    def post_det_procedure(self, frame):
        if self.worker_vehicle_detector.has_feedback():
            qa_boxes = self.worker_vehicle_detector.get()

            if qa_boxes is not None:
                self.worker_vehicle_tracker.put(qa_boxes, frame)

    def post_tracker_procedure(self):
        


    def reset(self):
        pass

    def save(self):
        pass

    def stop(self):
        self.running = False
        self.th.join(1)

    #Function to check if overstay
    def check_overstay(self):
        for vehicle in self.vehicle_tracker.vehicles:

            #More than 1800 frames (3 minutes, assuming 10 FPS)
            if vehicle.time >= OVERSTAY_THRES:
                vehicle.overstay = True

    #Function to Calculate Cosine Similarity with all the features
    def calculate_cosine(self, feature):
        scores = []

        for item in self.vehicle_tracker.vehicles:
            score = self.vehicle_feature_extractor.calculate_cosine_similarity(feature, item.feature)
            scores.append((score, item.id))

        return scores

    #Function to check if boxes are within limits, box1 is for buffer, box2 is for detection
    def within_limits(self, box1):
        limits = 10

        results = []

        for item in self.vehicle_tracker.vehicles:
            #Check x1 and y1
            if (item.bbox[0] <= (box1[0] + limits) and item.bbox[0] >= (box1[0] - limits)) and (item.bbox[1] <= (box1[1] + limits) and item.bbox[1] >= (box1[1] - limits)):
                results.append(True)
            else:
                results.append(False)
        
        return results

    #Function to search detections
    def search_detections(self, detections, img):
        overstay = []
        id_list = []

        for item in detections:
            #Crop Vehicle out
            coordinates = item
            #coordinates = item[:4].type(torch.int).tolist()
            roi = img[coordinates[1] : coordinates[3], coordinates[0] : coordinates[2]]

            #Get Vehicle Features
            feature = self.vehicle_feature_extractor.get_features(roi)

            #If no vehicles
            if len(self.vehicle_tracker.vehicles) == 0:
                #Get Vehicle Attributes
                attributes = self.vehicle_attribute.predict(self.vehicle_attribute.pre_process(roi))

                #Append new vehicle item and get new ID
                new_id = self.vehicle_tracker.append_vehicle(coordinates, feature, attributes)
                id_list.append((coordinates,new_id))

                continue
            else:
                score = self.calculate_cosine(feature)
                within_limits_list = self.within_limits(coordinates)
                highest_score = max(score, key = lambda item:item[0])


                hit = within_limits_list[score.index(max(score, key = lambda item:item[0]))]
                current_vehicle = self.vehicle_tracker.get_vehicle_by_id(highest_score[1])

                #Append to overstay list
                if hit and current_vehicle.overstay:
                    overstay.append((coordinates, current_vehicle.id))

                #print(highest_score[0][0][0])
                #print(highest_score[1])

                if (highest_score[0][0][0] >= COSINE_UPPER_THRES) and not hit:
                    current_vehicle.update_bbox(coordinates)
                    #current_vehicle.update_feature(roi)
                    current_vehicle.update_feature(feature)
                    current_vehicle.set_update(True)
                    #print('Updating vehicle bbox')
                    id_list.append((coordinates,current_vehicle.id))

                #Match both conditions, increment time by 1
                elif (highest_score[0][0][0] >= COSINE_UPPER_THRES) and hit:
                    if current_vehicle.overstay and current_vehicle.captured is False:
                        cv2.imwrite('/home/jeff/Desktop/output_overstay/{}_{}.jpg'.format(output_count, current_vehicle.attributes), roi)
                        current_vehicle.captured = True
                    
                    current_vehicle.update_time(1)
                    #current_vehicle.update_feature(roi)
                    #current_vehicle.update_bbox(coordinates)
                    current_vehicle.set_update(True)
                    id_list.append((coordinates,current_vehicle.id))
                    #print('Increment by 1')
                #Low confidence score but hit coordinates, possible occlusion
                elif (highest_score[0][0][0] < COSINE_UPPER_THRES) and hit:
                    if current_vehicle.overstay and current_vehicle.captured is False:
                        cv2.imwrite('/home/jeff/Desktop/output_overstay/{}_{}.jpg'.format(output_count, current_vehicle.attributes), roi)
                        current_vehicle.captured = True
                        
                    current_vehicle.update_time(1)
                    #current_vehicle.update_feature(roi)
                    current_vehicle.set_update(True)
                    id_list.append((coordinates, current_vehicle.id))
                    #print('Occlussion')
                    #break
                #Add new Vehicle
                elif (highest_score[0][0][0] <= COSINE_LOWER_THRES) and not hit:
                    #Get Vehicle Attributes
                    attributes = self.vehicle_attribute.predict(self.vehicle_attribute.pre_process(roi))
                    #tracker.append_vehicle(coordinates, roi)
                    new_id = self.vehicle_tracker.append_vehicle(coordinates, feature, attributes)
                    #print('Adding new vehicle')
                    id_list.append((coordinates, new_id))
                    break
                else:
                    #print('No action')
                    pass

        return overstay, id_list


'''
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
                    y += 16
                    for a, m in zip(ATTRIBUTES, t.par):
                        if a == 'Female' and not m:
                            a = 'Male'
                            m = True
                        if m:
                            cv2.putText(frame, a, (x + w + 3, y),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, t.color, 2)
                            y += 16
'''