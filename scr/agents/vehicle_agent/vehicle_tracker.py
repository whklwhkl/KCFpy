
#Vehicle Class
class Vehicle():
    def __init__(self, id, bbox, feature, attributes):
        self.id = id
        self.bbox = bbox
        self.feature = feature
        self.time = 0
        self.miss = 25
        self.update = False
        self.overstay = False
        self.captured = False
        self.attributes = attributes

    #Function for updating bbox when vehicle movement is detected (x1, x1) +- 10 exceeds
    def update_bbox(self, bbox):
        self.bbox = bbox

    #Function to updating the feature of the cropped ROI
    def update_feature(self, feature):
        self.feature = feature

    #Function to update/increment the time value of the vehicle
    def update_time(self, increment):
        self.time += increment

    #Function to decrease the miss value (HP) of the Vehicle if there is a mismatch or it is not detected
    def decrease_miss(self):
        self.miss -= 1

    #Function to reset the miss value and adds the contra of frames that was not detected to the time variable
    def reset_miss(self):
        contra = 25 - self.miss
        self.time += contra
        self.miss = 25

    #Function to set the update flag of the Vehicle object to check if object is updated in the main loop
    def set_update(self, condition):
        self.update = condition

class Vehicle_Tracker():
    def __init__(self):
        self.current_id = 1
        self.vehicles = []
        print('--- Vehicle Tracker Initialised ---')

    #Function to append vehicle
    def append_vehicle(self, bbox, feature, attributes):
        self.vehicles.append(Vehicle(self.current_id, bbox, feature, attributes))
        self.current_id += 1

        return self.current_id - 1

    #Function to get vehicle by id
    def get_vehicle_by_id(self, id):
        for vehicle in self.vehicles:
            if vehicle.id == id:
                return vehicle