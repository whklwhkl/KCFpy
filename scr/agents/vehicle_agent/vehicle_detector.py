from .models import *  # set ONNX_EXPORT in models.py
from .utils.datasets import *
from .utils.utils import *

from skimage.measure import compare_ssim

class Vehicle_Detector():
	def __init__(self, opt, scene = 0):

		#If carpark barrier scene
		if scene == 1:
			self.max_area = 8000
		#If pickup point
		else:
			self.max_area = 3500

		# Initialize Device
		self.device = 'cuda'

		self.opt = opt

		#Initialise Model and Load Model
		self.model = Darknet(opt.cfg, opt.img_size)

		if opt.weights.endswith('.pt'):
			model.load_state_dict(torch.load(opt.weights, map_location = self.device)['model'])
		else: 
			_ = load_darknet_weights(self.model, opt.weights)

		self.model.to(self.device).eval().half()

		# Get classes and colors
		self.classes = load_classes(parse_data_cfg(opt.data)['names'])

		if self.opt.conf_thres == 0.3:
			self.scene = 1
		else:
			self.scene = 0

		print('--- Detector Initialised ---')

	#Function to perform detecetion
	def detect(self, img):
		#Pre-process
		img = self.pre_process(img)
		img = torch.from_numpy(img).half()
		img = img.to(self.device)

		if img.ndimension() == 3:
			img = img.unsqueeze(0)

		#print(img.shape)

		#Inference
		pred = self.model(img)[0]

		#print(pred)

		# Apply NMS
		pred = non_max_suppression(pred.float(), self.opt.conf_thres, self.opt.nms_thres)

		det = self.process_detections(pred)

		if det is not None:
			qa_boxes = self.qa(det)

			return qa_boxes
		else:
			return None

	#Function to perform pre-processing
	def pre_process(self, img):
		img = letterbox(img, new_shape = (256, 416))[0]

		# Convert
		img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
		img = np.ascontiguousarray(img, dtype=np.float16 if self.opt.half else np.float32)  # uint8 to fp16/fp32
		img /= 255.0  # 0 - 255 to 0.0 - 1.0

		return img

	#Function to process detections to filter desired class
	def process_detections(self, pred):
		for i, det in enumerate(pred):
			
			filtered_tensor = []

			if det is not None:
				for i, item in enumerate(det):
					#COCO Car, Bus and Truck classes
					if item[6] == 2 or item[6] == 5 or item[6] == 7:
						#print(item[6])
						filtered_tensor.append(item.unsqueeze(dim=0))

			if filtered_tensor == []:
				continue

			det = torch.cat(filtered_tensor)

			if det is not None and len(det):
				det[:, :4] = scale_coords([256, 416], det[:, :4], (720, 1280, 3)).round()

		if det is not None:
			return det
		else:
			return None

	#Function to perform Quality Assessment
	def qa(self, detections):
		bbox_list = []

		for item in detections:
			coordinates = item[:4].type(torch.int).tolist()

			#Filter zeroes
			if coordinates[0] <= 0 or coordinates[1] <= 0 or coordinates[2] <= 0 or coordinates[3] <= 0: continue

			width = coordinates[2] - coordinates[0]
			height = coordinates[3] - coordinates[1]
			area = width * height

			#Filter condition for barrier scene
			if self.scene == 1 and area >= self.max_area:
				bbox_list.append(coordinates)
			#Filter condition for pickup point scene
			elif coordinates[2] <= 1220 and coordinates[3] >= 215 and area >= self.max_area:
				bbox_list.append(coordinates)


		return bbox_list

#Function to resize image in rectangle form
def letterbox(img, new_shape=(416, 416), color=(128, 128, 128),
              auto=True, scaleFill=False, scaleup=True, interp=cv2.INTER_AREA):
    # Resize image to a 32-pixel-multiple rectangle https://github.com/ultralytics/yolov3/issues/232
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = max(new_shape) / max(shape)
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = new_shape
        ratio = new_shape[0] / shape[1], new_shape[1] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=interp)  # INTER_AREA is better, INTER_LINEAR is faster
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    
    return img, ratio, (dw, dh)



