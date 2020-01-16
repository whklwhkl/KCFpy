import torch
from sklearn.metrics.pairwise import cosine_similarity

import cv2
from PIL import Image

import numpy as np

from torchvision import transforms

class Vehicle_Feature_Extractor():
	def __init__(self):
		self.model = torch.jit.load('./models/reid.zip')
		self.model.eval()
		self.transform = transforms.Compose([transforms.Resize((224,224)),
								transforms.ToTensor(),
								transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])])

		print('---Vehicle Feature Extractor Initialised---')


	#Function to calculate cosine similarity
	def calculate_cosine_similarity(self, img_1, img_2):
		return cosine_similarity(img_1, img_2)

	#Function to perform feature extraction
	def get_features(self, img):
		#Pre-processing
		img = Image.fromarray(img)
		transformed = self.transform(img).unsqueeze_(0).cuda()

		#Inference
		feature = self.model(transformed).detach().cpu().numpy()

		return feature