import cv2

class Record:
	def __init__(self, output_file):
		self.hundred_frames = []
		self.output_file = output_file + '.avi'

	#Function that returns true if 100 frames are stored
	def check_save(self):
		return len(self.hundred_frames) == 100

	#Function to save video
	def save_video(self):
		out = cv2.VideoWriter(self.output_file, cv2.VideoWriter_fourcc(*'XVID'), 10.0, (640, 360))

		for item in self.hundred_frames:
			out.write(item)

		out.release()

		print('Saved video in: ', self.output_file)

	#Function to add frame
	def add_frame(self, frame):
		frame = cv2.resize(frame, (640, 360))
		self.hundred_frames.append(frame)