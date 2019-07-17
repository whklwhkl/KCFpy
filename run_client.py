import numpy as np
import cv2
import requests


URL = 'http://192.168.20.149:6668/stream'

URL1 = 'http://192.168.20.149:6668/stream1'

URL2 = 'http://192.168.20.149:6668/stream2'

URL3 = 'http://192.168.20.149:6668/stream3'

# prepare headers for http request
content_type = 'image/jpeg'
headers = {'content-type': content_type}

def post_request(frame):
	response = requests.post(URL, files={'img': frame})

	#print('test')

cap = cv2.VideoCapture(0)

print("Running Webcam...")

while(True):
    _, frame = cap.read()
    _, encode = cv2.imencode('.jpg', frame)
    response = requests.post(URL, data=encode.tostring(), headers = headers)
    response = requests.post(URL1, data=encode.tostring(), headers = headers)
    #response = requests.post(URL2, data=encode.tostring(), headers = headers)
    #response = requests.post(URL3, data=encode.tostring(), headers = headers)
    print(response)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()