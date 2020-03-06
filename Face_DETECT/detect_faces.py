import numpy as np
import argparse
import cv2

class face_detector:
	def __init__(self,prototxt_path,model_path):
		global net
		net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
		
	def detect_faces(self,img_path):
		image = cv2.imread(img_path)
		(h, w) = image.shape[:2]
		blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
			(300, 300), (104.0, 177.0, 123.0))
		net.setInput(blob)
		detections = net.forward()
		detected_faces = []
		count = 0
		for i in range(0, detections.shape[2]):
			confidence = detections[0, 0, i, 2]
			if confidence > 0.5:
				box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
				(startX, startY, endX, endY) = box.astype("int")
				count+=1
				width = abs(endX-startX)
				height = abs(endY-startY)
				ROI = image[startY:startY+height,startX:startX+width]
				detected_faces.append(ROI)
				cv2.imwrite('./static/Results/face_detect_{}.jpg'.format(count),ROI)
				text = "{:.2f}%".format(confidence * 100)
		return count,detected_faces