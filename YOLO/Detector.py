import cv2
import numpy as np
import matplotlib.pyplot as plt

class_names_path = './YOLO/coco.names'
classes = None
with open(class_names_path, 'r') as f:
	classes = [line.strip() for line in f.readlines()]
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

def getOutputsNames(net):
    layerNames = net.getLayerNames()
    return [layerNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

class Detector:
	def __init__(self,weights_path,config_path):
		global net
		net  = cv2.dnn.readNet(weights_path,config_path)
		global human_faces	
		global animals
		self.cnts = dict()
		human_faces = ["person"]
		animals = ["bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe"]


	def detectObject(self,image_path):
		
	    image = cv2.imread(image_path)

	    self.cnts = {'humans':0,'animals':0,'objects':0}
	    blob = cv2.dnn.blobFromImage(image, 1.0/255.0, (608,608), [0,0,0], True, crop=False)
	    Width = image.shape[1]
	    Height = image.shape[0]
	    net.setInput(blob)
	    outs = net.forward(getOutputsNames(net))
	    class_ids = []
	    confidences = []
	    boxes = []
	    conf_threshold = 0.1
	    nms_threshold = 0.1 

	    for out in outs: 
	        for detection in out:
	            scores = detection[5:]
	            class_id = np.argmax(scores)
	            confidence = scores[class_id]
	            if confidence > 0.5:
	                center_x = int(detection[0] * Width)
	                center_y = int(detection[1] * Height)
	                w = int(detection[2] * Width)
	                h = int(detection[3] * Height)
	                x = center_x - w / 2
	                y = center_y - h / 2
	                class_ids.append(class_id)
	                confidences.append(float(confidence))
	                boxes.append([x, y, w, h])

	    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

	    for i in indices:
	        i = i[0]
	        box = boxes[i]
	        x = box[0]
	        y = box[1]
	        w = box[2]
	        h = box[3]
	        lbl = str(classes[class_ids[i]])
	        if lbl in human_faces:
	        	self.cnts['humans'] +=1
	        elif lbl in animals:
	        	self.cnts['animals']+=1
	        else:
	        	self.cnts['objects']+=1
	        label = str(classes[class_ids[i]])
	        color = COLORS[class_ids[i]]
	        cv2.rectangle(image, (round(x), round(y)), (round(x+w), round(y+h)), color, 2)
	        cv2.putText(image, label, (round(x)-10,round(y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

	    t, _ = net.getPerfProfile()
	    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
	    cv2.putText(image, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))

	    return image,class_ids,self.cnts