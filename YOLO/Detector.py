import cv2
import numpy as np
import matplotlib.pyplot as plt

classNames = {0: 'background', 1:'person'}
classes = None
class_names_path = './YOLO/coco.names'
with open(class_names_path, 'r') as f:
	classes = [line.strip() for line in f.readlines()]
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
# Get names of output layers, output for YOLOv3 is ['yolo_16', 'yolo_23']
def getOutputsNames(net):
    layersNames = net.getLayerNames()
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


# Darw a rectangle surrounding the object and its class name 
def draw_pred(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
# Define a window to show the cam stream on it
# Load names classes

def adjust_gamma(image, gamma=1.5):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
 
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)

class Detector:
	def __init__(self,weights_path,config_path):
		global net
		net  = cv2.dnn.readNet(weights_path,config_path)
		global human_faces	
		global animals
		self.cnts = dict()
		human_faces = ["person"]
		animals = ["bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe"]

	    # if frame_cnt %10 == 0:
	def detectObject(self,image_path):
		
	    image = cv2.imread(image_path)
	#image=cv2.resize(image, (1920, 416))
	    #image = adjust_gamma(image, gamma=1)
	    #image = cv2.transpose(image, image)
	    self.cnts = {'humans':0,'animals':0,'objects':0}
	    blob = cv2.dnn.blobFromImage(image, 1.0/255.0, (416,416), [0,0,0], True, crop=False)
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
	        #each detection  has the form like this [center_x center_y width height obj_score class_1_score class_2_score ..]
	            scores = detection[5:]#classes scores starts from index 5
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

	    # apply  non-maximum suppression algorithm on the bounding boxes
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
	        draw_pred(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))

	    # Put efficiency information.
	    t, _ = net.getPerfProfile()
	    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
	    cv2.putText(image, label, (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0))
	    # cv2.imshow("object detection", image)

	    return image,class_ids,self.cnts