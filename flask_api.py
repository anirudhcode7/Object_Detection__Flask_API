import os
from flask import Flask,flash, request, redirect, url_for,render_template, jsonify
from werkzeug.utils import secure_filename
from YOLO.Detector import Detector
import cv2
from Face_DETECT.detect_faces import face_detector
import imutils
UPLOAD_FOLDER = '\\static\\Files\\'
ALLOWED_EXTENSIONS = {"mp4","mov","jpg","jpeg","png"}
# ALLOWED_EXTENSIONS_IMAGES = {"jpg","jpeg","png"}

app = Flask(__name__)
# app.config['SESSION_TYPE'] = 'filesystem'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
	return '.' in filename and filename.rsplit('.',1)[1].lower() in ALLOWED_EXTENSIONS 


@app.route("/")
def hello():
	return render_template('layout.html')

@app.route("/upload",methods = ["GET","POST"])

def upload_file():
	if request.method == 'POST':
		# check if the post request has the file part
		if 'file' not in request.files:
			# flash('No file part')
			return render_template('upload.html', Error='No file selected')
		file = request.files['file']
		if file.filename == '':
			# flash('No Selected file')
			return render_template('upload.html', Error = 'No file Selected')
		if file and allowed_file(file.filename):
			filename = secure_filename(file.filename)
			file.save(os.path.join(os.getcwd()+UPLOAD_FOLDER,file.filename))
			face_clss = face_detector('./Face_DETECT/deploy.prototxt.txt','./Face_DETECT/res10_300x300_ssd_iter_140000.caffemodel')
			face_count,detected_faces = face_clss.detect_faces(os.path.join(os.getcwd()+UPLOAD_FOLDER,file.filename))
			print(face_count)
			# stitcher = cv2.createStitcher() if imutils.is_cv3() else cv2.Stitcher_create()
			# (status, stitched) = stitcher.stitch(detected_faces)
			# print(status,stitched)
			# cv2.imwrite(os.path.join(os.getcwd()+UPLOAD_FOLDER,'stitched_image.jpg'),stitched)
			dec = Detector('./YOLO/yolov3-spp.weights','./YOLO/yolov3-spp.cfg')
			img,class_ids,cnts = dec.detectObject(os.path.join(os.getcwd()+UPLOAD_FOLDER,file.filename))
			print(class_ids,cnts)
			cv2.imwrite(os.path.join(os.getcwd()+UPLOAD_FOLDER,'detection.jpg'),img)
			# file.save(os.path.join(os.getcwd()+UPLOAD_FOLDER,))
			out = {'Human Faces':face_count,'animals':cnts['animals'],'objects':cnts['objects']}
			return jsonify(results = out)
		else:
			# flash('Check the File Extension')
			return render_template('upload.html',Error= "Check File Extension")
	elif request.method == 'GET':
		return render_template('upload.html') 



if __name__ == '__main__':
	app.run(debug = True)