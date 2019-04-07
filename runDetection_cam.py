# Providing two mode, image passing and live detection

######################################################
# TODO:
# (1) Make a loop for capturing frame from the video
# (2) The camera calibration
# (3) Use the result from the detection
######################################################



import argparse
import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import json
import time


from PIL import Image



# do not change this
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
	  (im_height, im_width, 3)).astype(np.uint8)
# Do not change this
def run_inference_for_single_image(image, graph):
	with graph.as_default():
		with tf.Session() as sess:
			  # Get handles to input and output tensors
			ops = tf.get_default_graph().get_operations()
			all_tensor_names = {output.name for op in ops for output in op.outputs}
			tensor_dict = {}
			for key in [
				  'num_detections', 'detection_boxes', 'detection_scores',
				  'detection_classes', 'detection_masks'
				]:
				tensor_name = key + ':0'
				if tensor_name in all_tensor_names:
				  tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
					  tensor_name)
			if 'detection_masks' in tensor_dict:
				# The following processing is only for single image
				detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
				detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
				# Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
				real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
				detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
				detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
				detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
					detection_masks, detection_boxes, image.shape[0], image.shape[1])
				detection_masks_reframed = tf.cast(
					tf.greater(detection_masks_reframed, 0.5), tf.uint8)
				# Follow the convention by adding back the batch dimension
				tensor_dict['detection_masks'] = tf.expand_dims(
					detection_masks_reframed, 0)
			image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

			  # Run inference
			output_dict = sess.run(tensor_dict,
									 feed_dict={image_tensor: np.expand_dims(image, 0)})

			  # all outputs are float32 numpy arrays, so convert types as appropriate
			output_dict['num_detections'] = int(output_dict['num_detections'][0])
			output_dict['detection_classes'] = output_dict[
				  'detection_classes'][0].astype(np.uint8)
			output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
			output_dict['detection_scores'] = output_dict['detection_scores'][0]
			if 'detection_masks' in output_dict:
				output_dict['detection_masks'] = output_dict['detection_masks'][0]
	return output_dict

def convert_image_to_cv(image):
	image_cv = cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)
	return image_cv

def convert_cv_to_image(image_cv):
	image = Image.fromarray(cv2.cvtColor(image_cv,cv2.COLOR_BGR2RGB))
	return image
# Boolean. [String] 
# Check path
def image_path_valid(image_path):
	if image_path.endswith(".png") or image_path.endswith(".jpg") or image_path.endswith(".jpeg"):
		return True
	else:
		return False

# Boolean. [String] 
# Check path
def model_path_valid(model_path):
	if model_path.endswith(".pb") :
		return True
	else:
		return False

# Dictionary. [String] 
# Convert the label mapping json file into dictionary
def load_label(jsonfile):
	label_mapping = {}
	json_file = open(jsonfile)
	json_str = json_file.read()
	json_data = json.loads(json_str)["item"]
	for i in json_data:
		label_mapping[i["id"]] = i["name"]
	print("label_mapping: {}".format(label_mapping))
	return label_mapping

# Object. [String] 
# Load the model
def load_model(model_path):
	if not model_path_valid(model_path):
		raise ValueError("Invalid model path.")
	detection_graph = tf.Graph()
	with detection_graph.as_default():
		od_graph_def = tf.GraphDef()
		with tf.gfile.GFile(model_path, 'rb') as fid:
			serialized_graph = fid.read()
			od_graph_def.ParseFromString(serialized_graph)
			tf.import_graph_def(od_graph_def, name='')
	return detection_graph

def read_image(image_path):
	# use pil to open image
	# check image_path
	if image_path_valid(image_path):
		print(image_path)
		image = Image.open(image_path)
		image = image.convert('RGB')
		return image
	else:
		raise ValueError("Image path invalid.") 
		return None



# ******
# to be done 
# ******
def video_capture(take_photo=False):
	vc = cv2.VideoCapture(0)

	if vc.isOpened(): 
		# time.sleep(3)
		rval, frame = vc.read()
		if not take_photo:
			return frame
	else:
		# rval = False
		return False

	if take_photo:
		cv2.namedWindow('preview',cv2.WINDOW_NORMAL)
		cv2.resizeWindow('preview', 300,300)
		while rval:
			rval, frame = vc.read()
			key = cv2.waitKey(20)
			resize = cv2.resize(frame,(int(frame.shape[1]*0.5),int(frame.shape[0]*0.5)))
			
			cv2.imshow("preview", resize)
			if key == 27: # exit on ESC
				cv2.imwrite('capture.png',frame)
				return frame

	return False

#######################
# PICK CORNER 
# Pick the corners of the working space,
# Those corners will be regarded as the 4 corners in the plane
# 
# Read the camera and wait untill all corners are picked
#######################
def click_pick(event,x,y):
	return None
def pick_corners():
	i = 4
	corners = []
	# cv2.namedWindow("Camera Calibration")
	# cv2.setMouseCallback("Camera Calibration",click_pick)

	plt.ion()
	# fig = plt.figure('Camera Calibration')
	# vc = cv2.VideoCapture(0)
	# if vc.isOpened(): # try to get the first frame
	# 	rval, frame = vc.read()
	# else:
	# 	rval = False
	frame = video_capture()
	while i > 0 :
		# rval, frame = vc.read()
		pt = plt.ginput(n=1, timeout=-1)
		pt = np.array(pt)
		print(pt.size)
		# return
		if pt.size > 0 and pt[0,0] > 0:
			corners.append(pt)
			i = i-1
			plt.figure(fig.number)
			plt.plot(pt[:,0],pt[:,1], 'bx', markersize=5)
			print(corners)

		plt.pause(0.05)
		plt.imshow(frame)
	plt.close()
	return corners

def cal_P2P(corners):
	objectpoints = corners
	ret, rvec, tvec = cv2.solvePnP(objpoints, corners, mtx, dist, flags = cv2.CV_EPNP)

# ******
# to be done 
# ******
def image_preprocessing(image):
	return image


def image_detection(model,image):
	output_dict = run_inference_for_single_image(image, model)
	# select only with confidence over 0.5
	return [
	  (name, score, box.tolist())
	  for (box, name, score) in
	  zip(output_dict['detection_boxes'], output_dict['detection_classes'], output_dict['detection_scores'])
	  if score > 0.5]
	# return output_dict

def visualize_result(image,result_arr,label_map):
	Image_cv = convert_image_to_cv(image)
	img = Image_cv
	font = cv2.FONT_HERSHEY_SIMPLEX
	box_num = len(result_arr)
	print("box num =",box_num)

	width, height = image.size
	print("Image height: {} x width: {}".format(height, width))

	for abox in result_arr:
		nor_ymin,nor_xmin,nor_ymax,nor_xmax = abox[2]
		score = int(abox[1]*100)
		class_index = abox[0]
		classname = label_map[class_index]

		(xmin, xmax, ymin, ymax) = (int(nor_xmin * width), int(nor_xmax * width),
									int(nor_ymin * height), int(nor_ymax * height))
		img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color=(255,0,0), thickness=3)
		# write text above the box
		img = cv2.putText(img, classname, (xmin,ymin), font,1,(0,0,0),2,cv2.LINE_8)
		img = cv2.putText(img, "score:{}".format(score), (xmin,ymax), font,1,(0,0,0),2,cv2.LINE_8)

	plt.axis('off')
	plt.imshow(convert_cv_to_image(img))
	plt.show()
	return None

def main():
	# Obtain the arguments
	parser = argparse.ArgumentParser()
	# parser.add_argument('-m', '--model_path', metavar='model_path',required=True, type=str)
	parser.add_argument('-i', '--img_path',metavar='img_path', type=str)
	parser.add_argument('-l', '--label_map',metavar='label_map',required=True, type=str)
	parser.add_argument('--live_detect', action='store_true')
	args = parser.parse_args()
	# model_path = args.model_path
	model_path = "./frozen_inference_graph.pb"
	img_path = args.img_path
	live_detect = args.live_detect
	label_map_path = args.label_map


	Image = None
	try:
		# If in live detection mode
		if live_detect:
			# Do camera callibration, obtain the parameters
			# Capture the image of the board first
			# Point out the 4 corners of the board in the image
			# corners_arr = pick_corners()
			# Calculate the matrix
			# Capture a frame from video


			Image_cv = video_capture(True)
			if Image_cv is None:
				raise ValueError("No Image")
			else:
				Image = convert_cv_to_image(Image_cv)
		else:
			# If in Image passing mode
			# read image
			Image = read_image(img_path)

		# Image.show()
		# quit()
		# Load label mapping
		label_map_dict = load_label(label_map_path)

		# Load model
		model = load_model(model_path)
		# Do necessary preprocessing
		processed_image = image_preprocessing(Image)

		# Convert image to nparray
		image_np = load_image_into_numpy_array(processed_image)

		timeStart = time.time()
		# print("Time Start: ",timeStart)
		# Fit the image to the model
		detection_result = image_detection(model,image_np)
		# The result returned is a list of boxes, which contains the class label, score and box coordinates
		# print(detection_result)
		timeEnd = time.time()
		# print("Time End: ",timeEnd)
		time_used = timeEnd - timeStart
		print("Time used: ",time_used)

		# Visualize the result 
		visualize_result(Image,detection_result,label_map_dict)

		# If in live detection  mode

			# Callibrate the object point from image point


	except ValueError as err:
		print("Invalid Value!!!!!",err.args[0])



main()