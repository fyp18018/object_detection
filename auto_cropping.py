import os
import pandas as pd
import cv2
import argparse
import matplotlib.pyplot as plt

class Preset:
	'''Preset class'''
	def __init__(self, filter_size=17, canny_thresh1=30, canny_thresh2=50, use_grayscale=True):
		self.filter_size = filter_size
		self.canny_thresh1 = canny_thresh1
		self.canny_thresh2 = canny_thresh2
		self.use_grayscale = use_grayscale

Preset.GEN_IMAGE = Preset(filter_size=17, canny_thresh1=30, canny_thresh2=50, use_grayscale=False)
Preset.REAL_PHOTO = Preset(filter_size=23, canny_thresh1=50, canny_thresh2=70, use_grayscale=True)


def readImage(image_filename):
	print("Reading {}".format(image_filename))
	img = cv2.imread(image_filename)
	return img

def calBoundingBox(img, use_preset= None):

	if use_preset is not None:
		filter_size = use_preset.filter_size
		canny_thresh1 = use_preset.canny_thresh1
		canny_thresh2 = use_preset.canny_thresh2
		use_grayscale = use_preset.use_grayscale
	# use opencv 
	img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if use_grayscale else img
	img3 = cv2.GaussianBlur(img2, (filter_size, filter_size), sigmaX=0)
	imgc = cv2.Canny(img3, canny_thresh1, canny_thresh2)
	xmin, ymin, w, h = cv2.boundingRect(imgc)

	return xmin, ymin, w, h

def calCuttingBox(xmin, ymin, w, h, target_width, target_height, img_w, img_h):
	new_xmin = w
	new_ymin = h
	# if the boxs width is narrower than target
	if w < target_width:
		new_xmin = xmin + (w/2) - (target_width/2)
		# if the new xmin is outside the original image
		if new_xmin < 0 :
			new_xmin = 0
		if new_xmin+target_width > img_w:
			new_xmin = img_w-target_width-1
	else:
		# have't think of what to do
		a = None
	if h < target_height:
		new_ymin = ymin + (h/2) - (target_height/2)
		# if the new ymin is outside the original image
		if new_ymin < 0 :
			new_ymin = 0
		if new_ymin+target_height > img_h:
			new_ymin = img_h-target_height-1
	else:
		# haven't think of what to do
		a = None

	return int(new_xmin), int(new_ymin)

def crop_img(img, new_xmin,new_ymin,target_width,target_height):
	new_ymax = int(new_ymin+target_height)
	print("new_ymax",new_ymax)
	new_xmax = int(new_xmin+target_width)
	print("new_xmax",new_xmax)
	return img[new_ymin:new_ymax, new_xmin:new_xmax]

def main():
	# get args for annotation csv file, image directory, and # of lines to read
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--img_dir',metavar='img_dir',required=True, type=str)
	parser.add_argument('-s', '--save_dir',metavar='save_dir',required=True, type=str)
	parser.add_argument('-j', '--target_height',metavar='target_height',required=True,type=int, default=400)
	parser.add_argument('-w', '--target_width',metavar='target_width',required=True,type=int, default=400)
	parser.add_argument('--use_real_image',action='store_true')
	parser.add_argument('--show_result',action='store_true')
	
	args = parser.parse_args()
	image_dir = args.img_dir
	target_height = args.target_height 
	target_width = args.target_width
	preset_set = Preset.GEN_IMAGE if not args.use_real_image else Preset.REAL_PHOTO
	show_result = args.show_result
	print("Target to chop the image into {} x {}".format(target_width,target_height))
	

	# create a new directory to store the cropped image
	save_dir = args.save_dir
	if not os.path.exists(save_dir):
		os.mkdir(save_dir)

	# Loop through all the image in the directory
	image_list = os.listdir(image_dir)
	for an_image in image_list:
		if an_image.endswith(".png") or an_image.endswith(".jpg") or an_image.endswith(".jpeg"):
			img = readImage(image_dir+an_image)
			# cv2.imshow('originalimage',img)
			# cv2.waitKey(0)
			xmin, ymin, w, h = calBoundingBox(img, preset_set)
			print("xmin, ymin, w, h: ",xmin, ymin, w, h)
			img_h, img_w = img.shape[:2]
			print("img_h, img_w: ",img_h, img_w)
			new_xmin, new_ymin = calCuttingBox(xmin, ymin, w, h,target_width,target_height, img_w, img_h)
			print("new_xmin, new_ymin: ",new_xmin, new_ymin)
			cropped = crop_img(img, new_xmin,new_ymin,target_width,target_height)
			print(cropped.shape)
			
			if show_result:
				while True:
					cv2.imshow("cropped",cropped)
					key = cv2.waitKey(30)
					if key == 27:
						break
			# store the image
			try:
				print("saving image ",save_dir+'/'+an_image)
				cv2.imwrite(save_dir+'/'+an_image,cropped)
			except Exception as e:
				raise e
	print("Done")



	return

main()