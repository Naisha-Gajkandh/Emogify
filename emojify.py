import cv2
import numpy as np
import dlib, os
from imutils import face_utils
from imutils.face_utils import FaceAligner
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
import tf_keras
from tf_keras.models import load_model
from preprocess_img import create_mask, get_bounding_rect
from blend import blend
from emoji_generator import draw_emoji
import time

CNN_MODEL = 'cnn_model_keras.h5'
SHAPE_PREDICTOR_68 = "shape_predictor_68_face_landmarks.dat"

cnn_model = load_model(CNN_MODEL)
shape_predictor_68 = dlib.shape_predictor(SHAPE_PREDICTOR_68)
detector = dlib.get_frontal_face_detector()

# Ask user for gender FIRST so the camera doesn't timeout waiting for input
gender_input = input("Enter your gender (male/female): ").strip().lower()
if gender_input not in ['male', 'female']:
    gender_input = 'neutral'

print("Detecting available cameras...")
cam = None

# First search for any DSHOW camera that CAN actually read frames
print("Scanning with DirectShow backend (recommended for Windows)...")
for i in range(5):
    temp_cam = cv2.VideoCapture(i, cv2.CAP_DSHOW) 
    if temp_cam.isOpened():
        ret, frame = temp_cam.read()
        if ret and frame is not None:
            print(f"Successfully connected to Camera {i} (DSHOW)!")
            cam = temp_cam
            break
    temp_cam.release()

if cam is None:
    # Fallback to direct default (MSMF) if DSHOW fails completely
    print("DSHOW backend failed, trying default MSMF backend...")
    for i in range(5):
        temp_cam = cv2.VideoCapture(i)
        if temp_cam.isOpened():
            ret, frame = temp_cam.read()
            if ret and frame is not None:
                print(f"Successfully connected to Camera {i} (Default)!")
                cam = temp_cam
                break
        temp_cam.release()

if cam is None:
    print("FATAL ERROR: Could not open any webcam. Please ensure your camera is plugged in and not used by another app.")
    exit(1)

print("Camera initialized successfully! Starting feed...")
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
fa = FaceAligner(shape_predictor_68, desiredFaceWidth=250)

# Removed get_emojis

def get_image_size():
	# Dynamically find first image in dataset/0/
	folder = 'dataset/0/'
	for f in os.listdir(folder):
		if f.endswith('.jpg'):
			img = cv2.imread(folder + f, 0)
			if img is not None:
				return img.shape
	return (100, 100)  # fallback

image_x, image_y = get_image_size()

def keras_process_image(img):
	img = cv2.resize(img, (image_x, image_y))
	img = np.array(img, dtype=np.float32)
	img = np.reshape(img, (1, image_x, image_y, 1))
	return img

def keras_predict(model, image):
	processed = keras_process_image(image)
	pred = model.predict(processed, verbose=0)
	pred_probab = pred[0]
	pred_class = list(pred_probab).index(max(pred_probab))
	return max(pred_probab), pred_class

def fun_util(gender):
	while True:
		ret, img = cam.read()
		if not ret:
			print("Failed to grab frame")
			break
		img = cv2.flip(img, 1)
		gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		faces = detector(gray)
		if len(faces) > 0:
			for i, face in enumerate(faces):
				try:
					shape_68 = shape_predictor_68(img, face)
					shape = face_utils.shape_to_np(shape_68)
					mask = create_mask(shape, img)
					masked = cv2.bitwise_and(gray, mask)
					maskAligned = fa.align(mask, gray, face)
					faceAligned = fa.align(masked, gray, face)
					if maskAligned is None or faceAligned is None:
						continue
					(x0, y0, x1, y1) = get_bounding_rect(maskAligned)
					if x1 == 0 or y1 == 0:
						continue
					faceAligned = faceAligned[y0:y1, x0:x1]
					faceAligned = cv2.resize(faceAligned, (100, 100))
					(x, y, w, h) = face_utils.rect_to_bb(face)
					cv2.imshow('faceAligned', faceAligned)
					cv2.imshow('face #{}'.format(i), img[y:y+h, x:x+w])
					pred_probab, pred_class = keras_predict(cnn_model, faceAligned)
					
					# Dynamically generate emoji based on class & gender
					emoji_img = draw_emoji(gender, pred_class)
					img = blend(img, emoji_img, (x, y, w, h))
				except Exception as e:
					print(f"Face processing error: {e}")
					continue
		cv2.imshow('img', img)
		if cv2.waitKey(1) == ord('q'):
			break

	cam.release()
	cv2.destroyAllWindows()

# Warm-up prediction to avoid first-frame lag
keras_predict(cnn_model, np.zeros((image_x, image_y), dtype=np.uint8))
fun_util(gender_input)