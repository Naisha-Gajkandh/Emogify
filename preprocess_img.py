import cv2
import numpy as np
from imutils import contours

def euclidean_distance(a, b):
	return float(np.linalg.norm(np.array(a, dtype=np.float32) - np.array(b, dtype=np.float32)))

def highest_euclidean_distance(fixed_point, *other_points):
	"""Return largest euclidean distance from fixed_point to any of other_points.
	   All arguments are (x, y) tuples / numpy arrays.
	"""
	largest_distance = 0
	for point in other_points:
		dist = euclidean_distance(fixed_point, point)
		if dist > largest_distance:
			largest_distance = dist
	return int(largest_distance)

def centroid(shape, *point_indices):
	num = len(point_indices)
	x = sum(float(shape[i][0]) for i in point_indices)
	y = sum(float(shape[i][1]) for i in point_indices)
	return (int(x / num), int(y / num))

def get_points(face_part, shape):
	points = [shape[point] for point in face_part]
	return np.array(points)

def create_mask(shape, img):
	height, width, channels = img.shape
	mask = np.zeros((height, width), dtype=np.uint8)
	right_eye = (36, 37, 38, 39, 40, 41)
	left_eye  = (42, 43, 44, 45, 46, 47)
	mouth     = (48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59)
	nose      = (27, 31, 33, 35)
	left_eyebrow  = (17, 18, 19, 20, 21)
	right_eyebrow = (22, 24, 25, 26)

	middle_right_eye = centroid(shape, *right_eye)
	middle_left_eye  = centroid(shape, *left_eye)

	radius_right_eye = int(highest_euclidean_distance(
		middle_right_eye, *[shape[i] for i in right_eye]))
	radius_left_eye  = int(highest_euclidean_distance(
		middle_left_eye,  *[shape[i] for i in left_eye]))

	mask = cv2.circle(mask, (int(middle_right_eye[0]), int(middle_right_eye[1])), max(radius_right_eye, 1), 255, -1)
	mask = cv2.circle(mask, (int(middle_left_eye[0]), int(middle_left_eye[1])),  max(radius_left_eye, 1),  255, -1)

	middle_mouth  = centroid(shape, *mouth)
	radius_mouth  = int(highest_euclidean_distance(
		middle_mouth, *[shape[i] for i in mouth]))
	mask = cv2.circle(mask, (int(middle_mouth[0]), int(middle_mouth[1])), max(radius_mouth, 1), 255, -1)

	mask = cv2.fillPoly(mask, [get_points(left_eyebrow, shape)],  255)
	mask = cv2.fillPoly(mask, [get_points(right_eyebrow, shape)], 255)
	mask = cv2.fillPoly(mask, [get_points(nose, shape)],          255)
	return mask

def get_bounding_rect(img):
	"""Return (x0, y0, x1, y1) bounding box that contains all contours in img."""
	cnts = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = cnts[0] if len(cnts) == 2 else cnts[1]

	if len(cnts) == 0:
		# No contours found — return full image bounds
		h, w = img.shape[:2]
		return 0, 0, w, h

	boundingBoxes = contours.sort_contours(cnts, method='left-to-right')[1]

	# Fix: initialize x0,y0,x1,y1 from the very first bounding box
	x0, y0, w0, h0 = boundingBoxes[0]
	x1 = x0 + w0
	y1 = y0 + h0

	for (x, y, w, h) in boundingBoxes[1:]:
		if x < x0:
			x0 = x
		if y < y0:
			y0 = y
		if x + w > x1:
			x1 = x + w
		if y + h > y1:
			y1 = y + h

	return x0, y0, x1, y1