import cv2, os, random
import numpy as np

def get_image_size():
	# Dynamically find first image in dataset/0/
	folder = 'dataset/0/'
	for f in sorted(os.listdir(folder)):
		if f.endswith('.jpg'):
			img = cv2.imread(folder + f, 0)
			if img is not None:
				return img.shape
	return (100, 100)  # fallback

gestures = os.listdir('dataset/')
gestures.sort(key = int)
begin_index = 0
end_index = 5
image_x, image_y = get_image_size()

if len(gestures)%5 != 0:
	rows = int(len(gestures)/5)+1
else:
	rows = int(len(gestures)/5)

full_img = None
last_img_num = int(input('Enter last image number: '))
for i in range(rows):
	col_img = None
	for j in range(begin_index, end_index):
		img_path = "dataset/%s/%d.jpg" % (j, random.randint(1, last_img_num))
		img = cv2.imread(img_path, 0)
		if img is None:
			img = np.zeros((image_y, image_x), dtype = np.uint8)
		if col_img is None:
			col_img = img
		else:
			col_img = np.hstack((col_img, img))

	begin_index += 5
	end_index += 5
	if full_img is None:
		full_img = col_img
	else:
		full_img = np.vstack((full_img, col_img))


cv2.imshow("faces", full_img)
cv2.imwrite('full_img.jpg', full_img)
cv2.waitKey(0)