import os
from PIL import Image
import numpy as np
import cv2
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "dataset")
face_cascade = cv2.CascadeClassifier("cascades/data/haarcascade_frontalface_alt2.xml")


current_id = 1
label_ids = {}
y_labels = []
x_train = []
for root, dirs, files in os.walk(image_dir):
	for file in files:
	 if file.endswith("png") or file.endswith("jpg"):
           path = os.path.join(root, file)
           label=os.path.basename(os.path.dirname(path)).replace(" ", "-").lower()
           if not label in label_ids:
               label_ids[label] = current_id
               current_id += 1
           id_ = label_ids[label]
           print(label_ids)
           #x_train.append(path)
           pil_image=Image.open(path).convert("L")
           image_array = np.array(pil_image, "uint8")
           print(image_array)
           faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)
           for (x, y, w, h) in faces:
               roi = image_array[y:y + h, x:x + w]
               x_train.append(roi)
               y_labels.append(id_)





