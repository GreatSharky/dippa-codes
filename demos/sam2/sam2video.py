from ultralytics import YOLO, SAM
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image

yolo_model = YOLO("yolo11s.pt")

results = yolo_model.predict("media/videos/box.mp4")
index = 40
tens = results[index].boxes.cpu()
box = tens.xyxy[0]

sam_model = SAM("sam2.1_s.pt")
samresults = sam_model(results[index].orig_img, bboxes=box)
print(samresults[0].masks.data.shape)
mask = samresults[0].masks.cpu().data
plt.imshow(results[index].orig_img)
h, w = mask.shape[-2:]
color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
color = color.reshape(1,1,-1)
mask_image =  mask.reshape(h, w, 1) * color
plt.imshow(mask_image)
plt.show()

