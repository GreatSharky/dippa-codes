from ultralytics import SAM
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt

sam_model = SAM("sam2.1_s.pt")
img_ok = cv2.imread("ok.jpg")
cv2.imshow("original",img_ok)
print(img_ok.shape)
sam_results = sam_model(img_ok,points=[64,80])
mask = sam_results[0].masks.cpu().data
h,w = mask.shape[-2:]
mask = mask.reshape(h,w,1).numpy()
img = img_ok*mask
print(img_ok[64,64,:])
print(img[64,64,:])

cv2.imshow("mask", img)
plt.imshow(img)
plt.show()