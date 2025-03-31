from ultralytics import SAM
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt

sam_model = SAM("sam2.1_b.pt")
file_names = ["ok1", "next", "previous", "1", "2","3","4","5"]
for name in file_names:
    img_ok = cv2.imread(f"{name}.jpg")
    sam_results = sam_model(img_ok,points=[64,90])
    mask = sam_results[0].masks.cpu().data
    h,w = mask.shape[-2:]
    mask = mask.reshape(h,w,1).numpy()
    img = img_ok*mask
    cv2.imwrite(f"{name}_mask.jpg", img)
    