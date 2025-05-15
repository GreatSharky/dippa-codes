"""This is the segmentation"""
from ultralytics import SAM
import cv2
import os

def file_index(x):
    x_index = int(x[:x.find("_")])
    return x_index 

class Segmentor():
    def __init__(self, model="sam2.1_b.pt"):
        self.sam = SAM(model)

    def segment(self, image):
        masks = self.sam(image, points=[[64,80]], labels=[1])
        mask = masks[0].masks.cpu().data
        print(mask)
        h,w = mask.shape[-2:]
        mask = mask.reshape(h,w,1).numpy()
        return image*mask
    
if __name__ == "__main__":
    path = "tmp"
    files = [f for f in os.listdir(path) if "_cap.jpg" in f]
    file = ""
    sam = Segmentor()
    while True:
        files = [f for f in os.listdir(path) if "_cap.jpg" in f]
        if files:
            latest_cap = max(files, key=file_index)
            if file != file_index(latest_cap):
                print(latest_cap)
                file = file_index(latest_cap)
                img = cv2.imread(f"tmp/{latest_cap}")
                mask = sam.segment(img)
                cv2.imwrite(f"tmp/{file}_mask.jpg", mask)
