"""This is the segmentation"""
from ultralytics import SAM
import cv2

class Segmentor():
    def __init__(self, model="sam2.1_b.pt"):
        self.sam = SAM(model)

    def segment(self, image):
        masks = self.sam(image, points=[[64,80], [1,1], [127,127]], labels=[1,0,0])
        mask = masks[0].masks.cpu().data
        h,w = mask.shape[-2:]
        mask = mask.reshape(h,w,1).numpy()
        return image*mask