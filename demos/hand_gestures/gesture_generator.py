import cv2
import numpy as np
import os
from ultralytics import SAM
import time

cam_ip = os.environ.get("CAM_IP")

class GestureGenrator():
    def __init__(self, cam: str, model: str):
        self.__cap = cv2.VideoCapture(cam)
        self.__gestures = ["ok", "next", "previous", "1", "2","3","4","5"]
        self.__bb = [90,300,128,128]
        self.__sam = SAM(model)

    def start(self):
        gesture_counter = 0
        gesture_index = 0
        for i in range(20000):
            _, frame = self.__cap.read()
            if 99 == i%100:
                gesture = self.__get_bb_frame(frame)
                masked_image = self.__masked_image(gesture)
                file_name = f"{self.__gestures[gesture_index]}_{gesture_counter}.jpg"
                cv2.imwrite(f"masks2/{file_name}", masked_image)
                gesture_index += 1
            if gesture_index == len(self.__gestures):
                gesture_index = 0
                gesture_counter += 1
            if gesture_counter == len(self.__gestures):
                break
            frame = self.__add_red_rectangle(frame)
            frame = cv2.flip(frame, 1)
            text = f'{self.__gestures[gesture_index]} sign in {100-i%100}: {gesture_counter}'
            frame = self.__add_text(frame, text)
            cv2.imshow("win", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


    def __get_bb_frame(self, frame):
        box = self.__bb
        return frame[box[1]:box[1]+box[2], box[0]:box[0]+box[3],:]
    
    def __add_red_rectangle(self, frame, l=1):
        try:
                frame[self.__bb[1]:self.__bb[1]+self.__bb[3],self.__bb[0]:self.__bb[0]+l,:] = [0,0,255]
                frame[self.__bb[1]:self.__bb[1]+l,self.__bb[0]:self.__bb[0]+self.__bb[2],:] = [0,0,255]
                frame[self.__bb[1]:self.__bb[1]+self.__bb[3]+l,self.__bb[0]+self.__bb[2]:self.__bb[0]+self.__bb[2]+l,:] = [0,0,255]
                frame[self.__bb[1]+self.__bb[3]:self.__bb[1]+self.__bb[3]+l, self.__bb[0]:self.__bb[0]+self.__bb[2]+l,:] = [0,0,255]
                return frame
        except IndexError as e:
            print(e)
            return frame
    
    def __masked_image(self, capture):
        masks = self.__sam(capture, points=[[64,80], [3,3], [123,123]], labels=[1,0,0])
        mask = masks[0].masks.cpu().data
        h,w = mask.shape[-2:]
        mask = mask.reshape(h,w,1).numpy()
        return capture*mask
    
    def __add_text(self, frame, text):
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (300,250)
        fontScale              = 1
        fontColor              = (0,0,0)
        thickness              = 2
        lineType               = 1
        cv2.putText(frame, text, 
            bottomLeftCornerOfText, 
            font, 
            fontScale,
            fontColor,
            thickness,
            lineType)
        return frame
    

gg = GestureGenrator(cam_ip, "sam2.1_b.pt")
gg.start()
