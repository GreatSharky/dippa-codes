import cv2
import numpy as np
import os
import time

def red_square(frame, x, y, w, h,l=1):
    c1 = (x,y)
    frame[y:y+h,x:x+l,:] = [0,0,255]
    frame[y:y+l,x:x+w,:] = [0,0,255]
    frame[y:y+h+l,x+w:x+w+l,:] = [0,0,255]
    frame[y+h:y+h+l, x:x+w+l,:] = [0,0,255]
    return frame


font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (300,250)
fontScale              = 1
fontColor              = (0,0,0)
thickness              = 2
lineType               = 1

cam_ip = os.environ.get("CAM_IP")

cap = cv2.VideoCapture(cam_ip)
box = [90,200,200,200]
for i in range(4000):
    ret, frame = cap.read()
    if i == 200:
        ok = frame
        ok = ok[box[1]:box[1]+box[2],box[0]:box[0]+box[3],:]
        cv2.imshow(str(ok.shape), ok)
        cv2.imwrite("ok.jpg", ok)
    cv2.putText(frame,f'OK sign {200-i}', 
    bottomLeftCornerOfText, 
    font, 
    fontScale,
    fontColor,
    thickness,
    lineType)
    frame = red_square(frame, box[0],box[1],box[2],box[3],2)
    cv2.imshow("win", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    print(i)

cap.release()
cv2.destroyAllWindows()