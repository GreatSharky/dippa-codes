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
box = [90,300,2*64,2*64]
sign_names = ["ok", "next", "previous", "1", "2","3","4","5",""]
sign_index = 0
for i in range(len(sign_names)*200):
    ret, frame = cap.read()
    if 199 == i%200:
        ok = frame
        ok = ok[box[1]:box[1]+box[2],box[0]:box[0]+box[3],:]
        cv2.imwrite(f"{sign_names[sign_index]}.jpg", ok)
        sign_index += 1
    cv2.putText(frame,f'{sign_names[sign_index]} sign {200-i%200}', 
    bottomLeftCornerOfText, 
    font, 
    fontScale,
    fontColor,
    thickness,
    lineType)
    frame = red_square(frame, box[0],box[1],box[2],box[3],2)
    cv2.imshow("win", frame)
    if cv2.waitKey(1) & 0xFF == ord('q') or sign_names[sign_index] == "":
        break
    print(i)

cap.release()
cv2.destroyAllWindows()