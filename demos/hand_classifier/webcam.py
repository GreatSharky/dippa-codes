"""This is the webcam part"""
import cv2
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class Webcam_filemonitor(FileSystemEventHandler):
    def __init__(self, cam):
        super().__init__()
        self.webcam = Webcam_logic(cam)

    def on_created(self, event):
        if not event.is_directory:
            file_path = event.src_path
            file = file_path[len("tmp/"):]
            if "_mask.jpg" in file


class Webcam_logic():
    def __init__(self, cam: str):
        self.__cap = cv2.VideoCapture(cam)
        self.__gestures = ["ok", "next", "previous", "1", "2","3","4","5"]
        self.__bb = [90,300,128,128]
        os.makedirs("tmp", exist_ok=True)
        self.cap_time = False
        self.mask_ready = False
        self.class_ready = False

    def start(self):
        for i in range(20000):
            ret, self.frame = self.__cap.read()
            # Do capture, add box, add latest mask
            if self.cap_time:
                capture = self.capture()
                file = f"{i}"
                cv2.imwrite(f"tmp/{file}.jpg", capture)
            if self.mask_ready:
                mask_file = f"tmp/{file}_mask.jpg"
                mask_img = cv2.imread(mask_file)
                self.__add_image(1,1, mask_img)

            self.frame = self.__add_red_rectangle(self.frame)
            self.frame = cv2.flip(self.frame, 1) # Flip for more inuitivness
            # Add text
            text = ""
            if self.class_ready:
                class_file = f"{file}.tmp"
                classification = self.read_classification(class_file)
                text = f"Classified as {classification}"
                self.frame = self.__add_text(self.frame, text, corner=(200,2))
            cv2.imshow("win", self.frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def __add_image(self, x, y, img):
        self.frame[y:y+img.shape[1], x:x+img.shape[0],:] = img

    def capture(self):
        box = self.__bb
        return self.frame[box[1]:box[1]+box[2], box[0]:box[0]+box[3],:]
    
    def read_classification(self, file):
        with open(f"tmp/{file}", "r") as file:
            classification = file.read()
        return classification
    
    def __add_red_rectangle(self, l=1):
        try:
            self.frame[self.__bb[1]:self.__bb[1]+self.__bb[3],self.__bb[0]:self.__bb[0]+l,:] = [0,0,255]
            self.frame[self.__bb[1]:self.__bb[1]+l,self.__bb[0]:self.__bb[0]+self.__bb[2],:] = [0,0,255]
            self.frame[self.__bb[1]:self.__bb[1]+self.__bb[3]+l,self.__bb[0]+self.__bb[2]:self.__bb[0]+self.__bb[2]+l,:] = [0,0,255]
            self.frame[self.__bb[1]+self.__bb[3]:self.__bb[1]+self.__bb[3]+l, self.__bb[0]:self.__bb[0]+self.__bb[2]+l,:] = [0,0,255]
        except IndexError as e:
            print(e)
            return self.frame
    
    def __add_text(self, frame, text, corner=(300,250)):
        font                   = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = corner
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
