"""This is the webcam part"""
import cv2
import os
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class Webcam():
    def __init__(self, cam: str):
        self.__cap = cv2.VideoCapture(cam)
        self.__bb = [90,200,128,128]
        self.__path = "tmp"
        os.makedirs(self.__path, exist_ok=True)
        self.clear_tmp_folder()
        self.cap_time = True
        self.mask_ready = False
        self.class_ready = False

    def start(self):
        file = "0"
        text = ""
        for i in range(20000):
            self.check_programstate(file)
            ret, self.frame = self.__cap.read()
            # Do capture, add box, add latest mask
            if self.cap_time and 99 == i %100:
                capture = self.capture()
                file = f"{i}"
                cv2.imwrite(f"{self.__path}/{file}_cap.jpg", capture)
                self.cap_time = False
                print("Capture made", f"{self.__path}/{file}_cap.jpg")
            if self.mask_ready:
                mask_file = f"{self.__path}/{file}_mask.jpg"
                mask_img = cv2.imread(mask_file)
                if type(mask_img) != type(None):
                    self.__add_image(1,1, mask_img)
                else:
                    print(file)

            self.frame = self.__add_red_rectangle()
            self.frame = cv2.flip(self.frame, 1) # Flip for more inuitivness
            # Add text
            if self.class_ready:
                class_file = f"{file}.tmp"
                classification = self.read_classification(class_file)
                text = f"Classified as {classification}"
                self.cap_time = True
                file = ""
            self.frame = self.__add_text(self.frame, text)
            cv2.imshow("win", self.frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def check_programstate(self, index):
        path = self.__path
        files = [f for f in os.listdir(path)]
        self.mask_ready = f"{index}_mask.jpg" in files
        self.class_ready = f"{index}.tmp" in files

    def __add_image(self, x, y, img):
        self.frame[y:y+img.shape[1], x:x+img.shape[0],:] = img

    def capture(self):
        box = self.__bb
        return self.frame[box[1]:box[1]+box[2], box[0]:box[0]+box[3],:]
    
    def read_classification(self, file):
        with open(f"{self.__path}/{file}", "r") as file:
            classification = file.read()
        return classification
    
    def __add_red_rectangle(self, l=1):
        x = self.__bb[0]
        y = self.__bb[1]
        w = self.__bb[2]
        h = self.__bb[3]
        self.frame[y:y+h, x:x+l, :] = [0,0,255]
        self.frame[y:y+l, x:x+w, :] = [0,0,255]
        self.frame[y:y+h+l, x+w:x+w+l, :] = [0,0,255]
        self.frame[y+h:y+h+l, x:x+w+l, :] = [0,0,255]
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
    
    def clear_tmp_folder(self):
        for file in os.listdir(self.__path):
            os.unlink(f"{self.__path}/{file}")


if __name__ == "__main__":
    cam_ip = os.getenv("CAM_IP")
    cam = Webcam(cam_ip)
    cam.start()
