import cv2
import numpy as np
import time

class Webcam():
    def __init__(self):
        pass

    def take_photo(self, save: bool):
        cam = cv2.VideoCapture(0)
        result, image = cam.read()
        if result:
            cam.release()
            self.image = image
            if save:
                cv2.imwrite("img.png", image)
            return self.image
        else:
            print("Error")

if __name__ == "__main__":
    time.sleep(3)
    cam = Webcam()
    image = cam.take_photo(False)
    left = image[:,:320,:]
    right = image[:,320:,:]
    cv2.imwrite("righ.png", right)
    cv2.imwrite("left.png", left)