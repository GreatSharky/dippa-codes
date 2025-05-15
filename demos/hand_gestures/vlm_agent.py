import ollama
import base64
import cv2
import numpy as np

class VLM():
    def __init__(self, model: str, gestures: list, samples: int):
        self.gestures = gestures
        self.samples = samples
        self.model = model
        self.system_msgs = []
        self.user_msgs = []

    def inference(self):
        response = self.system_msgs+self.user_msgs
        response = ollama.chat(model=self.model, messages=response)
        return response

    def get_image(self, file):
        jpg = cv2.imread(f"{file}")
        _, jpg = cv2.imencode(".jpg", jpg)
        b64 = base64.b64encode(jpg).decode("utf-8")
        return b64