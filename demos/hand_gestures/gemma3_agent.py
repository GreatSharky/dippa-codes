import ollama
import base64
import cv2
import numpy as np
import time
from vlm_agent import VLM

class VLM_gemma(VLM):
    def __init__(self, model: str, gestures: list, descriptions, samples: int):
        super().__init__(model, gestures, samples)
        self.rng = np.random.default_rng(1)
        self.descriptions = descriptions
        self.system_msgs = self.__system_prompt()
    
    def create_user_msg(self,img_file):
        img = self.get_image(img_file)
        msg = {
            "role" : "user",
            "content" : "What is in this image?",
            "images" : [img]
        }
        self.user_msgs = [msg]

    def __create_system_msg(self, description, index, img_file):
        msg = {
            "role" : "system",
            "content" : f"This image belongs to class {index}. Its' charasteristics are {description}"
        }
        return msg
    
    def __system_prompt(self):
        start = {
            "role" : "system",
            "content" : "You are an state of the art hand gesture classifier. You are given images of different hand gestures wtih a short description of their characteristics for training and then you need to respond to the user requests with only what class the user image belongs to"
        }
        msgs = [start]
        img_id = self.rng.choice(5,self.samples,replace=False)
        for index, ges in enumerate(self.gestures):
            for i in range(self.samples):
                file_name = f"masks/{ges}_{img_id[i]}.jpg"
                img_b64 = self.get_image(file_name)
                msg = self.__create_system_msg(self.descriptions[ges], index, img_b64)
                msgs.append(msg)
        return msgs

descriptions = {
    "ok" : "it resembles the OK sign. The thumb and the index finger create an circle while the three other fingers are extended.",
    "1" : "it has only the index finger pointing upwards and all the other fingers tucked down. The palm is facing the camera.",
    "2" : "it resembles the V sign for victory or for peace. The index and middle finger are extended while the three other fingers are down. The palm is facing the camera.",           
    "3" : "it has the index finger, the middle finger and the ring finger extended while the thumb and the pinky are tucked down. The palm is facing the camera.",
    "4" : "four fingers besides the thumb are extended. The thumb is tucked down into the middle of the palm and the palm is facing the camera.",
    "5" : "all five fingers are extended and separated from each other. The palm is facing the camera."
}
vlm = VLM_gemma("gemma3:12b", ["ok","1","2","3","4", "5"], descriptions, 1)


id = "4"
print("Real", id)
vlm.create_user_msg(f"masks/{id}_5.jpg")
res = vlm.inference()
print("VLM",res.message.content)