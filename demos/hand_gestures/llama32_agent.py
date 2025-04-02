import ollama
import base64
import cv2
import numpy as np
import time
from vlm_agent import VLM

class VLM_llama(VLM):
    def __init__(self, model: str, gestures: list, samples: int):
        super().__init__(model, gestures, samples)
        self.rng = np.random.default_rng(1)
        self.system_msgs = self.__system_prompt()

    def __create_sys_msg(self, role, class_id, img):
        msg = {
            "role" : role,
            "content" : f"""You are state of the art hand gestrure classifier.
            This image is of class {class_id}. 
            Respond to the user only with which class the input image belongs.""",
            "images" : [img]
        }
        return msg
    
    def create_user_msg(self,img_file):
        img = self.get_image(img_file)
        msg = {
            "role" : "user",
            "content" : "What class is this image? Respond using only the class.",
            "images" : [img]
        }
        self.user_msgs = [msg]
    
    def __system_prompt(self):
        msgs = []
        img_id = self.rng.choice(8,self.samples,replace=False)
        print(img_id)
        for index, ges in enumerate(self.gestures):
            for i in range(self.samples):
                file_name = f"masks/{ges}_{img_id[i]}.jpg"
                img_b64 = self.get_image(file_name)
                msg = self.__create_sys_msg("system", index+1, img_b64)
                msgs.append(msg)
        return msgs

# Uses llama3.2-vision:10B
vlm = VLM_llama("llama3.2-vision", ["ok","1", "5"], 3)
txt = "What is the class of this img? Respond using only the classifier."
vlm.create_user_msg("masks/5_5.jpg")
res = vlm.inference()
print(res.message.content)