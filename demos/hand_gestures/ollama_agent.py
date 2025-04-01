import ollama
import base64
import cv2
import numpy as np
import time

class VLM():
    def __init__(self, model: str, gestures: list, samples: int):
        self.gestures = gestures
        self.samples = samples
        self.rng = np.random.default_rng(3)
        self.model = model
        self.system_msgs = self.__system_prompt()

    def inference(self, img):
        img_b64 = self.__get_image(img)
        msg = {
            "role" : "user",
            "content" : "What class is this image? Answer only using the class id.",
            "images" : [img_b64]
        }
        response = self.system_msgs+[msg]
        response = ollama.chat(model=self.model, messages=self.system_msgs+[msg])
        return response

    def __get_image(self, file):
        jpg = cv2.imread(f"masks/{file}")
        _, jpg = cv2.imencode(".jpg", jpg)
        b64 = base64.b64encode(jpg).decode("utf-8")
        return b64

    def __create_sys_msg(self, role, class_id, img):
        msg = {
            "role" : role,
            "content" : f"""You are state of the art hand gestrure classifier.
            This image is of class {class_id}. 
            Respond to the user only with which class the input image belongs.""",
            "images" : [img]
        }
        return msg
    
    def __system_prompt(self):
        msgs = []
        img_id = self.rng.choice(8,self.samples,replace=False)
        for i in range(self.samples):
            for index, ges in enumerate(self.gestures):
                file_name = f"{ges}_{img_id[i]}.jpg"
                print(index+1, file_name)
                img_b64 = self.__get_image(file_name)
                msg = self.__create_sys_msg("system", index+1, img_b64)
                msgs.append(msg)
        return msgs

vlm = VLM("llama3.2-vision", ["ok","1", "5"], 3)
res = vlm.inference(img="5_5.jpg")
print(res.message.content)