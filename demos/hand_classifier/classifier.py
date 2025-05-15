"""This one will classify the segmented image"""
from gemma3_agent import VLM_gemma
import os
from segment import file_index
import socket


if __name__ == "__main__":
    descriptions = {
        "ok" : "it resembles the OK sign. The thumb and the index finger create an circle while the three other fingers are extended.",
        "1" : "it has only the index finger pointing upwards and all the other fingers tucked down. The palm is facing the camera.",
        "2" : "it resembles the V sign for victory or for peace. The index and middle finger are extended while the three other fingers are down. The palm is facing the camera.",           
        "3" : "it has the index finger, the middle finger and the ring finger extended while the thumb and the pinky are tucked down. The palm is facing the camera.",
        "4" : "four fingers besides the thumb are extended. The thumb is tucked down into the middle of the palm and the palm is facing the camera.",
        "5" : "all five fingers are extended and separated from each other. The palm is facing the camera."
    }
    vlm1 = VLM_gemma("gemma3:12b", ["ok","1","2","3","4", "5"], descriptions, 1)
    vlm2 = VLM_gemma("gemma3:12b", ["ok","1","2","3","4", "5"], descriptions, 1)
    path = "tmp"
    files = [f for f in os.listdir(path) if "_mask.jpg"in f]
    file = ""
    HOST="192.168.125.1"
    PORT = 5000
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST,PORT))
    print("TCP client initialized")
    while True:
        files = [f for f in os.listdir(path) if "_mask.jpg"in f]
        if files:
            latest_mask = max(files, key=file_index)
            if file != file_index(latest_mask):
                print(latest_mask)
                file = file_index(latest_mask)
                file_name = f"tmp/{latest_mask}"
                vlm1.create_user_msg(file_name)
                result1 = vlm1.inference()
                vlm2.create_user_msg(file_name)
                result2 = vlm2.inference()
                if result1.message.content == result2.message.content:
                    response = f"Class {result1.message.content}"
                else:
                    response = f"Unclear: res {result1.message.content} and {result2.message.content}"
                print(response)
                with open(f"tmp/{file}.tmp", "+w") as tmpfile:
                    tmpfile.write(response)
                s.sendall(response.encode())
