import requests
import base64
import cv2
import json
import ollama

jpeg = cv2.imread("s.jpg")
res, jpeg =cv2.imencode(".jpg", jpeg)
jpg_text = base64.b64encode(jpeg).decode("utf-8")
response = ollama.chat(model="llama3.2-vision", messages=[
    {
        "role" : "system",
        "content" : "You are rock-paper-scissors(RPS) judge. You need to answer what RPS object the hand in the image is representing. Respond using JSON",
        "format" : "json",
    },
    {
        "role" : "user",
        "conent" : "Which RPS choice is in the image? Answer only using JSON.",
        "format" : "json",
        "images" : [jpg_text]
    }
])
print(response.message.content)
