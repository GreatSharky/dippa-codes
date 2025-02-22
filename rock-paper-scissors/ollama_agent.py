import requests
import base64
import cv2
import json
import ollama

image = cv2.imread("image.png")
result, jpeg = cv2.imencode(".jpg", image)
jpg_text = base64.b64encode(jpeg).decode("utf-8")

response = ollama.chat(model="llama3.2-vision", messages=[
    {
        "role" : "system",
        "content" : "You are rock-paper-scissors(RPS) judge. You need to answer what RPS object the hand in the image is representing. Answer using json."
    },
    {
        "role" : "user",
        "conent" : "Which RPS choice is in the image. Asnswer using json.",
        "format" : "json",
        "images" : [jpg_text]
    }
])
print(response.message.content)
