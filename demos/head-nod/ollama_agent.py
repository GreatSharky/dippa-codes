import requests
import base64
import cv2
import json
import ollama

images = []
jpeg = cv2.imread("rc.jpg")
res, jpeg =cv2.imencode(".jpg", jpeg)
jpg_text = base64.b64encode(jpeg).decode("utf-8")
images.append(jpg_text)

response = ollama.chat(model="llava", messages=[
    {
        "role" : "user",
        "content" : "Is the head leaning to left or right in the image relative to the camera? Answer in JSON format.",
        "images" : images
    }
])
print(response.message.content)
