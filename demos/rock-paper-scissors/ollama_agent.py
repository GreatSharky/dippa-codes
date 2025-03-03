import requests
import base64
import cv2
import json
import ollama

for i in ["r", "p", "s"]:
    jpeg = cv2.imread(i+"1.jpg")
    print(i)
    res, jpeg =cv2.imencode(".jpg", jpeg)
    jpg_text = base64.b64encode(jpeg).decode("utf-8")
    response = ollama.chat(model="llama3.2-vision", messages=[
        {
            "role" : "system",
            "content" : """{
                "game" : "rock-paper-scissors",
                "role" : "referee",
                "description" : "return response object in json format",
                "choice" : {
                    "fist" : "rock",
                    "victory or peace sign" : "scissors",
                    "open palm or open hand" : "paper"
                },
                "response" : {
                    "choice" : "choice made in input image",
                },
                "response_format" : "json"
            },""",
            "format" : "json"
        },
        {
            "role" : "user",
            "content" : "Input image. Respond in JSON format.",
            "format" : "json",
            "images" : [jpg_text]
        }
    ])
    print(response.message.content)
