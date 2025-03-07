from ultralytics import YOLO
import matplotlib.pyplot as plt

model = YOLO("yolo11s.pt")

results = model.predict("media/videos/box.mp4")
print(results[0].boxes.xyxy)
