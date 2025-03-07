from ultralytics import YOLO

model = YOLO("yolo11s.pt")

results = model.track(source="http://192.168.1.157:4747/video", show=True)