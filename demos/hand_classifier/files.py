import os

while True:
    path = "tmp"
    files = [f for f in os.listdir(path)]
    print(files)