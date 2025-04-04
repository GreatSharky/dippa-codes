"""This one is for file monitoring"""
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import time
        

class File_monitor():
    def __init__(self, folder):
        class Capture_Handler(FileSystemEventHandler):
            def on_created(self, event):
                if not event.is_directory:
                    e =event.src_path
                    print(e)
                    print(e[len("tmp/"):])
        event_handler = Capture_Handler()
        self.folder = folder
        self.observer = Observer()
        self.observer.schedule(event_handler, folder, recursive=False)

    def start(self):
        self.observer.start()

file_monitor = File_monitor("tmp")
file_monitor.start()
try:
    while True:
        time.sleep(1)
except:
    file_monitor.observer.stop()
file_monitor.observer.join()