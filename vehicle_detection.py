import sys
from ultralytics import YOLO

source = sys.argv[1] if len(sys.argv) > 1 else "traffic.mp4"
YOLO("yolo11n.pt").track(source=source, show=True, save=True)
