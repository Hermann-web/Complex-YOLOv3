
from pathlib import Path
import pandas as pd
import torch
import cv2 

import sys 
sys.path.append("bytetrack/tracker")

import argparse

from bytetrack.tracker.byte_tracker import BYTETracker

def make_parser():
    """
    parser from bytetrack_repo/tools/demo_track.py
    """
    parser = argparse.ArgumentParser("ByteTrack Demo!")
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    return parser


IMG_DIR = "data/KITTI/object/2011_09_26_drive_0106_sync/image_2"
IMG_DIR = Path(IMG_DIR)

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=False)

# MODEL_PATH = Path("checkpoints/tiny-yolov3_ckpt_epoch-220.pth")
# use_cuda = True
# device = "cuda" if torch.cuda.is_available() and use_cuda else "cpu"
# checkpoint = torch.load(MODEL_PATH, map_location=device)
# model.load_state_dict(checkpoint['model_state_dict'])
# model.eval()

args = make_parser().parse_args()
tracker = BYTETracker(args)

for img_path in IMG_DIR.iterdir():
    image = cv2.imread(str(img_path))
    # with torch.no_grad():
    results = model(image)
    detections = results.xyxy
    assert len(detections)==1
    detections = detections[0]
    print(f"detections = {detections}")
    # print(f"detections: {detections.shape}")
    img_info = {"id": 0}
    height, width = image.shape[:2]
    img_info["height"] = height
    img_info["width"] = width
    img_info["raw_img"] = image
    online_targets = tracker.update(detections, [img_info['height'], img_info['width']], [img_info['height'], img_info['width']])

    # print(f"online_targets = {online_targets}")

    break
