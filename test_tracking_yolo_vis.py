
import os
from pathlib import Path
import time
import pandas as pd
import torch
import cv2 
import os.path as osp
import sys

from bytetrack.timer import Timer 
from constants import BYTETRACK_TRACK_IMAGES_FOLDER, OUTPUT_FOLDER
from modules.images_to_video import images_to_video 
sys.path.append("bytetrack/tracker")

from bytetrack.run_tracker import run_tracker_on_frame
import argparse

from bytetrack.tracker.byte_tracker import BYTETracker



def image_demo(files:list, predictor, vis_folder, current_time, args):

    tracker = BYTETracker(args)

    files.sort()

    timer = Timer()
    results = []

    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    save_folder = osp.join(vis_folder, timestamp)

    for frame_id, img_path in enumerate(files, 1):

        image = cv2.imread(str(img_path))

        # Detect objects
        outputs = predictor(image)

        img_info = {"id": 0}
        height, width = image.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = image
        
        # scale = min(exp.test_size[0] / float(img_info['height'], ), exp.test_size[1] / float(img_info['width']))
        scale = 1

        detections = []
        if outputs[0] is not None:
            outputs = outputs[0]
            # outputs = outputs.cpu().numpy()
            # detections = outputs[:, :7]
            # detections[:, :4] /= scale
            detections = outputs

            _results, timer, online_im = run_tracker_on_frame(frame_id=frame_id, 
                                                              tracker=tracker,
                                                              detections=detections, 
                                                              aspect_ratio_thresh=args.aspect_ratio_thresh,
                                                              min_box_area=args.min_box_area,                                                              
                                                              height=img_info['height'], width=img_info['width'], 
                                                              raw_img=img_info['raw_img'], timer=timer
                                                              )
            results.extend(_results)

        # result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
        if args.save_result:
            os.makedirs(save_folder, exist_ok=True)
            cv2.imwrite(osp.join(save_folder, osp.basename(img_path)), online_im)

        if frame_id % 20 == 0:
            print('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

        ch = cv2.waitKey(0)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break

    if args.save_result:
        res_file = osp.join(vis_folder, f"{timestamp}.txt")
        with open(res_file, 'w') as f:
            f.writelines(results)
        print(f"save results to {res_file}")
    
    return save_folder


def make_parser():
    """
    parser from bytetrack_repo/tools/demo_track.py
    """
    parser = argparse.ArgumentParser("ByteTrack Demo!")
    parser.add_argument("--model", default='yolov5', choices=["yolov5", "yolov7"], type=str, help="model to test out")
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )
    parser.add_argument("--fps", default=30, type=int, help="frame rate (fps)")
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking")
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=1.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    return parser

def main():

    IMG_DIR = "data/KITTI/object/2011_09_26_drive_0106_sync/image_2"
    IMG_DIR = Path(IMG_DIR)

    args = make_parser().parse_args()


    if args.model =="yolov5":
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=False) # or yolov5m, yolov5l, yolov5x, custom

    elif args.model =="yolov8":
        # model = torch.hub.load('ultralytics/yolov8', 'yolov8n', force_reload=False)
        raise NotImplementedError("yolov8 model trough torch hub not working")

    elif args.model =="yolov7":
        # wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6.pt
        model = torch.hub.load('WongKinYiu/yolov7', 'custom', 'yolov7-e6.pt', force_reload=False, trust_repo=True)

    else:
        raise ValueError(f"unhandled model. found model = {args.model}")

    out_file_name = f"cam-output-{args.model}.avi"


    current_time = time.localtime()

    predictor = lambda x: model(x).xyxy

    vis_folder = str(BYTETRACK_TRACK_IMAGES_FOLDER)

    files = list(IMG_DIR.iterdir())

    save_folder = image_demo(files=files, predictor=predictor, vis_folder=vis_folder, current_time=current_time, args=args)

    save_img_video_folder = OUTPUT_FOLDER / "yolo-track"
    save_img_video_folder.mkdir(exist_ok=True)

    save_img_video_path = save_img_video_folder / out_file_name

    images_to_video(
        sorted(list(Path(save_folder).iterdir())),
        save_img_video_path,
    )

    print(f"saved video to {save_img_video_path} ")

if __name__ =="__main__":
    main()
