from pathlib import Path
from typing import List
import numpy as np
import math
import os
import argparse
import cv2
import time
import torch

from bytetrack.timer import Timer

from constants import OUTPUT_FOLDER
from modules.images_to_video import images_to_video
import utils.utils as utils
from models import *
import torch.utils.data as torch_data

import utils.kitti_utils as kitti_utils
import utils.kitti_aug_utils as aug_utils
import utils.kitti_bev_utils as bev_utils
import utils.kitti_prediction_utils as pred_utils
import utils.kitty_tracking_utils as tracking_utils
from utils.kitti_yolo_dataset import KittiYOLODataset
import utils.config as cnf
import utils.mayavi_viewer as mview

import sys
sys.path.append("bytetrack/tracker")
from bytetrack.tracker.byte_tracker import BYTETracker
from bytetrack.run_tracker import run_tracker_on_frame

OUTPUT_FOLDER_COMPLEX_YOLO_TRACK = OUTPUT_FOLDER / "complex-yolo-track"
OUTPUT_FOLDER_COMPLEX_YOLO_TRACK.mkdir(exist_ok=True)

# where to save images post detection
OUTPUT_FOLDER_COMPLEX_YOLO_BEV = OUTPUT_FOLDER_COMPLEX_YOLO_TRACK/"bev-images"
OUTPUT_FOLDER_COMPLEX_YOLO_BEV.mkdir(exist_ok=True)
OUTPUT_FOLDER_COMPLEX_YOLO_CAM = OUTPUT_FOLDER_COMPLEX_YOLO_TRACK/"cam-images"
OUTPUT_FOLDER_COMPLEX_YOLO_CAM.mkdir(exist_ok=True)




TEST_TRACKING = True
TEST_DETECTION = True
TEST_TRACKING_FROM_IMG = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_def", type=str, default="config/complex_tiny_yolov3.cfg", help="path to model definition file")
    parser.add_argument("--weights_path", type=str, default="checkpoints/tiny-yolov3_ckpt_epoch-220.pth", help="path to weights file")
    parser.add_argument("--class_path", type=str, default="data/classes.names", help="path to class label file")
    parser.add_argument("--conf_thres", type=float, default=0.5, help="object confidence threshold")
    parser.add_argument("--nms_thres", type=float, default=0.5, help="iou thresshold for non-maximum suppression")
    parser.add_argument("--img_size", type=int, default=cnf.BEV_WIDTH, help="size of each image dimension")
    parser.add_argument("--split", type=str, default="valid", help="text file having image lists in dataset")
    parser.add_argument("--folder", type=str, default="training", help="directory name that you downloaded all dataset")
    
    # for bytetrack
    parser.add_argument(
        "--save_result",
        action="store_true",
        help="whether to save the inference result of image/video",
    )
    parser.add_argument("--fps", default=30, type=int, help="frame rate (fps)")
    parser.add_argument("--track_thresh", type=float, default=0.5, help="tracking confidence threshold: score")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.8, help="matching threshold for tracking: iou")
    parser.add_argument(
        "--aspect_ratio_thresh", type=float, default=0.6,
        help="threshold for filtering out boxes of which aspect ratio are above the given value."
    )
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--mot20", dest="mot20", default=False, action="store_true", help="test mot20.")
    
    # parse the user args
    opt = parser.parse_args()
    print(opt)

    if TEST_TRACKING:
        # some initialisation for tracking
        tracker = BYTETracker(opt)
        results = []
        start_time = time.time()
        timer = Timer()
        vis_folder = OUTPUT_FOLDER_COMPLEX_YOLO_TRACK
        timestamp = time.strftime("%Y_%m_%d_%H_%M_%S-bev", time.localtime())
        save_folder = vis_folder / timestamp
        save_folder.mkdir(exist_ok=True)
        res_file = str(vis_folder/f"{timestamp}.txt")

    if TEST_TRACKING_FROM_IMG:
        # some initialisation for tracking
        tracker_img = BYTETracker(opt)
        cam_results = []
        start_time2 = time.time()
        timer2 = Timer()
        vis_folder = OUTPUT_FOLDER_COMPLEX_YOLO_TRACK
        timestamp = time.strftime("%Y_%m_%d_%H_%M_%S-img", time.localtime())
        save_folder2 = vis_folder / timestamp
        save_folder2.mkdir(exist_ok=True)
        res_file2 = str(vis_folder/f"{timestamp}.txt")
        

    # some initialisation for model
    classes = utils.load_classes(opt.class_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)
    # Load checkpoint weights
    model.load_state_dict(torch.load(opt.weights_path, map_location=device))
    # Eval mode
    model.eval()
    
    # instantiate the dataloader
    dataset = KittiYOLODataset(cnf.root_dir, split=opt.split, mode='TEST', folder=opt.folder, data_aug=False)
    data_loader = torch_data.DataLoader(dataset, 1, shuffle=False)

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

                          
    for index, (img_paths, bev_maps) in enumerate(data_loader):

        print(f"img_paths = {img_paths}")
        
        # Configure bev image
        input_imgs = Variable(bev_maps.type(Tensor))

        # Get detections 
        with torch.no_grad():
            detections_base = model(input_imgs)
            detections_base: List[torch.Tensor] = utils.non_max_suppression_rotated_bbox(detections_base, opt.conf_thres, opt.nms_thres) 
        img_detections = []  # Stores detections for each image index
        img_detections.extend(detections_base)


        # get BEV_map from bev img data
        bev_maps:np.ndarray = torch.squeeze(bev_maps).numpy()
        RGB_Map = np.zeros((cnf.BEV_WIDTH, cnf.BEV_WIDTH, 3))
        RGB_Map[:, :, 2] = bev_maps[0, :, :]  # r_map
        RGB_Map[:, :, 1] = bev_maps[1, :, :]  # g_map
        RGB_Map[:, :, 0] = bev_maps[2, :, :]  # b_map
        RGB_Map *= 255
        RGB_Map = RGB_Map.astype(np.uint8)


        if TEST_TRACKING:
            # some data for tracking
            height, width, _ = bev_maps.shape
            bev_img_info = {
                "height": height,
                "width": width,
                "raw_img": RGB_Map
            }
            frame_id = index
            end_time = time.time()
            print(f"FPS: {(1.0/(end_time-start_time)):0.2f}")
            start_time = end_time
            print(f"img_info: width={width} heigth={height}")
            print(f"img_detections: type={type(img_detections)} len={len(img_detections)} elt0:type={type(img_detections[0])} shape={img_detections[0].shape}")

        if TEST_TRACKING_FROM_IMG:
            img2d = cv2.imread(img_paths[0])
            calib = kitti_utils.Calibration(img_paths[0].replace(".png", ".txt").replace("image_2", "calib"))
            objects_pred = pred_utils.predictions_to_kitti_format(img_detections, calib, img2d.shape, opt.img_size, add_conf=True)  
            objects_pred_parsed = tracking_utils.objects_pred_parsing_for_bytetrack(objects_pred, calib, img2d)
            objects_pred_parsed = torch.Tensor(objects_pred_parsed)
            print("objects_pred_parsed.shape = ",objects_pred_parsed.shape)
            # fuck the v does not contain the conf (score); the info is lost in the img_detections -> predictions_to_kitti_format -> objects_pred
            # and there's no sense or proper indexing. I may need to update the object 

            img2d = mview.show_image_with_boxes(img2d, objects_pred, calib, False)
            cv2.imshow("img2d", img2d)
            
            # some data for tracking
            height, width = img2d.shape[:2]
            cam_img_info = {
                "height": height,
                "width": width,
                "raw_img": img2d
            }
            
            frame_id = index
            end_time2 = time.time()
            print(f"FPS: {(1.0/(end_time2-start_time2)):0.2f}")
            start_time2 = end_time2
            print(f"cam_img_info: width={width} heigth={height}")
            print(f"cam_img_detections: elt0:type={type(objects_pred_parsed)} shape={objects_pred_parsed.shape}")

        # get detections
        detections = img_detections[0]

        # Rescale boxes to original image
        detections = utils.rescale_boxes(detections, opt.img_size, RGB_Map.shape[:2])

        if 1:

            if TEST_DETECTION:
                for x, y, w, l, im, re, conf, cls_conf, cls_pred in detections:
                    yaw = np.arctan2(im, re)
                    # Draw rotated box
                    bev_utils.drawRotatedBox(RGB_Map, x, y, w, l, yaw, cnf.colors[int(cls_pred)])

                img2d = cv2.imread(img_paths[0])
                calib = kitti_utils.Calibration(img_paths[0].replace(".png", ".txt").replace("image_2", "calib"))
                objects_pred = pred_utils.predictions_to_kitti_format(img_detections, calib, img2d.shape, opt.img_size, add_conf=False)  
                img2d = mview.show_image_with_boxes(img2d, objects_pred, calib, False)

                cv2.imshow("bev img", RGB_Map)
                cv2.imshow("img2d", img2d)

                index_str = str(index).zfill(3)

                # Save BEV image
                bev_image_path = OUTPUT_FOLDER_COMPLEX_YOLO_BEV / f"BEV_image_{index_str}.jpg"
                cv2.imwrite(str(bev_image_path), RGB_Map)

                # Save 2D image
                img2d_path = OUTPUT_FOLDER_COMPLEX_YOLO_CAM / f"2D_image_{index_str}.jpg"
                cv2.imwrite(str(img2d_path), img2d)

                if cv2.waitKey(0) & 0xFF == 27:
                    break

            if TEST_TRACKING:
                detections_parsed = tracking_utils.parse_detections_for_bev(detections)
                # convert numpy object to tensor
                detections_parsed = torch.Tensor(detections_parsed)

                # the actual tracking
                print("running tracker ...")
                tracklets, timer, online_im = run_tracker_on_frame(frame_id=frame_id, 
                                                                tracker=tracker,
                                                                detections=detections_parsed, 
                                                                aspect_ratio_thresh=opt.aspect_ratio_thresh,
                                                                min_box_area=opt.min_box_area,                                                              
                                                                height=bev_img_info['height'], width=bev_img_info['width'], 
                                                                raw_img=bev_img_info['raw_img'], timer=timer
                                                                )
                print("done running tracker")
                results.extend(tracklets)
                print(f"nb_detection = {detections.shape[0]}")
                print(f"nb_tracklets = {len(tracklets)}")

                # result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
                if opt.save_result and online_im is not None:
                    print(f"detections: {type(detections)}: {detections.shape}") #<class 'torch.Tensor'>: torch.Size([11, 9])
                    print(f"online_im: type={type(online_im)} shape={online_im.shape if online_im is not None else ''}")
                    cv2.imwrite(str(save_folder / Path(img_paths[0]).name), online_im)

                if frame_id % 20 == 0:
                    print('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer.average_time)))

                ch = cv2.waitKey(0)
                if ch == 27 or ch == ord("q") or ch == ord("Q"):
                    break

        if TEST_TRACKING_FROM_IMG:
            detections_parsed = objects_pred_parsed
            assert detections_parsed.shape[0] > 0
            assert detections_parsed.shape[1] == 5

            # the actual tracking
            print("running tracker ...")
            tracklets, timer, online_im = run_tracker_on_frame(frame_id=frame_id, 
                                                            tracker=tracker_img,
                                                            detections=detections_parsed, 
                                                            aspect_ratio_thresh=opt.aspect_ratio_thresh,
                                                            min_box_area=opt.min_box_area,                                                              
                                                            height=cam_img_info['height'], width=cam_img_info['width'], 
                                                            raw_img=cam_img_info['raw_img'], timer=timer2
                                                            )
            print("done running tracker")
            cam_results.extend(tracklets)
            print(f"nb_detection = {detections_parsed.shape[0]}")
            print(f"nb_tracklets = {len(tracklets)}")

            # result_image = predictor.visual(outputs[0], img_info, predictor.confthre)
            if opt.save_result and online_im is not None:
                print(f"detections: {type(detections_parsed)}: {detections_parsed.shape}") #<class 'torch.Tensor'>: torch.Size([11, 9])
                print(f"online_im: type={type(online_im)} shape={online_im.shape if online_im is not None else ''}")
                cv2.imwrite(str(save_folder2 / Path(img_paths[0]).name), online_im)

            if frame_id % 20 == 0:
                print('Processing frame {} ({:.2f} fps)'.format(frame_id, 1. / max(1e-5, timer2.average_time)))

            ch = cv2.waitKey(0)
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break

    if TEST_DETECTION:

        save_img_video_path = OUTPUT_FOLDER_COMPLEX_YOLO_TRACK / "bev-output-complex-yolo-detect.avi"
        images_to_video(
            sorted(list(OUTPUT_FOLDER_COMPLEX_YOLO_BEV.iterdir())),
            save_img_video_path,
        )
        print(f"saved video to {save_img_video_path} ")

        save_img_video_path = OUTPUT_FOLDER_COMPLEX_YOLO_TRACK / "cam-output-complex-yolo-detect.avi"
        images_to_video(
            sorted(list(OUTPUT_FOLDER_COMPLEX_YOLO_CAM.iterdir())),
            save_img_video_path,
        )
        print(f"saved video to {save_img_video_path} ")
    
    if TEST_TRACKING:
        if opt.save_result:
            with open(res_file, 'w') as f:
                f.writelines(results)
            print(f"save results to {res_file}")
        
        save_img_video_path = OUTPUT_FOLDER_COMPLEX_YOLO_TRACK / "bev-output-complex-yolo-track.avi"

        images_to_video(
            sorted(list(Path(save_folder).iterdir())),
            save_img_video_path,
        )

        print(f"saved video to {save_img_video_path} ")
    
    if TEST_TRACKING_FROM_IMG:
        if opt.save_result:
            with open(res_file2, 'w') as f:
                f.writelines(cam_results)
            print(f"save cam_results to {res_file2}")
        
        save_img_video_path = OUTPUT_FOLDER_COMPLEX_YOLO_TRACK / "cam-output-complex-yolo-track.avi"

        images_to_video(
            sorted(list(Path(save_folder2).iterdir())),
            save_img_video_path,
        )

        print(f"saved video to {save_img_video_path} ")

