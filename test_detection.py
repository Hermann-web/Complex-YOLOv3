import numpy as np
import argparse
import cv2
import time
import torch

from constants import OUTPUT_FOLDER
from modules.images_to_video import images_to_video
import utils.utils as utils
from models import *
import torch.utils.data as torch_data

import utils.kitti_utils as kitti_utils
import utils.kitti_bev_utils as bev_utils
import utils.kitti_prediction_utils as pred_utils
from utils.kitti_yolo_dataset import KittiYOLODataset
import utils.config as cnf
import utils.mayavi_viewer as mview

OUTPUT_FOLDER_COMPLEX_YOLO = OUTPUT_FOLDER / "complex-yolo-track"

OUTPUT_FOLDER_COMPLEX_YOLO_BEV = OUTPUT_FOLDER_COMPLEX_YOLO/"dev-images"

OUTPUT_FOLDER_COMPLEX_YOLO_CAM = OUTPUT_FOLDER_COMPLEX_YOLO/"cam-images"

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
    opt = parser.parse_args()
    print(opt)



    classes = utils.load_classes(opt.class_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)
    # Load checkpoint weights
    model.load_state_dict(torch.load(opt.weights_path, map_location=device))
    # Eval mode
    model.eval()
    
    dataset = KittiYOLODataset(cnf.root_dir, split=opt.split, mode='TEST', folder=opt.folder, data_aug=False)
    data_loader = torch_data.DataLoader(dataset, 1, shuffle=False)

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    start_time = time.time()                        
    for index, (img_paths, bev_maps) in enumerate(data_loader):
        
        # Configure bev image
        input_imgs = Variable(bev_maps.type(Tensor))

        # Get detections 
        with torch.no_grad():
            detections = model(input_imgs)
            detections = utils.non_max_suppression_rotated_bbox(detections, opt.conf_thres, opt.nms_thres) 
        
        end_time = time.time()
        print(f"FPS: {(1.0/(end_time-start_time)):0.2f}")
        start_time = end_time

        img_detections = []  # Stores detections for each image index
        img_detections.extend(detections)

        bev_maps = torch.squeeze(bev_maps).numpy()

        RGB_Map = np.zeros((cnf.BEV_WIDTH, cnf.BEV_WIDTH, 3))
        RGB_Map[:, :, 2] = bev_maps[0, :, :]  # r_map
        RGB_Map[:, :, 1] = bev_maps[1, :, :]  # g_map
        RGB_Map[:, :, 0] = bev_maps[2, :, :]  # b_map
        
        RGB_Map *= 255
        RGB_Map = RGB_Map.astype(np.uint8)
        
        for detections in img_detections:
            if detections is None:
                continue

            # Rescale boxes to original image
            detections = utils.rescale_boxes(detections, opt.img_size, RGB_Map.shape[:2])
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

    images_to_video(
        sorted(list(OUTPUT_FOLDER_COMPLEX_YOLO_BEV.iterdir())),
        OUTPUT_FOLDER_COMPLEX_YOLO / "bev-output.avi",
    )
    images_to_video(
        sorted(list(OUTPUT_FOLDER_COMPLEX_YOLO_CAM.iterdir())),
        OUTPUT_FOLDER_COMPLEX_YOLO / "img2d-output.avi",
    )
