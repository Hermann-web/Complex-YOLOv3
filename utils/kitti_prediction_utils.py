from typing import List
import numpy as np
import math

import torch
import utils.kitti_utils as kitti_utils
import utils.kitti_aug_utils as aug_utils
import utils.kitti_bev_utils as bev_utils
import utils.config as cnf

def predictions_to_kitti_format(img_detections:List[torch.Tensor], calib:kitti_utils.Calibration, img_shape_2d, img_size:int, RGB_Map=None, add_conf:bool=False):
    predictions = np.zeros([50, 8], dtype=np.float32)
    count = 0
    for detections in img_detections:
        if detections is None:
            continue
        # Rescale boxes to original image
        for x, y, w, l, im, re, conf, cls_conf, cls_pred in detections:
            yaw = np.arctan2(im, re)
            predictions[count, :] = cls_pred, x/img_size, y/img_size, w/img_size, l/img_size, im, re, conf
            count += 1
    print("predictions.shape[1]-1 = ",predictions.shape[1]-1)
    print("dddd = ",predictions[:, :predictions.shape[1]-1].shape)
    print("predictions.shape[0] = ",predictions.shape[0])
    predictions = bev_utils.inverse_yolo_target(predictions, cnf.boundary, add_conf=True)
    assert predictions.ndim ==2 and predictions.shape[1]==9
    if predictions.shape[0]:
        predictions[:, 1: predictions.shape[1]-1] = aug_utils.lidar_to_camera_box(predictions[:, 1:predictions.shape[1]-1], calib.V2C, calib.R0, calib.P)

    objects_new: List[kitti_utils.Object3d] = []
    corners3d: List[np.ndarray] = []
    for index, l in enumerate(predictions):

        dd = {0:"Car", 1:"Pedestrian", 2:"Cyclist"}
        str = dd.get(l[0], "DontCare")
        line = '%s -1 -1 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0' % str

        obj = kitti_utils.Object3d(line)
        obj.t = l[1:4]
        obj.h,obj.w,obj.l = l[4:7]
        obj.ry = np.arctan2(math.sin(l[7]), math.cos(l[7]))
        if add_conf:
            obj.conf = l[8]
    
        _, corners_3d = kitti_utils.compute_box_3d(obj, calib.P)
        corners3d.append(corners_3d)
        objects_new.append(obj)

    if len(corners3d) > 0:
        corners3d = np.array(corners3d)
        img_boxes, _ = calib.corners3d_to_img_boxes(corners3d)

        img_boxes[:, 0] = np.clip(img_boxes[:, 0], 0, img_shape_2d[1] - 1)
        img_boxes[:, 1] = np.clip(img_boxes[:, 1], 0, img_shape_2d[0] - 1)
        img_boxes[:, 2] = np.clip(img_boxes[:, 2], 0, img_shape_2d[1] - 1)
        img_boxes[:, 3] = np.clip(img_boxes[:, 3], 0, img_shape_2d[0] - 1)

        img_boxes_w = img_boxes[:, 2] - img_boxes[:, 0]
        img_boxes_h = img_boxes[:, 3] - img_boxes[:, 1]
        box_valid_mask = np.logical_and(img_boxes_w < img_shape_2d[1] * 0.8, img_boxes_h < img_shape_2d[0] * 0.8)

    for i, obj in enumerate(objects_new):
        x, z, ry = obj.t[0], obj.t[2], obj.ry
        beta = np.arctan2(z, x)
        alpha = -np.sign(beta) * np.pi / 2 + beta + ry

        obj.alpha = alpha
        obj.box2d = img_boxes[i, :]

    if RGB_Map is not None:
        labels, noObjectLabels = bev_utils.read_labels_for_bevbox(objects_new)    
        if not noObjectLabels:
            labels[:, 1:] = aug_utils.camera_to_lidar_box(labels[:, 1:], calib.V2C, calib.R0, calib.P) # convert rect cam to velo cord

        target = bev_utils.build_yolo_target(labels)
        bev_utils.draw_box_in_bev(RGB_Map, target)

    return objects_new
