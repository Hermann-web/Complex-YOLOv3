from typing import Dict, List
import numpy as np
import cv2
import utils.kitti_utils as kitti_utils
import utils.kitti_bev_utils as bev_utils

def parse_detections_for_bev(detections):
    detections_parsed = np.zeros((len(detections), 5)) #x1,y1,x2,y2,score

    ix=-1
    for x, y, w, l, im, re, conf, cls_conf, cls_pred in detections:
        ix +=1
        yaw = np.arctan2(im, re)
                
        bev_corners = bev_utils.get_corners(x, y, w, l, yaw) # = (4,2) = 4 points(x,y)
        if 0:
            x1, y1 = bev_corners[0] #front left
            x2, y2 = bev_corners[2] #rear right
        else:
            x1, y1 = bev_corners.max(axis=0)
            x2, y2 = bev_corners.min(axis=0)

        print(f"bbox: x1y1x2y2={x1, y1, x2, y2} score={cls_conf}")
        detections_parsed[ix, [0,1,2,3]] = (x2, y2, x1, y1)
        detections_parsed[ix, [4]] = float(conf) #int(confidence))

    return detections_parsed


def objects_pred_parsing_for_bytetrack(objects_pred:List[kitti_utils.Object3d], calib:kitti_utils.Calibration, img2d:cv2.typing.MatLike):
    ''' Draw 3d bounding box in image
        qs: (8,3) array of vertices for the 3d box in following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
    '''
    
    img2d = img2d.copy()

    preds_parsed = np.zeros((len(objects_pred),5))
    for i,obj in enumerate(objects_pred):
        box3d_pts_2d, box3d_pts_3d = kitti_utils.compute_box_3d(obj, calib.P)

        if box3d_pts_2d is None:
            continue
        
        if 0:
            L = [2,6,7,3]
            L = [2,5,4,3]
            print("box3d_pts_2d= ",box3d_pts_2d)
            x1, y1 = box3d_pts_2d[L[0]]
            x2, y2 = box3d_pts_2d[L[2]]
            bev_corners = box3d_pts_2d[L]
        else:
            x1, y1 = box3d_pts_2d.max(axis=0)
            x2, y2 = box3d_pts_2d.min(axis=0)
            #front left, rear left, rear right, front right
            bev_corners = np.array([
                [x1, y1],
                [x1, y2],
                [x2, y2],
                [x2, y1]
            ])

        preds_parsed[i,:4] = (x2, y2, x1, y1)
        preds_parsed[i, 4] = float(obj.conf)

        
        corners_int = bev_corners.reshape(-1, 1, 2).astype(int)
        color = [0, 255, 255]
        cv2.polylines(img2d, [corners_int], True, color, 2)
        corners_int = bev_corners.reshape(-1, 2)
        cv2.line(img2d, (corners_int[0, 0].round().astype(int), corners_int[0, 1].round().astype(int)), (corners_int[3, 0].round().astype(int), corners_int[3, 1].round().astype(int)), (255, 255, 0), 2)
    
    # cv2.imshow("img2d with plane boxes", img2d)

    # print("preds_parsed = ",preds_parsed)
    return preds_parsed
