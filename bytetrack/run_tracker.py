
from bytetrack.timer import Timer 
from bytetrack.visualize import plot_tracking

def run_tracker_on_frame(frame_id, tracker, detections, height:int, width:int, raw_img:str, aspect_ratio_thresh:float=1.6, min_box_area:float=10, timer:Timer=None):
    """
    COPIED FROM Bytetrack_repo/tools/demo_track.py
    """
    results = []

    # Run tracker
    online_targets = tracker.update(detections, [height, width], [height, width])

    online_tlwhs = []
    online_ids = []
    online_scores = []
    for t in online_targets:
        tlwh = t.tlwh
        tid = t.track_id
        vertical = tlwh[2] / tlwh[3] > aspect_ratio_thresh
        if tlwh[2] * tlwh[3] > min_box_area and not vertical:
            online_tlwhs.append(tlwh)
            online_ids.append(tid)
            online_scores.append(t.score)
            # save results
            results.append(
                f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
            )
            if timer is not None: timer.toc()
            online_im = plot_tracking(
                raw_img, online_tlwhs, online_ids, frame_id=frame_id, fps=1. / timer.average_time
            )
        else:
            if timer is not None: timer.toc()
            online_im = raw_img

    return results, timer, online_im