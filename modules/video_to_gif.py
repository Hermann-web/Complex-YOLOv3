"""
video_to_gif.py

A script to convert a video file into a GIF using OpenCV.

Dependencies:
- OpenCV (cv2): A computer vision library for image and video processing.
- pathlib: An object-oriented interface to filesystem paths.
- imageio: A Python library for reading and writing a wide range of image, video, and GIF formats.

Usage:
    $ python video_to_gif.py <input_video> <output_gif>

Arguments:
    input_video: Path to the input video file. 
                 Example: input_video.mp4
    output_gif: Path to the output GIF file. 
                Example: output_gif.gif

Example:
    $ python video_to_gif.py input_video.mp4 output_gif.gif

Author: Your Name
Date: Date of creation/modification
"""

from pathlib import Path

import cv2
import imageio


def video_to_gif(input_video, output_gif):
    """
    Convert a video file into a GIF.

    Args:
        input_video (str): Path to the input video file.
        output_gif (str): Path to the output GIF file.

    Returns:
        bool: True if the GIF conversion is successful, False otherwise.
    """
    # Check if the input video file exists
    input_video_path = Path(input_video)
    if not input_video_path.exists():
        print(f"Input video file {input_video} not found.")
        return False

    # Initialize video capture
    cap = cv2.VideoCapture(str(input_video_path))

    # Check if the video is opened successfully
    if not cap.isOpened():
        print(f"Error opening video file: {input_video}")
        return False

    # Convert each frame to RGB and store in a list
    image_lst = []
    while True:
        try:
            ret, frame = cap.read()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except cv2.error:
            break

        image_lst.append(frame_rgb)

        cv2.imshow("Video to GIF", frame)
        key = cv2.waitKey(1)
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Convert the list of frames to GIF using imageio
    try:
        imageio.mimsave(str(output_gif), image_lst)
        return True
    except Exception as e:
        print(f"Error converting frames to GIF: {e}")
        return False


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Convert video to GIF")
    parser.add_argument(
        "input_video", help="Input video file. Example: input_video.mp4"
    )
    parser.add_argument("output_gif", help="Output GIF file. Example: output_gif.gif")
    args = parser.parse_args()

    input_video = args.input_video
    output_gif = args.output_gif

    success = video_to_gif(input_video, output_gif)
    if success:
        print("GIF conversion successful!")
    else:
        print("GIF conversion failed.")


if __name__ == "__main__":
    main()
