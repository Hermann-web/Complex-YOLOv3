"""
images_to_video.py

A script to convert a sequence of images into a video file using OpenCV.

Dependencies:
- OpenCV (cv2): A computer vision library for image and video processing.
- pathlib: An object-oriented interface to filesystem paths.

Usage:
    $ python images_to_video.py <input_images> <output_video>

Arguments:
    input_images: List of input image files to be converted into a video file. 
                  Example: image1.jpg image2.jpg
    output_video: Path to the output video file. 
                  Example: output_video.avi

Example:
    $ python images_to_video.py image1.jpg image2.jpg output_video.avi

Author: Your Name
Date: Date of creation/modification
"""

from pathlib import Path

import cv2


def images_to_video(image_files, output_path):
    img_array = []

    # Check if the output directory exists
    output_directory = Path(output_path).parent
    if not output_directory.exists():
        print(f"Output directory {output_directory} does not exist.")
        return False

    for img_filepath in image_files:
        img_filepath = Path(img_filepath)
        if not img_filepath.exists():
            print(f"File {img_filepath} not found.")
            return False

        img = cv2.imread(str(img_filepath))
        if img is None:
            print(f"Error reading image file: {img_filepath}")
            return False

        height, width, _ = img.shape
        size = (width, height)
        img_array.append(img)

    if not img_array:
        print("No valid images found.")
        return False

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*"DIVX")
    fps = 15
    out = cv2.VideoWriter(str(output_path), fourcc, fps, size)

    # Write images to video
    for img in img_array:
        out.write(img)

    # Release video writer
    out.release()

    return True


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Convert images to a video file")
    parser.add_argument(
        "input_images",
        nargs="+",
        help="Input image files. Example: image1.jpg image2.jpg",
    )
    parser.add_argument(
        "output_video", help="Output video file. Example: output_video.avi"
    )
    args = parser.parse_args()

    input_images = args.input_images
    output_video = args.output_video

    success = images_to_video(input_images, output_video)
    if success:
        print("Video conversion successful!")
    else:
        print("Video conversion failed.")


if __name__ == "__main__":
    main()
