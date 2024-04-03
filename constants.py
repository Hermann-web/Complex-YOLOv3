from pathlib import Path

OUTPUT_FOLDER = Path("docs/images/output")

# Create folders if they don't exist
OUTPUT_FOLDER_BEV = OUTPUT_FOLDER / "bev"
OUTPUT_FOLDER_CAM = OUTPUT_FOLDER / "cam"

OUTPUT_FOLDER_BEV.mkdir(parents=True, exist_ok=True)
OUTPUT_FOLDER_CAM.mkdir(parents=True, exist_ok=True)


BYTETRACK_TRACK_IMAGES_FOLDER = Path("bytetrack-outputs/")
BYTETRACK_TRACK_IMAGES_FOLDER.mkdir(exist_ok=True)
