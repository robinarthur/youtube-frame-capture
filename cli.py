import cv2
import os
from vidgear.gears import CamGear
import sys
import argparse
import time

parser = argparse.ArgumentParser(
    prog="downlaod_youtube_frames.",
    description="Program to automatically download youtube images datasets.",
    epilog="A simple and nice cli script to create youtube datasets",
)

parser.add_argument("--videolink", required=True, type=str, help="YouTube video link")
parser.add_argument("--destination", required=True, type=str, help="Target path to save imgz")
parser.add_argument("--showframe", required=False, action="store_true", help="Show the resulting frames")
parser.add_argument("--skip_frames", required=False, type=int, default=1, help="give an int == fps to get only 1 frame per second; fps/2 for 2 frames per second to store and process")

args = parser.parse_args()
path = args.destination
source = args.videolink
show_frame = args.showframe
skip_frames = args.skip_frames

time_start = time.time()

stream = CamGear(
    source=source,
    yt_dlp_format="311",
    stream_mode=True,
    time_delay=1,
    logging=True,
).start()

currentframe, save_counter = 0, 0

while True:

    frame = stream.read()
    if frame is None:
        break
    if show_frame:
        cv2.imshow("Output Frame", frame)
        
    # Only save frames that match the interval
    if currentframe % skip_frames == 0:
        # For sequential numbering use save_counter instead of currentframe
        name = path + f"./frame_{save_counter:05d}.jpg"  # Pad with zeros
        print(f"Creating... {name}")
        cv2.imwrite(name, frame)
        save_counter += 1  # Only increment when saving
        
    currentframe += 1
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
stream.stop()

time_end = time.time()

time_taken = round(time_end - time_start, 3)
print("=======================================================")
print("-------------------------------------------------------")
print(f"## The time taken to create dataset: {time_taken} seconds ##")
print("-------------------------------------------------------")
print("=======================================================")
