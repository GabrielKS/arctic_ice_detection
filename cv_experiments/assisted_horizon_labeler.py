"""
A tool to quickly label horizons on lots of images by trying to automatically detect the horizon and then letting the
user adjust. Controls:
 - q: save data file and quit
 - r: reset to automatic horizon
 - space: finalize and move on to the next image
 - s: save data file without quitting
 - k: skip to next image without finalizing this one
 - mouse x: horizon roll
 - mouse y: horizon pitch
"""

import os
import cv2
import cv_experiments.cv_common as cc
import pandas as pd
import numpy as np
import time

image_dir = os.path.abspath("../to_be_horizoned")
output_dir = os.path.abspath("../tabular/horizons")
filename_prefix = "horizons"
filename_suffix = "_%Y-%m-%dT%H-%M-%S"
dwidth, dheight = 1250, 700  # Display width and height
dname = "The Assisted Horizon Labeler"

columns = ["filename", "pitchFrac", "rollSlope"]
horizon_table = []
state = {"coords": [0, 0], "pre_offset": [dwidth/2, dheight/2], "post_offset": [0, 0], "mouse": [dwidth/2, dheight/2], "img": None}

def to_csv(filename):
    df = pd.DataFrame(horizon_table, columns=columns).set_index(columns[0])
    df.to_csv(os.path.join(output_dir, filename))

def from_csv(filename):
    df = pd.read_csv(os.path.join(output_dir, filename))
    return df.values.tolist()

def load():
    if os.path.exists(os.path.join(output_dir, filename_prefix+".csv")):
        horizon_table.extend(from_csv(filename_prefix+".csv"))

def save():
    to_csv(filename_prefix+".csv")
    to_csv(filename_prefix+time.strftime(filename_suffix)+".csv")

def best_hough(img):
    edges = cv2.Canny(cv2.resize(img, (120, 67)), 12800, 22400, apertureSize=7)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, None, 5, 15)

    if lines is not None:
        # Find the longest line and extend it across the frame
        (x1, y1, x2, y2) = max(lines, key=lambda line: (line[0][2]-line[0][0])**2+(line[0][1]-line[0][3])**2)[0]
        slope = (y2-y1)/(x2-x1)
        if abs(slope) < 100:
            return [-(y1+y2)/2/67+0.5, -slope]
    return [0, 0]

def mouser(event, x, y, flags, param):
    state["mouse"] = [x, y]
    x -= state["pre_offset"][0]
    y -= state["pre_offset"][1]
    state["coords"][0] = -y/dheight*0.5+state["post_offset"][0]
    state["coords"][1] = x/dwidth*0.5+state["post_offset"][1]
    draw_img()

def draw_img():
    img = state["img"]
    if img is None: return
    pitch_frac, roll_slope = state["coords"]
    img = cv2.resize(img, (dwidth, dheight))
    y0 = dheight*(0.5-pitch_frac)
    yoffset = -roll_slope*dwidth/2
    cv2.line(img, (0, int(y0-yoffset)), (dwidth, int(y0+yoffset)), (255, 0, 0), 1, cv2.LINE_AA)
    cv2.imshow(dname, img)

def reset():
    state["coords"] = best_hough(state["img"])
    state["pre_offset"] = state["mouse"]
    state["post_offset"] = state["coords"].copy()

def handle_image(img, name):
    state["img"] = img
    reset()
    quit = False
    skip = False
    while True:
        draw_img()
        key = cv2.waitKey(0) & 0xff
        if   key == ord(' '): break
        elif key == ord('q'): quit = True; break
        elif key == ord('s'): save()
        elif key == ord('r'): reset()
        elif key == ord('k'): skip = True; break
    if not skip: horizon_table.append([name, *state["coords"]])
    return quit

def main():
    load()
    cv2.namedWindow(dname)
    cv2.setMouseCallback(dname, mouser)
    for img,name in cc.load_dir(image_dir, colormode=cv2.IMREAD_COLOR):
        if name in set(row[0] for row in horizon_table): continue
        print(name)
        done = handle_image(img, name)
        if done: break
    save()

if __name__ == "__main__":
    main()
