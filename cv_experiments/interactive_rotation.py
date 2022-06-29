import os
import numpy as np
import cv2

input_dir = os.path.abspath("../representatives/rotation")
sources = None

def get_images():
    global sources
    image_paths = sorted(f.path for f in os.scandir(input_dir) if f.is_file() and os.path.basename(f.path)[0] != '.')
    sources = list(zip(map(lambda f: cv2.imread(f, 0), image_paths), map(lambda f: os.path.basename(f), image_paths)))

def mouser(event, x, y, flags, param):
    print(x)
    print(y)
    angle = np.arctan2(x-100, y-100)
    for i, (img, name) in enumerate(sources):
        do_image(img, name, angle)

def rotate_crop(width, height, angle):  # Inspired by https://stackoverflow.com/a/16778797
    longer, shorter = max(width, height), min(width, height)
    sin_a, cos_a = abs(np.sin(angle)), abs(np.cos(angle))
    if shorter <= 2*sin_a*cos_a*longer or abs(sin_a-cos_a) < 1e-5:
        rlonger, rshorter = 0.5*shorter/sin_a, 0.5*shorter/cos_a
        rwidth, rheight = (rlonger, rshorter) if width > height else (rshorter, rlonger)
    else:
        csd = cos_a*cos_a-sin_a*sin_a
        rwidth, rheight = (width*cos_a - height*sin_a)/csd, (height*cos_a - width*sin_a)/csd
    return rwidth, rheight

def do_image(img, name, angle):
    height, width = img.shape[:2]
    sf = min(width, height)/np.sqrt(width**2+height**2)
    rwidth, rheight = rotate_crop(width, height, angle)
    m = cv2.getRotationMatrix2D((width//2, height//2), angle*180/np.pi, sf)
    rotated = cv2.warpAffine(img, m, (width, height))
    xo = int((width-rwidth*sf)/2)+5
    yo = int((height-rheight*sf)/2)+5
    rotated = rotated[yo:rotated.shape[0]-yo, xo:rotated.shape[1]-xo]
    owidth, oheight = [min(width, height)]*2
    e = np.ones((oheight, owidth), np.uint8)*127
    xo = int((owidth-rotated.shape[1])/2)
    yo = int((oheight-rotated.shape[0])/2)
    e[yo:rotated.shape[0]+yo, xo:rotated.shape[1]+xo] = rotated
    cv2.imshow(name, e)

def main():
    get_images()
    for i, (img, name) in enumerate(sources):
        do_image(img, name, 0)
    for _,name in sources: cv2.setMouseCallback(name, mouser)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()
