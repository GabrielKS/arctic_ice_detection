import os
import cv2
import numpy as np
import cv_experiments.cv_common as cc

input = os.path.abspath("../representatives/fft_example.jpg")
output_root = os.path.abspath("../intermediate_images")

logscale = 13  # Experimentally chosen to fit myfft output in 255
shift_axes = [0]

# Done on an airplane with no Internet access; may not be the best way
def myfft(img, concat=True):
    f = np.fft.fftshift(np.fft.rfft2(np.float32(img)), axes=shift_axes)
    pic = logscale*np.log(f)
    pic1 = np.abs(pic)  # Magnitude
    pic2 = np.angle(pic)*256/(2*np.pi)+127  # Phase, scaled to fit in a pixel
    if concat: return np.concatenate([pic1, pic2], axis=1)
    else: return pic1, pic2

def myifft(img, img2=None):
    if img2 is not None: ipic1, ipic2 = img, img2
    else: ipic1, ipic2 = img[:,0:img.shape[1]//2], img[:,img.shape[1]//2:]
    upic2 = (ipic2-127)/255*2*np.pi  # Undo the angle scaling
    cpic = ipic1*(np.cos(upic2)+1j*np.sin(upic2))  # Polar to complex
    g = np.exp(cpic/logscale)
    des = np.fft.irfft2(np.fft.ifftshift(g, axes=shift_axes))
    return np.clip(des, 0, 255)

def main():
    img = cv2.imread(input, cv2.IMREAD_GRAYSCALE)
    img = cc.undistort(img, "large")
    # cv2.imshow("img", img)

    the_fft = myfft(img)
    cv2.imshow("fft", np.uint8(the_fft))

    the_ifft = myifft(np.uint8(the_fft))
    cv2.imshow("ifft", np.uint8(the_ifft))

    components = myfft(img, concat=False)
    cv2.imwrite(os.path.join(output_root, "fft_mag.png"), np.uint8(components[0]))
    cv2.imwrite(os.path.join(output_root, "fft_phs.png"), np.uint8(components[1]))

    cv2.waitKey(0)

if __name__ == "__main__":
    main()
