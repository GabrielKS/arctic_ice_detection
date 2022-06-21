import os
import cv2
import numpy as np
import cv_experiments.prototype_fft as prototype_fft

intermediate_root = os.path.abspath("../intermediate_images")

def main():
    while 1:
        # the_ifft = np.uint8(prototype_fft.myifft(
        #     cv2.imread(os.path.join(intermediate_root, "fft.png"), cv2.IMREAD_GRAYSCALE)))
        the_ifft = np.uint8(prototype_fft.myifft(
            cv2.imread(os.path.join(intermediate_root, "fft_mag.png"), cv2.IMREAD_GRAYSCALE),
            cv2.imread(os.path.join(intermediate_root, "fft_phs.png"), cv2.IMREAD_GRAYSCALE)))
        cv2.imshow("ifft", the_ifft)
        cv2.imwrite(os.path.join(intermediate_root, "ifft.png"), the_ifft)
        # Reload from disk every second so we can mess with the ffts elsewhere and see the results live
        if cv2.waitKey(1000) & 0xff == ord('q'): break

if __name__ == "__main__":
    main()
