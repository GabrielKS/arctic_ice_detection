"""
A modular implementation of the routines necessary to get from a semantically segmented camera image to a map in real
space. Prototyped in seg2info_exploration.ipynb and refined in seg2info_exploration_2.ipynb. Highly configurable with
sensible defaults throughout.
"""

import os
import cv2
import matplotlib.cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools
import copy

import cv_experiments.cv_common as cc

class Seg2Info:
    def __init__(self, processing_props={}, camera_props={}, output_props={}):
        # The general strategy here is: populate params with constant defaults, update with user specified values, fill
        # in any derived defaults if they aren't user specified. That way the derived defaults are derived from the
        # user specified values if relevant.

        # Processing parameters
        self.proc_props = {
            "label2value": {"water": 0, "sky": 1, "ice": 2, "other": 3, "none": 4},
            "four_values": ["water", "sky", "ice", "rest"],  # The four channels to use in our four-channel representation
            "upscale_factor": 4,  # Factor by which to upscale before doing anything lossy
            "sky_buffer": 16,  # Number of original (pre-upscaled) pixels' buffer of sky to leave when cropping
            "data_range": [0, 1],  # Range of valid data in our representation
            "transition_sample_locs": np.linspace(0.1, 0.8, 11),  # Range of brightness values to sample to find transition
            "logb": np.log10,  # The log and exponential functions to use
            "expb": lambda a: 10**a
        }
        self.proc_props.update(processing_props)
        if "rest_value" not in self.proc_props: self.proc_props["rest_value"] = np.eye(len(self.proc_props["four_values"]))[self.proc_props["four_values"].index("rest")]  # Representation of "rest" for use as a fill value

        # Various properties of the physical configuration of the camera
        self.cam_props = {
            "camera_height": 2.2,  # Previously 5, but 2.2 is more accurate for my configuration
            "horiz_fov": 98.0,
            "vert_fov": 54.5,
            # The image, pre-undistortion, has a vertical field of view of vert_fov.
            # Undistortion shrinks the middle of the image by dist_factor, resulting in a greater apparent vertical field of view.
            "dist_factor": 1.19
        }
        self.cam_props.update(camera_props)
        if "horizon_distance" not in self.cam_props: self.cam_props["horizon_distance"] = 3570.*np.sqrt(self.cam_props["camera_height"])
        if "near_distance" not in self.cam_props: self.cam_props["near_distance"] = np.tan(np.arctan(self.cam_props["horizon_distance"]/self.cam_props["camera_height"]) - 
            np.deg2rad(self.cam_props["vert_fov"]*self.cam_props["dist_factor"]))*self.cam_props["camera_height"]

        # Various properties of the desired output
        self.out_props = {
            "width": 500,
            "height": 279,
            "t_range": self.cam_props["horiz_fov"]*1.1,
            "min_distance": 2,
            "max_distance": 20000
        }
        self.out_props.update(output_props)


    # INPUT convenience functions
    @staticmethod
    def load_dirs(seginput_path, segmap_path, segmap_ext = ".png", sort_fn = lambda image: image["name"]):
        """Load original images and masks from directories into a dictionary structure"""
        input_exts = (".png", ".jpg", ".jpeg")
        images = []
        for path in os.listdir(seginput_path):
            if not path.endswith(input_exts): continue
            name = os.path.basename(path)
            images.append({"name": name, "seginput": cv2.imread(os.path.join(seginput_path, path)), "segmap":
                cv2.imread(os.path.join(segmap_path, os.path.splitext(name)[0]+segmap_ext), cv2.IMREAD_GRAYSCALE)})
        if sort_fn is not None: images.sort(key=sort_fn)
        return images


    # PREPROCESSING
    @staticmethod
    def apply(fn, images, src_key, dest_key=None):
        """Apply the function to every image, optionally storing the output
        :param fn: the function to apply
        :param images: the images, given as a list of dictionaries
        :param src_key: the key, or list of keys, of the input(s) from the images dictionary
        :param dest_key: the key, or list of keys, of the output(s) to the images dictionary.\
            If `None`, return the output instead of storing it
        """
        results = []
        for image in images:
            result = fn(*[image[k] for k in src_key]) if (type(src_key) == list) else fn(image[src_key])
            if dest_key is None: results.append(result)
            else:
                if type(dest_key) == list:
                    for i,k in enumerate(dest_key): image[k] = result[i]
                else: image[dest_key] = result
        if dest_key is None: return results

    def clamp(self, data):
        """Restrict each pixel to the valid range"""
        return np.clip(data, *self.proc_props["data_range"])

    def cat2cont(self, img):
        """Turn a categorical map into a single-channel map where 0 is water, 1 is ice, and everything else is `NaN`"""
        water_cat, ice_cat = self.proc_props["label2value"]["water"], self.proc_props["label2value"]["ice"]
        result = np.where((img != water_cat) & (img != ice_cat), np.nan, img)
        result = np.where(result == water_cat, 0., result)
        result = np.where(result == ice_cat, 1., result)
        return result

    def one_hot_ify(self, img):
        """Turn a categorical map into a one-hot encoded map with one channel per category"""
        return np.stack([img == v for v in self.proc_props["label2value"].values()], axis=-1).astype(np.float32)

    def one_hot_four(self, img):
        """Turn a categorical map into a one-hot, four-channel map"""
        channels = [((img == self.proc_props["label2value"][name]) if name in self.proc_props["label2value"] else False) for name in self.proc_props["four_values"]]
        stacked = np.stack(np.broadcast_arrays(*channels), axis=-1).astype(np.float32)
        # Insert the "rest" channel where there's a channel name not in cat_map
        rest_i = [name in self.proc_props["label2value"] for name in self.proc_props["four_values"]].index(False)
        stacked[:,:,rest_i] = 1-stacked.sum(axis=-1)
        return self.clamp(stacked)

    def upscale(self, img, interpolation_method=cv2.INTER_LINEAR):
        """Scale an image"""
        return self.clamp(cv2.resize(img, (int(img.shape[1]*self.proc_props["upscale_factor"]), int(img.shape[0]*self.proc_props["upscale_factor"])),
            interpolation=interpolation_method))

    def undistort(self, img, interpolation_method=cv2.INTER_LINEAR):
        """Undistort an image"""
        return self.clamp(cc.undistort(img, "new", interpolation_method, self.proc_props["rest_value"]))

    def sky_edge(self, img, rest_thresh=0.5):
        """Find the sky edge by searching down each column for the transition from no water/ice to some water/ice.\
            Do this with a bunch of different samples, throw out the bad ones if possible, and average
        """
        oceanness = np.max([img[:,:,self.proc_props["four_values"].index("water")], img[:,:,self.proc_props["four_values"].index("ice")]], axis=0)
        # Enforce monotonicity (took me a while to realize the lack of this was causing problems...)
        oceanness = np.maximum.accumulate(oceanness, axis=0)
        isrest = img[:,:,self.proc_props["four_values"].index("rest")] > rest_thresh

        samples = np.stack([np.apply_along_axis(np.searchsorted, 0, oceanness, x, side="right")
            for x in self.proc_props["transition_sample_locs"]], axis=-1)
        # If the pixel immediately above us is `rest`, we haven't actually found anything
        y = np.clip(samples.T-self.proc_props["upscale_factor"], 0, None)
        samples = np.where(~isrest[y, np.arange(img.shape[1])].T, samples, np.nan)
        # If we hit a bounds, we haven't actually found anything
        filtered_samples = np.where(((samples > 0) & (samples < img.shape[0])), samples, np.nan)
        result = np.mean(filtered_samples, axis=-1)
        # Report np.nan if anything is bordering `rest`; report the bounds if anything is out of bounds
        return np.where(np.isnan(result), np.min(samples, axis=-1), result)
    
    # We need to filter out NaNs before passing to fit
    def find_horizon(self, img):
        """Find a horizon as the line of best fit of the sky_edge"""
        width = img.shape[1]
        x = np.arange(width)-width//2
        y = self.sky_edge(img)
        valids = ~np.isnan(x) & ~np.isnan(y)
        return np.polynomial.Polynomial.fit(x[valids], y[valids], 1).convert()

    def rotate_image(self, img, line, interpolation_method=cv2.INTER_LINEAR):
        """Rotate an image to make the given line horizontal without changing its y-intercept"""
        height,width = img.shape[:2]
        intercept,slope = line
        center = (width//2, intercept)  # Rotate about the midpoint of the horizon line
        angle = np.arctan(slope)
        scale = np.cos(angle)  # Scale down so we don't erase information
        rot_mat = cv2.getRotationMatrix2D(center, angle*180/np.pi, scale)
        # Add some extra height so we don't erase information
        result = cv2.warpAffine(img, rot_mat, (width, height+width),
            flags=interpolation_method, borderValue=self.proc_props["rest_value"])
        return self.clamp(result), scale, height

    @staticmethod
    def interpolate_sky_edge(edge, line):
        padded = np.zeros(len(edge)+2, dtype=edge.dtype)
        intercept, slope = line
        padded[1:-1] = edge
        padded[0] = -slope*len(edge)/2+intercept
        padded[-1] = slope*len(edge)/2+intercept
        interpolated = pd.Series(padded).interpolate()
        return interpolated.to_numpy()[1:-1]
    
    # Identical to original except added interpolation line
    def adjust_and_crop(self, img, line, height, interpolation_method=cv2.INTER_LINEAR):
        """After an image has been rotated so the horizon is roughly horizontal, this translates\
            each column of pixels to make the horizon perfectly flat, then moves the horizon so it is\
            buffer_px away from the top of the image
        """

        buffer_px = self.proc_props["upscale_factor"]*self.proc_props["sky_buffer"]
        intercept,_ = line
        intercept = int(intercept)
        # Search search_range pixels up and down of the intercept for the [ice and water] to sky edge
        search_range = int(0.05*height)
        new_edge = self.sky_edge(img[(intercept-search_range):(intercept+search_range), :])+(intercept-search_range)
        new_edge = Seg2Info.interpolate_sky_edge(new_edge, line)
        delta = buffer_px-new_edge
        orig_height, width = img.shape[:2]
        mapx = np.tile(np.arange(width), (orig_height, 1))
        mapy = np.tile(np.arange(orig_height), (width, 1)).T
        translated = cv2.remap(img, mapx.astype(np.float32), (mapy-delta).astype(np.float32),
            interpolation_method, borderValue=self.proc_props["rest_value"])
        new_height = width+buffer_px
        return translated[:new_height,:]

    def horizon_blur(self, img, n_iters=2, blur_height=5, init_blur = (3, 3),
        x_blur=lambda x: (x[0]-x)*50/x[0]+3, y_blur=lambda x: ((x[0]-x)*50/x[0]+3)*0.05+1):
        """Blur the image near the horizon to mitigate low resolution
        :param img: the input image
        :param horizon_height: the number of pixels below the top of the image at which to find the horizon
        :param n_iters: number of iterations
        :param blur_height: the number of pixels below the horizon at which to begin blurring
        :param x_blur: function that takes an array of number of pixels below the horizon\
            and produces an x-axis blur radius
        :param y_blur: function that takes an array of number of pixels below the horizon\
            and produces an x-axis blur radius
        """
        
        horizon_height = self.proc_props["upscale_factor"]*self.proc_props["sky_buffer"]
        for _ in range(n_iters):
            img = cv2.blur(img, (self.proc_props["upscale_factor"]*init_blur[0],
                self.proc_props["upscale_factor"]*init_blur[1]))
            res = img.copy()
            coords = np.arange(blur_height*self.proc_props["upscale_factor"], -1, -1)
            for this_coord, this_x_blur, this_y_blur in zip(coords+horizon_height, x_blur(coords), y_blur(coords)):
                res[:int(this_coord),:] = cv2.blur(img[:int(this_coord),:], (int(this_x_blur), int(this_y_blur)))
            img = res
        return img


    # COORDINATE TRANSFORMATION
    def y2dist(self, y):
        """Convert a y-coordinate (px) in the log-polar plot to a real-space distance (m)"""
        height = self.out_props["height"]
        min_distance = self.out_props["min_distance"]
        max_distance = self.out_props["max_distance"]
        logb = self.proc_props["logb"]
        return self.proc_props["expb"]((height-y)*(logb(max_distance/min_distance)/height)+logb(min_distance))
    
    def dist2y(self, dist):
        """Convert a real-space distance (m) to a y-coordinate (px) in the log-polar plot"""
        height = self.out_props["height"]
        min_distance = self.out_props["min_distance"]
        max_distance = self.out_props["max_distance"]
        logb = self.proc_props["logb"]
        return height-logb(dist/min_distance)/logb(max_distance/min_distance)*height

    def x2t(self, x, width):
        """Convert an x-coordinate (px) in the log-polar plot to a real-space angle (deg)"""
        return x*self.out_props["t_range"]/width-self.out_props["t_range"]/2

    def t2x(self, t):
        """Convert a real-space angle (deg) to an x-coordinate (px) in the log-polar plot"""
        return t*self.out_props["width"]/self.out_props["t_range"]+self.out_props["width"]/2

    def log_polar_to_real_cartesian(self, ts, ls):
        """Convert theta, log-radius coordinates in a log-polar plot to x, y coordinates in real space
        :param tr: the x-coordinate in the log-polar image (px); could be a numpy vector
        :param lr: the y-coordinate in the log-polar image (px); could be a numpy vector
        :param width: the log-polar image width (px)
        :param height: the log-polar image height (px)
        :param t_range: the angle spanned by the x-axis (deg)
        :param min_distance: the distance from the boat at the bottom of the image (m)
        :param max_distance: the distance from the boat at the top of the image (m)
        :return: a tuple of x- and y-coordinates in real space
        """
        
        try: len_tr = len(ts)
        except TypeError: len_tr = None
        try: len_lr = len(ls)
        except TypeError: len_lr = None
        if not len_tr == len_lr: raise ValueError(f"Lengths of tr and lr must be the same; got {len_tr} and {len_lr}")
        
        distance = self.y2dist(ls)
        theta = ts*(self.out_props["t_range"]/self.out_props["width"])-(self.out_props["t_range"]/2)

        xr = np.sin(np.deg2rad(theta))*distance
        yr = np.cos(np.deg2rad(theta))*distance
        return xr, yr

    def real_cartesian_to_camera(self, xr, yr, width, height, scale):
        """Convert x, y coordinates in real space to x, y coordinates in a camera image
        :param xr: the x coordinate in real space (m); could be a numpy vector
        :param yr: the y coordinate in real space (m); could be a numpy vector
        :param width: the width of the camera image (px)
        :param height: the height of the camera image (px)
        :param camera_props: a dictionary of necessary camera properties
        :param scale: the per-image scaling factor from earlier (dimensionless, slightly less than 1)
        :return: a tuple of x- and y-coordinates in camera space
        """

        horizon_distance = self.cam_props["horizon_distance"]
        near_distance = self.cam_props["near_distance"]
        spread = np.tan(np.deg2rad(self.cam_props["horiz_fov"]/scale/2))
        points_real_to_camera = [
            ((-spread*horizon_distance, horizon_distance), (0,     0,    )),
            (( spread*horizon_distance, horizon_distance), (width, 0,    )),
            ((-spread*near_distance,    near_distance   ), (0,     height)),
            (( spread*near_distance,    near_distance   ), (width, height)),
        ]
        src_points, dst_points = zip(*points_real_to_camera)
        mat = cv2.getPerspectiveTransform(np.float32(src_points), np.float32(dst_points))

        # cv2.perspectiveTransform needs a very specific shape
        points = np.stack([xr, yr], -1)[np.newaxis].astype(np.float32)
        scalar_input = (points.ndim == 2)
        if scalar_input: points = points[np.newaxis]
        transformed = cv2.perspectiveTransform(points, mat)
        xc, yc = transformed[0, :, 0], transformed[0, :, 1]
        if scalar_input: xc, yc = xc[0], yc[0]
        return xc, yc

    def camera_to_log_polar(self, img, scale, height, interpolation_method=cv2.INTER_LINEAR, fix_horizon=False):
        """Convert a camera image to a log-polar plot in real space
        :param img: the input image in camera space
        :param scale: amount by which the image has been scaled
        :param height: the original height of the camera image, after any upsampling but before other preprocessing
        :param final_width: the output width
        :param final_height: the output height
        :param t_range: the desired angle range of the output (deg)
        :param min_distance: the desired distance from the boat at the bottom of the image (m)
        :param max_distance: the desired distance from the boat at the top of the image (m)
        :param camera_props: a dictionary of camera properties
        :param fill_value: value to use for pixels in the output for which there is no data
        :param interpolation_method: the OpenCV interpolation method to use
        :param fix_horizon: artificially simulate a disk horizon instead of a line
        """

        final_width = self.out_props["width"]
        final_height = self.out_props["height"]
        camera_width = img.shape[1]

        x_in = np.tile(np.arange(final_width), (final_height, 1)).astype(np.float32).reshape(-1)
        y_in = np.tile(np.arange(final_height), (final_width, 1)).T.astype(np.float32).reshape(-1)
        x_inter, y_inter = self.log_polar_to_real_cartesian(x_in, y_in)
        x_out, y_out = self.real_cartesian_to_camera(x_inter, y_inter, camera_width, height, scale)
        y_out[y_out < 0] = np.nan  # Don't extend beyond the horizon
        x_out = x_out.reshape((final_height, final_width))
        y_out = y_out.reshape((final_height, final_width))
        horizon_height = self.dist2y(self.cam_props["horizon_distance"])
        # Artificially create a circular horizon rather than a line. In practice, the changes due to this are always well
        # under a pixel in the original image (that's why the horizon appears to be a line in the first place), so this is
        # more or less cosmetic and is not manipulation.
        if fix_horizon: y_out[:int(np.ceil(horizon_height)), :] = np.nan
        y_out += self.proc_props["sky_buffer"]*self.proc_props["upscale_factor"]  # Account for the sky buffer
        return cv2.remap(img, x_out, y_out, interpolation_method, borderValue=self.proc_props["rest_value"])

    def four_to_one(self, img, valid_fn=lambda ice, water, rest: (rest < 0.75) & (ice+water > 0), clip_range = [0.1, 0.35]):
        """Take an image that's been operated upon in one_hot_four format and convert to cat2cont (single-channel) format
        :param img: input image
        :param valid_fn: function taking ice, water, and rest masks and outputting a boolean mask of pixels with valid data
        :param clip_range: after conversion, clip the data to this range and rescale to fill [0, 1]
        """

        ice = self.clamp(img[:, :, self.proc_props["four_values"].index("ice")])
        water = self.clamp(img[:, :, self.proc_props["four_values"].index("water")])
        rest = self.clamp(img[:, :, self.proc_props["four_values"].index("sky")]+img[:, :, self.proc_props["four_values"].index("rest")])
        iciness = np.divide(ice, ice+water, where=(ice+water != 0))
        out = np.where(valid_fn(ice, water, rest), iciness, np.nan)
        out = np.clip(out, *clip_range)
        out = (out-clip_range[0])/(clip_range[1]-clip_range[0])
        return out

    def whole_pipeline(self, img, interpolation_method=cv2.INTER_CUBIC):
        """Run the entire seg2info pipeline on one image array, with all interpolation done using interpolation_method"""
        img = self.one_hot_four(img)
        img = self.upscale(img, interpolation_method=interpolation_method)
        img = self.undistort(img, interpolation_method=interpolation_method)
        horizon = self.find_horizon(img)
        img, scale, height = self.rotate_image(img, horizon, interpolation_method=interpolation_method)
        img = self.adjust_and_crop(img, horizon, height, interpolation_method=interpolation_method)
        img = self.horizon_blur(img)
        img = self.camera_to_log_polar(img, scale, height, interpolation_method=interpolation_method)
        img = self.four_to_one(img)
        return img


    # OUTPUT convenience functions
    @staticmethod
    def image2title(image):
        """Get a nice title from an image dictionary"""
        return f"{image['name'][:9]}...{image['name'][-7:]}"

    @staticmethod
    def simple_composite(ax, image, img_key="seginput", map_key="segmap", title=None):
        """Show a composite of an actual image and its segmentation map"""
        if title is None: title = Seg2Info.image2title(image)
        ax.set_title(title)
        ax.imshow(cv2.cvtColor(image[img_key], cv2.COLOR_BGR2RGB))
        ax.imshow(image[map_key], alpha=0.25, cmap="tab20", vmax=4, interpolation="none")

    def plot_mask(self, ax, image, map_key="segmap", title=None):
        """Detect what format a mask is in and plot it accordingly"""
        if title is None: title = self.image2title(image)
        ax.set_title(title)
        img = image[map_key]
        if img.dtype.kind == 'f':  # Continuous
            if img.ndim > 2:  # New encoding: four-channel one-hot-esque
                red = img[:, :, self.proc_props["four_values"].index("sky")]+img[:, :, self.proc_props["four_values"].index("rest")]
                green = img[:, :, self.proc_props["four_values"].index("ice")]
                blue = img[:, :, self.proc_props["four_values"].index("water")]
                viz = np.uint8(np.clip(np.stack([red, green, blue], axis=-1), 0, 1)*255)
                ax.imshow(viz, interpolation="none")
            else:  # Old encoding: [0, 1] water to ice scale and NaNs for everything else
                my_cmap = copy.copy(matplotlib.cm.get_cmap("gray"))
                my_cmap.set_bad(color="red")
                ax.imshow(img, cmap=my_cmap, interpolation="none")
        else:  # Categorical
            ax.imshow(img, cmap="tab20", vmax=4, interpolation="none")

    def plot_line(self, ax, image, map_key="segmap", line_key="line", title=None):
        """Plot a mask with a line on it"""
        width = image[map_key].shape[1]
        intercept,slope = image[line_key]
        self.plot_mask(ax, image, map_key=map_key, title=title)
        ax.plot((0, width), (intercept-slope*width//2, intercept+slope*width//2), "--", color="#FFFF00", linewidth=2.5)

    @staticmethod
    def plot_all(images, plot_fn, adjust_fn=None, cols=3, size=5):
        """Plot a grid of images
        :param images: the list of image dictionaries
        :param plot_fn: the function in axis, image array to call for each image
        :param cols: number of columns in the grid
        :param size: size of each subplot
        :param adjust_fn: optionally specify a function that operates on the whole figure"""

        rows = int(np.ceil(len(images)/cols))
        fig,axs = plt.subplots(rows, cols, figsize=(cols*size, rows*size))
        if adjust_fn is not None: adjust_fn(fig)
        for ax in axs.ravel(): ax.axis("off")
        for ax,image in zip(axs.ravel(), images):
            plot_fn(ax, image)

    def plot_key(self, images, key, adjust_fn=None):
        """Convenience function to plot_all with plot_mask on a given key"""
        self.plot_all(images, lambda ax,img: self.plot_mask(ax, img, map_key=key), adjust_fn)

    def plot_arr(self, arr, names, adjust_fn=None):
        """Convenience function to plot_all with plot_mask but from a list of image arrays"""
        counter = itertools.count()
        self.plot_all(arr, lambda ax,img: self.plot_mask(ax, {"img": img, "name": names[next(counter)]}, map_key="img"), adjust_fn)

    def plot_mat(self, mat, ax=None):
        """Convenience function to plot a single image array"""
        if ax is None: ax = plt.gca()  # Evaluate on invocation!
        self.plot_mask(ax, {"img": mat}, "img", "")

    def log_minor_ticks(self, d0, d1):
        """Generate the minor ticks in a logarithmic axis"""  # rather cleverly, if I do say so myself
        exprange = self.proc_props["expb"](np.arange(int(np.floor(self.proc_props["logb"](d0))), int(np.ceil(self.proc_props["logb"](d1)))+1))
        all_ticks = (np.outer(exprange, np.arange(2, self.proc_props["expb"](1)))).ravel()
        return all_ticks[(d0 <= all_ticks) & (all_ticks <= d1)]

    def plot_log_polar(self, ax, image, map_key="logpolar", title=None, t_step=10):
        """Apply plot_mask to data that is *already* in log-polar coordinates and add appropriate axes"""
        self.plot_mask(ax, image, map_key, title)

        ax.axis("on")
        # It's possible we could use pcolormesh and some more transformation matrices to get matplotlib to do log scales
        # for us… but that would basically be repeating work from above. I'd rather just construct the axes manually:
        min_distance = self.out_props["min_distance"]
        max_distance = self.out_props["max_distance"]
        t_range = self.out_props["t_range"]
        height,_ = image[map_key].shape[:2]
        major_distances = self.proc_props["expb"](np.arange(np.ceil(self.proc_props["logb"](min_distance)), np.floor(self.proc_props["logb"](max_distance))+1)).astype(int)
        ax.set_yticks(self.dist2y(major_distances))
        ax.set_yticklabels(major_distances)
        ax.set_yticks(self.dist2y(self.log_minor_ticks(min_distance, max_distance)), minor=True)
        major_angles = np.arange(0, int(np.floor(t_range/2))+1, t_step)
        major_angles = np.concatenate([-major_angles[::-1], major_angles[1:]])
        ax.set_xticks(self.t2x(major_angles))
        # ax.set_xticklabels(np.vectorize(lambda x: f"{x:.1f}")(major_angles))  # Format floating-point labels
        ax.set_xticklabels(major_angles)

        ax.set_ylabel("Distance (m)")
        ax.set_xlabel("Angle (deg)")


def main():
    """This module is not meant to be run directly"""
    pass

if __name__ == "__main__":
    main()
