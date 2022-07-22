"""
A modular implementation of the routines necessary to get from a semantically segmented camera image to a map in real
space. Prototyped in seg2info_exploration.ipynb and refined in seg2info_exploration_2.ipynb. Highly configurable with
sensible defaults throughout.
"""

import os
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import itertools
import copy

from cv_experiments.labelme2fastai import label2value
import cv_experiments.cv_common as cc

four_values = ["water", "sky", "ice", "rest"]  # The four channels to use in our four-channel representation
rest_value = np.eye(len(four_values))[four_values.index("rest")]  # Representation of "rest" for use as a fill value
upscale_factor = 4  # Factor by which to upscale before doing anything lossy
sky_buffer = 16  # Number of original (pre-upscaled) pixels' buffer of sky to leave when cropping
data_range = [0, 1]  # Range of valid data in our representation

# Various properties of the physical configuration of the camera
camera_props = {
    "camera_height": 5.,
    "horiz_fov": 98.,
    "vert_fov": 54.5,
}
camera_props["horizon_distance"] = 3570.*np.sqrt(camera_props["camera_height"])
camera_props["near_distance"] = np.tan(np.deg2rad(90-camera_props["vert_fov"]))*camera_props["camera_height"]

# Various properties of the desired output
output_props = {
    "width": 500,
    "height": 279,
    "t_range": camera_props["horiz_fov"]*1.1,
    "min_distance": 3,
    "max_distance": 30000
}

# The log and exponential functions to use
logb = np.log10
expb = lambda a: 10**a



# INPUT convenience functions
def load_dirs(seginput_path, segmap_path, segmap_ext = ".png"):
    """Load original images and masks from directories into a dictionary structure"""
    input_exts = (".png", ".jpg", ".jpeg")
    images = []
    for path in os.listdir(seginput_path):
        if not path.endswith(input_exts): continue
        name = os.path.basename(path)
        images.append({"name": name, "seginput": cv2.imread(os.path.join(seginput_path, path)), "segmap":
            cv2.imread(os.path.join(segmap_path, os.path.splitext(name)[0]+segmap_ext), cv2.IMREAD_GRAYSCALE)})
    return images



# PREPROCESSING
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

def clamp(data):
    """Restrict each pixel to the valid range"""
    return np.clip(data, *data_range)

def cat2cont(img, cat_map=label2value):
    """Turn a categorical map into a single-channel map where 0 is water, 1 is ice, and everything else is `NaN`"""
    water_cat, ice_cat = cat_map["water"], cat_map["ice"]
    result = np.where((img != water_cat) & (img != ice_cat), np.nan, img)
    result = np.where(result == water_cat, 0., result)
    result = np.where(result == ice_cat, 1., result)
    return result

def one_hot_ify(img, cat_map=label2value):
    """Turn a categorical map into a one-hot encoded map with one channel per category"""
    return np.stack([img == v for v in cat_map.values()], axis=-1).astype(np.float32)

def one_hot_four(img, cat_map=label2value, four_values=four_values):
    """Turn a categorical map into a one-hot, four-channel map"""
    channels = [((img == cat_map[name]) if name in cat_map else False) for name in four_values]
    stacked = np.stack(np.broadcast_arrays(*channels), axis=-1).astype(np.float32)
    # Insert the "rest" channel where there's a channel name not in cat_map
    rest_i = [name in cat_map for name in four_values].index(False)
    stacked[:,:,rest_i] = 1-stacked.sum(axis=-1)
    return clamp(stacked)

def upscale(img, factor=upscale_factor, interpolation_method=cv2.INTER_LINEAR):
    """Scale an image"""
    return clamp(cv2.resize(img, (int(img.shape[1]*factor), int(img.shape[0]*factor)),
        interpolation=interpolation_method))

def undistort(img, interpolation_method=cv2.INTER_LINEAR):
    """Undistort an image"""
    return clamp(cc.undistort(img, "new", interpolation_method))

def sky_edge(img, sample_locs=np.linspace(0.1, 0.8, 11)):
    """Find the sky edge by searching down each column for the transition from no water/ice to some water/ice.\
        Do this with a bunch of different samples, throw out the bad ones if possible, and average
    """
    oceanness = np.max([img[:,:,four_values.index("water")], img[:,:,four_values.index("ice")]], axis=0)
    # Enforce monotonicity (took me a while to realize the lack of this was causing problems...)
    oceanness = np.maximum.accumulate(oceanness, axis=0)

    samples = np.stack([np.apply_along_axis(np.searchsorted, 0, oceanness, x, side="right")
        for x in sample_locs], axis=-1)
    filtered_samples = np.where(((samples > 0) & (samples < img.shape[0])), samples, np.nan)
    result = np.mean(filtered_samples, axis=-1)
    return np.where(np.isnan(result), np.min(samples, axis=-1), result)

def find_horizon(img):
    """Find a horizon as the line of best fit of the sky_edge"""
    edge = sky_edge(img)
    width = img.shape[1]
    return np.polynomial.Polynomial.fit(np.arange(width)-width//2, edge, 1).convert()

def rotate_image(img, line, interpolation_method=cv2.INTER_LINEAR, fill_value=rest_value):
    """Rotate an image to make the given line horizontal without changing its y-intercept"""
    height,width = img.shape[:2]
    intercept,slope = line
    center = (width//2, intercept)  # Rotate about the midpoint of the horizon line
    angle = np.arctan(slope)
    scale = np.cos(angle)  # Scale down so we don't erase information
    rot_mat = cv2.getRotationMatrix2D(center, angle*180/np.pi, scale)
    # Add some extra height so we don't erase information
    result = cv2.warpAffine(img, rot_mat, (width, height+width),
        flags=interpolation_method, borderValue=fill_value)
    return clamp(result), scale, height

def adjust_and_crop(img, line, height, fill_value=rest_value,
    buffer_px=upscale_factor*sky_buffer, interpolation_method=cv2.INTER_LINEAR):
    """After an image has been rotated so the horizon is roughly horizontal, this translates\
        each column of pixels to make the horizon perfectly flat, then moves the horizon so it is\
        buffer_px away from the top of the image
    """

    intercept,_ = line
    intercept = int(intercept)
    # Search search_range pixels up and down of the intercept for the [ice and water] to sky edge
    search_range = int(0.05*height)
    new_edge = sky_edge(img[(intercept-search_range):(intercept+search_range), :])+(intercept-search_range)
    delta = buffer_px-new_edge
    orig_height, width = img.shape[:2]
    mapx = np.tile(np.arange(width), (orig_height, 1))
    mapy = np.tile(np.arange(orig_height), (width, 1)).T
    translated = cv2.remap(img, mapx.astype(np.float32), (mapy-delta).astype(np.float32),
        interpolation_method, borderValue=fill_value)
    new_height = width+buffer_px
    return translated[:new_height,:]

def horizon_blur(img, horizon_height=upscale_factor*sky_buffer, n_iters=2,
    blur_height=upscale_factor*5, x_blur=lambda x: (x[0]-x)*50/x[0]+3, y_blur=lambda x: ((x[0]-x)*50/x[0]+3)*0.05+1):
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
    
    for _ in range(n_iters):
        img = cv2.blur(img, (upscale_factor*3, upscale_factor*3))
        res = img.copy()
        coords = np.arange(blur_height, -1, -1)
        for this_coord, this_x_blur, this_y_blur in zip(coords+horizon_height, x_blur(coords), y_blur(coords)):
            res[:int(this_coord),:] = cv2.blur(img[:int(this_coord),:], (int(this_x_blur), int(this_y_blur)))
        img = res
    return img



# COORDINATE TRANSFORMATION
def y2dist(y, height, min_distance, max_distance):
    """Convert a y-coordinate (px) in the log-polar plot to a real-space distance (m)"""
    return expb((height-y)*(logb(max_distance/min_distance)/height)+logb(min_distance))
def dist2y(dist, height, min_distance, max_distance):
    """Convert a real-space distance (m) to a y-coordinate (px) in the log-polar plot"""
    return height-logb(dist/min_distance)/logb(max_distance/min_distance)*height

def x2t(x, width, t_range):
    """Convert an x-coordinate (px) in the log-polar plot to a real-space angle (deg)"""
    return x*t_range/width-t_range/2
def t2x(t, width, t_range):
    """Convert a real-space angle (deg) to an x-coordinate (px) in the log-polar plot"""
    return t*width/t_range+width/2

def log_polar_to_real_cartesian(ts, ls, width, height, t_range, min_distance=1, max_distance=10000):
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
    
    distance = y2dist(ls, height, min_distance, max_distance)
    theta = ts*(t_range/width)-(t_range/2)

    xr = np.sin(np.deg2rad(theta))*distance
    yr = np.cos(np.deg2rad(theta))*distance
    return xr, yr

def real_cartesian_to_camera(xr, yr, width, height, scale, camera_props=camera_props):
    """Convert x, y coordinates in real space to x, y coordinates in a camera image
    :param xr: the x coordinate in real space (m); could be a numpy vector
    :param yr: the y coordinate in real space (m); could be a numpy vector
    :param width: the width of the camera image (px)
    :param height: the height of the camera image (px)
    :param camera_props: a dictionary of necessary camera properties
    :param scale: the per-image scaling factor from earlier (dimensionless, slightly less than 1)
    :return: a tuple of x- and y-coordinates in camera space
    """

    horizon_distance = camera_props["horizon_distance"]
    near_distance = camera_props["near_distance"]
    spread = np.tan(np.deg2rad(camera_props["horiz_fov"]/scale/2))
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

def camera_to_log_polar(img, scale, height, final_width=output_props["width"], final_height=output_props["height"],
    t_range=output_props["t_range"], min_distance=output_props["min_distance"],
    max_distance=output_props["max_distance"], camera_props=camera_props, fill_value=rest_value,
    interpolation_method=cv2.INTER_LINEAR, fix_horizon=False):
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

    camera_width = img.shape[1]
    x_in = np.tile(np.arange(final_width), (final_height, 1)).astype(np.float32).reshape(-1)
    y_in = np.tile(np.arange(final_height), (final_width, 1)).T.astype(np.float32).reshape(-1)
    x_inter, y_inter = log_polar_to_real_cartesian(x_in, y_in, final_width, final_height, t_range,
        min_distance, max_distance)
    x_out, y_out = real_cartesian_to_camera(x_inter, y_inter, camera_width, height, scale, camera_props)
    y_out[y_out < 0] = np.nan  # Don't extend beyond the horizon
    x_out = x_out.reshape((final_height, final_width))
    y_out = y_out.reshape((final_height, final_width))
    horizon_height = dist2y(camera_props["horizon_distance"], final_height, min_distance, max_distance)
    # Artificially create a circular horizon rather than a line. In practice, the changes due to this are always well
    # under a pixel in the original image (that's why the horizon appears to be a line in the first place), so this is
    # more or less cosmetic and is not manipulation.
    if fix_horizon: y_out[:int(np.ceil(horizon_height)), :] = np.nan
    y_out += sky_buffer*upscale_factor  # Account for the sky buffer
    return cv2.remap(img, x_out, y_out, interpolation_method, borderValue=fill_value)

def four_to_one(img, valid_fn=lambda ice, water, rest: (rest < 0.75) & (ice+water > 0), clip_range = [0.1, 0.35]):
    """Take an image that's been operated upon in one_hot_four format and convert to cat2cont (single-channel) format
    :param img: input image
    :param valid_fn: function taking ice, water, and rest masks and outputting a boolean mask of pixels with valid data
    :param clip_range: after conversion, clip the data to this range and rescale to fill [0, 1]
    """

    ice = clamp(img[:,:,four_values.index("ice")])
    water = clamp(img[:,:,four_values.index("water")])
    rest = clamp(img[:,:,four_values.index("sky")]+img[:,:,four_values.index("rest")])
    iciness = np.divide(ice, ice+water, where=(ice+water != 0))
    out = np.where(valid_fn(ice, water, rest), iciness, np.nan)
    out = np.clip(out, *clip_range)
    out = out-clip_range[0]/(clip_range[1]-clip_range[0])
    return out

def whole_pipeline(img, interpolation_method=cv2.INTER_CUBIC):
    """Run the entire seg2info pipeline on one image array, with all interpolation done using interpolation_method"""
    img = one_hot_four(img)
    img = upscale(img, interpolation_method=interpolation_method)
    img = undistort(img, interpolation_method=interpolation_method)
    horizon = find_horizon(img)
    img, scale, height = rotate_image(img, horizon, interpolation_method=interpolation_method)
    img = adjust_and_crop(img, horizon, height, interpolation_method=interpolation_method)
    img = horizon_blur(img)
    img = camera_to_log_polar(img, scale, height, interpolation_method=interpolation_method)
    img = four_to_one(img)
    return img

# OUTPUT convenience functions
def image2title(image):
    """Get a nice title from an image dictionary"""
    return f"{image['name'][:9]}...{image['name'][-7:]}"

def simple_composite(ax, image, img_key="seginput", map_key="segmap", title=None):
    """Show a composite of an actual image and its segmentation map"""
    if title is None: title = image2title(image)
    ax.set_title(title)
    ax.imshow(cv2.cvtColor(image[img_key], cv2.COLOR_BGR2RGB))
    ax.imshow(image[map_key], alpha=0.25, cmap="tab20", vmax=4, interpolation="none")

def plot_mask(ax, image, map_key="segmap", title=None, cat_map=label2value):
    """Detect what format a mask is in and plot it accordingly"""
    if title is None: title = image2title(image)
    ax.set_title(title)
    img = image[map_key]
    if img.dtype.kind == 'f':  # Continuous
        if img.ndim > 2:  # New encoding: four-channel one-hot-esque
            red = img[:,:,four_values.index("sky")]+img[:,:,four_values.index("rest")]
            green = img[:,:,four_values.index("ice")]
            blue = img[:,:,four_values.index("water")]
            viz = np.uint8(np.clip(np.stack([red, green, blue], axis=-1), 0, 1)*255)
            ax.imshow(viz, interpolation="none")
        else:  # Old encoding: [0, 1] water to ice scale and NaNs for everything else
            my_cmap = copy.copy(matplotlib.cm.get_cmap("gray"))
            my_cmap.set_bad(color="red")
            ax.imshow(img, cmap=my_cmap, interpolation="none")
    else:  # Categorical
        ax.imshow(img, cmap="tab20", vmax=4, interpolation="none")

def plot_line(ax, image, map_key="segmap", line_key="line"):
    """Plot a mask with a line on it"""
    width = image[map_key].shape[1]
    intercept,slope = image[line_key]
    plot_mask(ax, image, map_key=map_key)
    ax.plot((0, width), (intercept-slope*width//2, intercept+slope*width//2), "--", color="pink", linewidth=2.5)

def plot_all(images, plot_fn, adjust_fn=None, cols=3, size=5):
    """Plot a grid of images
    :param images: the list of image dictionaries
    :param plot_fn: the function in axis, image array to call for each image
    :param cols: number of columns in the grid
    :param size: size of each subplot
    :param adjust_fn: optionally specify a function that operates on the whole figure"""

    rows = int(np.ceil(len(images)/cols))
    fig,axs = plt.subplots(rows, cols, figsize=(rows*size, cols*size))
    if adjust_fn is not None: adjust_fn(fig)
    for ax in axs.ravel(): ax.axis("off")
    for ax,image in zip(axs.ravel(), images):
        plot_fn(ax, image)

def plot_key(images, key, adjust_fn=None):
    """Convenience function to plot_all with plot_mask on a given key"""
    plot_all(images, lambda ax,img: plot_mask(ax, img, map_key=key), adjust_fn)

def plot_arr(arr, names, adjust_fn=None):
    """Convenience function to plot_all with plot_mask but from a list of image arrays"""
    counter = itertools.count()
    plot_all(arr, lambda ax,img: plot_mask(ax, {"img": img, "name": names[next(counter)]}, map_key="img"), adjust_fn)

def plot_mat(mat, ax=plt.gca()):
    """Convenience function to plot a single image array"""
    plot_mask(ax, {"img": mat}, "img", "")

def log_minor_ticks(d0, d1):
    """Generate the minor ticks in a logarithmic axis"""  # rather cleverly, if I do say so myself
    exprange = expb(np.arange(int(np.floor(logb(d0))), int(np.ceil(logb(d1)))+1))
    all_ticks = (np.outer(exprange, np.arange(2, expb(1)))).ravel()
    return all_ticks[(d0 <= all_ticks) & (all_ticks <= d1)]

def plot_log_polar(ax, image, map_key="logpolar", min_distance=output_props["min_distance"],
    max_distance=output_props["max_distance"], t_range=output_props["t_range"], t_step=10):
    """Apply plot_mask to data that is *already* in log-polar coordinates and add appropriate axes"""
    plot_mask(ax, image, map_key)

    ax.axis("on")
    # It's possible we could use pcolormesh and some more transformation matrices to get matplotlib to do log scales
    # for us… but that would basically be repeating work from above. I'd rather just construct the axes manually:
    height, width = image[map_key].shape[:2]
    major_distances = expb(np.arange(np.ceil(logb(min_distance)), np.floor(logb(max_distance))+1)).astype(int)
    ax.set_yticks(dist2y(major_distances, height, min_distance, max_distance))
    ax.set_yticklabels(major_distances)
    ax.set_yticks(dist2y(log_minor_ticks(min_distance, max_distance), height, min_distance, max_distance), minor=True)
    major_angles = np.arange(0, int(np.floor(t_range/2))+1, t_step)
    major_angles = np.concatenate([-major_angles[::-1], major_angles[1:]])
    ax.set_xticks(t2x(major_angles, width, t_range))
    # ax.set_xticklabels(np.vectorize(lambda x: f"{x:.1f}")(major_angles))  # Format floating-point labels
    ax.set_xticklabels(major_angles)

    ax.set_ylabel("Distance (m)")
    ax.set_xlabel("Angle (deg)")



def main():
    """This module is not meant to be run directly"""
    pass

if __name__ == "__main__":
    main()
