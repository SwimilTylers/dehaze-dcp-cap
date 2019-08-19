import skimage.color
import numpy as np
import utils
import advops
import scipy.ndimage.filters
from PIL import Image


def get_depth(hsv, theta0=0.121779, theta1=0.959710, theta2=-0.780245, delta=0.041337):
    depth = theta0 + theta1 * hsv[:, :, 2] + theta2 * hsv[:, :, 1]
    return depth + np.random.normal(0, delta, depth.shape)


def get_atm_light(rgb, depth, ratio):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    r, g, b = r.reshape(-1), g.reshape(-1), b.reshape(-1)
    h, w = depth.shape
    depth = depth.reshape(-1)
    idx = (np.argsort(depth))
    idx = idx[int(np.floor((1- ratio) * h * w)):]
    return list((np.mean(r[idx]), np.mean(g[idx]), np.mean(b[idx])))


def record_intermediate_result(irs, key, value):
    if irs is not None:
        irs[key] = value


def remove_haze(file_name, d_ksize=15, beta=1.0, select_ratio=0.001, a_ubound=None, gf_ksize=(80, 80),
                t_mode="original", refine=True, ires=None):
    rgb = np.asarray(Image.open(file_name))
    record_intermediate_result(ires, "raw", rgb)

    hsv = skimage.color.rgb2hsv(rgb)

    depth = get_depth(hsv)
    depth = scipy.ndimage.filters.minimum_filter(depth, d_ksize)
    record_intermediate_result(ires, "depth", depth)

    gal = utils.mat_upper_bound(get_atm_light(rgb, depth, ratio=select_ratio), a_ubound)
    record_intermediate_result(ires, "gal", gal)

    if t_mode == "original" or (t_mode == "dynamic" and max(rgb.shape) <= 1000):
        trsm = utils.mat_bound(np.exp(-beta * depth), 0.1, 0.9)
        record_intermediate_result(ires, "t_raw", trsm)
        trsm = utils.mat_bound(advops.guided_filter(skimage.color.rgb2gray(rgb), trsm, kernel_size=gf_ksize), 0.1, 0.9)
        record_intermediate_result(ires, "t", trsm)
    elif t_mode == "fast" or (t_mode == "dynamic" and max(rgb.shape) > 1000):
        trsm = utils.im_resize(utils.mat_bound(np.exp(-beta * depth), 0.1, 0.9), 0.25)
        record_intermediate_result(ires, "t_raw", trsm)

        gray_im = utils.im_resize(skimage.color.rgb2gray(rgb), 0.25)
        trsm = utils.mat_bound(advops.guided_filter(gray_im, trsm, kernel_size=gf_ksize), 0.1, 0.9)
        record_intermediate_result(ires, "t", trsm)

        trsm = utils.im_resize(trsm, (np.size(rgb, 0), np.size(rgb, 1)))
    else:
        trsm = None

    result = np.empty(rgb.shape, 'uint8')

    for c in range(np.size(rgb, 2)):
        buf = gal[c] + (rgb[:, :, c] - gal[c]) / trsm
        result[:, :, c] = utils.mat_bound(buf, 0, 255).astype('uint8')

    if refine:
        return advops.refine_image(result)
    else:
        return result


if __name__ == '__main__':
    img = remove_haze('img/pic2.png')
    utils.show(img)
