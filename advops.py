import numpy as np
import skimage.morphology as kernel
import skimage.filters as filters


def box_filter(src, kernel_size: tuple):
    im_h, im_w = src.shape
    kn_h, kn_w = kernel_size

    dst = np.zeros(src.shape)

    cum = np.cumsum(src, axis=0)
    dst[:kn_h+1, :] = cum[kn_h:2*kn_h+1, :]
    dst[kn_h+1:im_h-kn_h, :] = cum[2*kn_h+1:, :] - cum[:im_h-2*kn_h-1, :]
    dst[im_h-kn_h:, :] = np.repeat(cum[im_h-1, :].reshape(1, -1), kn_w, axis=0) - cum[im_h-2*kn_h-1:im_h-kn_h-1, :]

    cum = np.cumsum(dst, 1)
    dst[:, :kn_w+1] = cum[:, kn_w:2*kn_w+1]
    dst[:, kn_w+1:im_w-kn_w] = cum[:, 2*kn_w+1:] - cum[:, :im_w-2*kn_w-1]
    dst[:, im_w-kn_w:] = np.repeat(cum[:, im_w-1].reshape(-1, 1), kn_w, axis=1) - cum[:, im_w-2*kn_w-1:im_w-kn_w-1]

    return dst


def guided_filter(guidance_img, filter_img, kernel_size, epsilon=1e-6):
    if np.max(guidance_img) > 1:
        guidance_img = guidance_img.astype('float') / 255
    if np.max(filter_img) > 1:
        filter_img = filter_img.astype('float') / 255

    h, w = guidance_img.shape
    local_patch_size = box_filter(np.ones((h, w)), kernel_size)

    mean_g = box_filter(guidance_img, kernel_size) / local_patch_size
    mean_f = box_filter(filter_img, kernel_size) / local_patch_size
    mean_gf = box_filter(guidance_img * filter_img, kernel_size) / local_patch_size
    cov_gf = mean_gf - mean_g * mean_f

    mean_g2 = box_filter(guidance_img * guidance_img, kernel_size) / local_patch_size
    var_g = mean_g2 - mean_g * mean_g

    a = cov_gf / (var_g + epsilon)
    b = mean_f - a * mean_g

    mean_a = box_filter(a, kernel_size) / local_patch_size
    mean_b = box_filter(b, kernel_size) / local_patch_size

    return mean_a * guidance_img + mean_b


def refine_image(img, max_size=2000):
    h, w = np.size(img, 0), np.size(img, 1)
    result = np.empty(img.shape, 'uint8')
    if max(h, w) > max_size:
        result = img
    else:
        for c in range(np.size(img, 2)):
            result[:, :, c] = filters.rank.autolevel(img[:, :, c], kernel.rectangle(h, w))
    return result
