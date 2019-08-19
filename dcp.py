import numpy as np
import utils
import time
import advops
import scipy.ndimage.filters as filters


# dark_channel
def get_dark_channel(rgb_channel, ksize, ):
    dark_channel = np.min(rgb_channel, axis=2)
    dark_channel = filters.minimum_filter(dark_channel, ksize)
    return dark_channel


# global atmospheric light
def get_atm_light(background, filter_mat, ratio):
    total_num = filter_mat.size
    tops = int(max(np.floor(total_num * ratio), 1))

    filter_mat = filter_mat.astype('int')

    counts = np.bincount(filter_mat.flatten())
    crt_cum = 0
    for threshold in reversed(range(counts.size)):
        crt_cum += counts[threshold]
        if crt_cum >= tops:
            gal = np.zeros(np.size(background, 2))
            for c in range(np.size(background, 2)):
                buf = background[:, :, c]
                gal[c] = np.mean(buf[filter_mat >= threshold])
            return gal
    return None


def get_trsm_mat(original_im, gal, omega, ksize):
    buffer_im = np.empty(original_im.shape, 'float64')
    if len(gal.shape) > 1:
        gal = np.squeeze(gal)

    for c in range(np.size(original_im, 2)):
        buffer_im[:, :, c] = original_im[:, :, c] / gal[c]

    return 1 - omega * get_dark_channel(buffer_im, ksize)


def get_remove(original_im, gal, transmission, ires=None):
    result_im = np.empty(original_im.shape, 'uint8')

    if ires is None:
        for c in range(np.size(original_im, 2)):
            result_im[:, :, c] = utils.mat_bound(gal[c] + (original_im[:, :, c] - gal[c]) / transmission, 0, 255)
    else:
        ires_color = {}
        color_name = ['r', 'g', 'b']
        for c in range(np.size(original_im, 2)):
            ires_procedure = {'step 1': original_im[:, :, c] - gal[c]}
            ires_procedure['step 2'] = ires_procedure['step 1'] / transmission
            ires_procedure['step 3'] = utils.mat_bound(gal[c] + ires_procedure['step 2'], 0, 255)
            result_im[:, :, c] = ires_procedure['step 3']
            ires_color[color_name[c]] = ires_procedure

        ires['get_remove'] = ires_color

    return result_im.astype('uint8')


def record_intermediate_result(irs, key, value):
    if irs is not None:
        irs[key] = value


def record_progress(verbose, msg, t=None):
    if verbose:
        if t is None:
            print(msg)
        else:
            print(msg+", t=", time.time()-t)


def remove_haze(raw_image, dc_ksize=15, select_ratio=0.001, a_ubound=None,
                omega=0.95, t_lbound=0.1, gf_ksize=(80, 80), t_mode="original", refine=True, verbose=False, ires=None):
    if isinstance(raw_image, str):
        original_im = utils.get_color_pic(raw_image).astype('float64')
    else:
        original_im = raw_image
    record_intermediate_result(ires, 'raw', original_im)
    record_progress(verbose, "loaded")

    tic = time.time()
    dark_channel = get_dark_channel(original_im, ksize=dc_ksize)
    record_progress(verbose, "dark channel", tic)
    record_intermediate_result(ires, 'dark', dark_channel)

    tic = time.time()
    gal = utils.mat_upper_bound(get_atm_light(original_im, dark_channel, ratio=select_ratio), a_ubound)
    record_progress(verbose, "atmosphere light", tic)
    record_intermediate_result(ires, 'gal', gal)

    if t_mode == "original" or (t_mode == "dynamic" and max(original_im.shape) <= 1000):
        tic = time.time()
        trsm_est = utils.mat_lower_bound(get_trsm_mat(original_im, gal, omega, ksize=dc_ksize), t_lbound)
        record_progress(verbose, "estimating transmission", tic)
        record_intermediate_result(ires, 't_raw', trsm_est)

        tic = time.time()
        gray_im = utils.img_color_change(original_im)
        trsm = utils.mat_lower_bound(advops.guided_filter(gray_im, trsm_est, kernel_size=gf_ksize), t_lbound)
        record_progress(verbose, "refining transmission", tic)
        record_intermediate_result(ires, 't', trsm)
    elif t_mode == "fast" or (t_mode == "dynamic" and max(original_im.shape) > 1000):
        smaller_im = utils.im_resize(original_im, 0.25)

        tic = time.time()
        trsm_est = utils.mat_lower_bound(get_trsm_mat(smaller_im, gal, omega, ksize=dc_ksize), t_lbound)
        record_progress(verbose, "estimating transmission", tic)
        record_intermediate_result(ires, 't_raw', trsm_est)

        tic = time.time()
        gray_im = utils.img_color_change(smaller_im)
        trsm = utils.mat_lower_bound(advops.guided_filter(gray_im, trsm_est, kernel_size=gf_ksize), t_lbound)
        record_progress(verbose, "refining transmission", tic)
        record_intermediate_result(ires, 't', trsm)

        trsm = utils.im_resize(trsm, (np.size(original_im, 0), np.size(original_im, 1)))

    else:
        trsm = None

    tic = time.time()
    result = get_remove(original_im, gal, trsm, ires=None)
    record_progress(verbose, "recovering", tic)

    if refine:
        record_progress(verbose, "refining image")
        return advops.refine_image(result)
    else:
        return result


if __name__ == '__main__':
    trace = {}
    img = remove_haze('big1.jpeg', dc_ksize=15, a_ubound=240, ires=None, verbose=True)
    utils.show(img)
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    utils.show(r)
    utils.show(g)
    utils.show(b)
