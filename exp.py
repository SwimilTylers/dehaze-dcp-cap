import time
import utils
import dcp as dark_channel_prior
import cap as color_attenuation_prior


def resolve_ires(series_name, method, ires, rgb, gray):
    for f in rgb:
        t = ires[f]
        if t is not None:
            utils.archive(series_name, method+"."+f, t)

    for f in gray:
        t = ires[f]
        if t is not None:
            utils.archive(series_name, method+"."+f, t, is_gray=True)


if __name__ == '__main__':
    image_names = utils.list_images_recursive('img')

    for im_name in image_names:
        print(im_name)
        im = utils.get_color_pic(im_name).astype('float64')
        ires = {}
        tic = time.time()
        im_dcp = dark_channel_prior.remove_haze(im, t_mode="dynamic", ires=ires, refine=False)
        print("dcp finished, t=", time.time()-tic)
        utils.archive(im_name, "dcp", im_dcp)
        resolve_ires(im_name, "dcp", ires, ["raw", "dark"], ["t_raw", "t"])

        tic = time.time()
        im_cap = color_attenuation_prior.remove_haze(im_name, t_mode="dynamic", ires=ires, refine=False)
        print("cap finished, t=", time.time()-tic)
        utils.archive(im_name, "cap", im_cap)
        resolve_ires(im_name, "cap", ires, ["raw"], ["depth", "t_raw", "t"])
