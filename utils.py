import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from skimage import color, transform


def show(a, cmap=None):
    plt.figure()
    if cmap is not None:
        plt.imshow(a, cmap=cmap)
    else:
        plt.imshow(a)
    plt.axis('off')
    plt.show()


def mat_upper_bound(k, b):
    if b is None:
        return k
    else:
        return np.minimum(k, b)
        # return np.where(k < b, k, np.ones(k.shape) * b)


def mat_lower_bound(k, b):
    if b is None:
        return k
    else:
        return np.maximum(k, b)
        # return np.where(k > b, k, np.ones(k.shape) * b)


def mat_bound(k, lbound, ubound):
    return mat_upper_bound(mat_lower_bound(k, lbound), ubound)


def get_color_pic(file_name):
    return np.asarray(Image.open(file_name))


def get_gray_pic(file_name):
    img = Image.open(file_name)
    img = img.convert("L")
    return np.asarray(img)


def img_color_change(img, mode="rgb2gray"):
    if mode == "rgb2gray":
        return color.rgb2gray(img)
    elif mode == "gray2rgb":
        return color.gray2rgb(img)


def im_resize(im, new_shape):
    if not isinstance(new_shape, tuple):
        h, w = np.size(im, 0), np.size(im, 1)
        h, w = int(np.ceil(h * new_shape)), int(np.ceil(w * new_shape))
    else:
        h, w = new_shape

    return transform.resize(im, (h, w))


def archive(series_name, image_name, arc_image, is_gray=False):
    if not os.path.exists("./result/" + series_name):
        os.makedirs("./result/" + series_name)

    if is_gray:
        plt.imsave("./result/" + series_name + "/" + image_name + ".jpg", arc_image, cmap="gray")
    else:
        img = Image.fromarray(arc_image.astype('uint8')).convert('RGB')
        img.save("./result/" + series_name + "/" + image_name + ".jpg", quality=95)


def list_images_recursive(root):
    file_list = []

    if os.path.exists(root):
        this_dir = os.listdir(root)
        if this_dir is None:
            return None
        else:
            for fn in this_dir:
                fn = os.path.join(root, fn)
                if os.path.isdir(fn):
                    t = list_images_recursive(fn)
                    if t is not None:
                        file_list.append(t)
                else:
                    file_list.append(fn)
            return file_list
    else:
        return None


if __name__ == '__main__':
    image = get_color_pic('pic1.png')
    image = img_color_change(img_color_change(image), "gray2rgb")
    show(image)
