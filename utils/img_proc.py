from PIL import Image


def get_dim(img):
    """
    return biggest of image dimensions
    :param img: absolute path to image [str]
    :return: max(width, height) [int]
    """
    im = Image.open(img)
    wid, hgt = im.size
    return [wid, hgt]