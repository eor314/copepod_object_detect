from bs4 import BeautifulSoup
import copy
import glob
import pandas as pd
import os
from utils.img_proc import get_dim


def read_xml(xmlptf):
    """
    read in an xml file and return BeautifulSoup object
    :param xmlptf: absolute path to xml fine
    :return out: bs4 object
    """

    # read document to workspace
    with open(xmlptf, 'r') as ff:
        temp = ff.read()
        ff.close()

    # make it into a soup object
    out = BeautifulSoup(temp, 'xml')

    return out


def populate_voc(template, outdir, img_base, bbox_obj):
    """
    iterate over all the xml files in the annotations directory, consolidate, and copy
    :param template: path to xml template to work from [str]
    :param outdir: path to output directory (VOC parent) [str]
    :param img_base: image basename [str]
    :param bbox_obj: bounding box coordinates and label [json]
    """

    # read the template
    bs_data = read_xml(template)

    # insert folder path
    fold = bs_data.folder
    fold.string = os.path.basename(outdir)

    # insert filename
    fname = bs_data.filename
    fname.string = img_base

    # get the full filepath
    imgptf = os.path.join(outdir, 'images', img_base)

    # get the whole dimensions
    wd, ht = get_dim(imgptf)
    width = bs_data.size.width
    width.string = str(wd)
    height = bs_data.size.height
    height.string = str(ht)

    # read in the bbox and class info
    tmp = pd.read_json(bbox_obj)

    # copy the object tag for however many bounding boxes are in the ROI
    flag = 1

    # check that there is more than one labeled region in image
    while flag < tmp.shape[0]:
        bs_data.annotation.append(copy.copy(bs_data.object))
        flag += 1

    # select all empty elements in the xml document
    nns = bs_data.select('name:empty')  # list of empty name tags
    xmins = bs_data.select('xmin:empty')
    ymins = bs_data.select('ymin:empty')
    xmaxs = bs_data.select('xmax:empty')
    ymaxs = bs_data.select('ymax:empty')

    for ii in range(tmp.shape[0]):
        # enter the label string
        name = nns[ii]
        name.string = tmp['value'][ii]

        # enter the corresponding bbox location
        xmin = xmins[ii]
        xmin.string = str(tmp['xmin'][ii])
        ymin = ymins[ii]
        ymin.string = str(tmp['ymin'][ii])
        xmax = xmaxs[ii]
        xmax.string = str(tmp['xmax'][ii])
        ymax = ymaxs[ii]
        ymax.string = str(tmp['ymax'][ii])

    outpath = os.path.join(outdir, 'annotations', os.path.splitext(os.path.basename(imgptf))[0]+'.xml')

    # save it
    with open(outpath, 'w') as ff:
        ff.write(str(bs_data))
        ff.close()
