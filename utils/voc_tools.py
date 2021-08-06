from bs4 import BeautifulSoup
import copy
import glob
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


def populate_voc(template, outdir, imgptf, bbox, label):
    """
    iterate over all the xml files in the annotations directory, consolidate, and copy
    :param template: xml template to work from
    :param outdir: path to output directory (where new files are copied)
    :param imgptf: where the image lives
    :param bbox: bounding box coordinates [array, each row corresponds to label]
    :param label: labels corresponding to bboxes and segments [list of ind]
    """

    # read the template
    bs_data = read_xml(template)

    # insert folder path
    fold = bs_data.folder
    fold.string = os.path.split(imgptf)[0]

    # insert filename
    fname = bs_data.filename
    fname.string = os.path.split(imgptf)[1]

    # get the whole dimensions
    wd, ht = get_dim(imgptf)
    width = bs_data.size.width
    width.string = str(wd)
    height = bs_data.size.height
    height.string = str(ht)

    # copy the object tag for however many bounding boxes are in the ROI
    flag = 1

    # check that there is more than one labeled region in image
    if len(bbox.shape) > 1:
        while flag < bbox.shape[0]:
            bs_data.annotation.append(copy.copy(bs_data.object))
            flag += 1

    # select all empty elements in the xml document
    nns = bs_data.select('name:empty')  # list of empty name tags
    xmins = bs_data.select('xmin:empty')
    ymins = bs_data.select('ymin:empty')
    xmaxs = bs_data.select('xmax:empty')
    ymaxs = bs_data.select('ymax:empty')

    if len(nns) > 1:
        # bounding box info
        # for now just select the first tuple and hardcode other into [012221]
        for ii in range(bbox.shape[0]):
            # enter the label string
            name = nns[ii]
            name.string = label[ii]

            # enter the corresponding bbox location
            bb = bbox[ii, ::]
            xmin = xmins[ii]
            xmin.string = str(bb[1])
            ymin = ymins[ii]
            ymin.string = str(bb[0])
            xmax = xmaxs[ii]
            xmax.string = str(bb[3])
            ymax = ymaxs[ii]
            ymax.string = str(bb[2])
    else:
        ii = 0
        # enter the label string
        name = nns[ii]
        name.string = label[ii]

        # enter the corresponding bbox location
        xmin = xmins[ii]
        xmin.string = str(bbox[1])
        ymin = ymins[ii]
        ymin.string = str(bbox[0])
        xmax = xmaxs[ii]
        xmax.string = str(bbox[3])
        ymax = ymaxs[ii]
        ymax.string = str(bbox[2])

    outpath = os.path.join(outdir, os.path.splitext(os.path.basename(imgptf))[0]+'.xml')

    # save it
    with open(outpath, 'w') as ff:
        ff.write(str(bs_data))
        ff.close()
