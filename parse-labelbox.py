"""
Read label box export and convert to VOC format. For more information regarding LabelBox export format visit:

https://docs.labelbox.com/data-model/en/index-en#label

date: 8/5/21
author: @eor314
"""
import argparse
import os
import glob
import pandas as pd
import progressbar
from tqdm import tqdm
import requests
import shutil
from utils.voc_tools import populate_voc


def parse_row(obj):
    """
    parse label-box row
    :param obj: dictionary from "Label" entry in labelbox export
    :return: pared down dictionary with bbox info
    """

    tmp = pd.DataFrame(columns=['value', 'ymin', 'xmin', 'ymax', 'xmax'])
    try:
        for item in obj['objects']:
            if 'bbox' in item.keys():
                ii = tmp.shape[0]
                tmp.at[ii, 'value'] = item['value']
                tmp.at[ii, 'ymin'] = item['bbox']['top']
                tmp.at[ii, 'xmin'] = item['bbox']['left']
                tmp.at[ii, 'ymax'] = item['bbox']['top'] + item['bbox']['height']
                tmp.at[ii, 'xmax'] = item['bbox']['left'] + item['bbox']['width']

        return tmp.to_json()
    except KeyError:
        return None


def download_image(url: str, path: str) -> bool:
    """Download an image by URL and save to a specified path."""
    res = requests.get(url, stream=True)
    if res.status_code == 200:
        with open(path, 'wb') as f:
            for chunk in res.iter_content(1024):
                f.write(chunk)
        return True
    return False


if __name__ == '__main__':

    # define parser
    parser = argparse.ArgumentParser(description='read LabelBox export and convert to VOC')

    parser.add_argument('labelbox', metavar='labelbox', help='Absolute path to label box export [JSON]')
    parser.add_argument('out_dir', metavar='out_dir', help='Where to put output')
    parser.add_argument('--xml_temp', metavar='xml_temp', default='voc_template.xml',
                        help='path to xml template for VOC [default: data-wrangling/voc_template.xml]')

    args = parser.parse_args()

    # load the labelbox export as dataframe
    labelbox = pd.read_json(args.labelbox)
    out_dir = args.out_dir
    if args.xml_temp == 'voc_template.xml':
        # if arg is default value grab the template in subdir
        xml_template = os.path.join(os.getcwd(), 'data-wrangling', args.xml_temp)
    else:
        xml_template = args.xml_temp

    # make image dir
    img_dir = os.path.join(out_dir, 'images')
    if not os.path.exists(img_dir):
        os.mkdir(img_dir)

    img_base = [os.path.basename(line) for line in glob.glob(os.path.join(img_dir,'*.jpg'))]

    # make annotation dir
    ann_dir = os.path.join(out_dir, 'annotations')
    if not os.path.exists(ann_dir):
        os.mkdir(ann_dir)
    else:
        shutil.rmtree(ann_dir)
        os.mkdir(ann_dir)

    # remove questionable annotations
    labelbox = labelbox[(labelbox['Created By'] != 'akhen@ucsd.edu') | labelbox.Reviews.astype(bool)]

    # remove skipped images
    labelbox = labelbox[labelbox['Skipped'] == 0]

    print('unpacking annotations...')
    tqdm().pandas()  # for progress bar
    labelbox['processed'] = labelbox.progress_apply(lambda x: parse_row(x['Label']), axis=1)
    tqdm().close()

    probs = []
    # write VOC annotations
    for ii in progressbar.progressbar(labelbox.index):

        # check that the image and annotations exist before continuing
        if labelbox['processed'][ii]:

            if not labelbox['External ID'][ii] in img_base:

                probs.append(labelbox['Labeled Data'][ii])
                # retrieve image
                try:
                    dl_pass = download_image(labelbox['Labeled Data'][ii], os.path.join(img_dir, labelbox['External ID'][ii]))

                    if dl_pass:
                        # make the annotation xml
                        populate_voc(xml_template, out_dir, labelbox['External ID'][ii], labelbox['processed'][ii])
                    else:
                        probs.append(labelbox['Labeled Data'][ii])
                except ConnectionError:
                    probs.append(labelbox['Labeled Data'][ii])
            else:
                populate_voc(xml_template, out_dir, labelbox['External ID'][ii], labelbox['processed'][ii])

    print('Done generating VOC, failed: ', len(probs))
    with open(os.path.join(out_dir, 'probs.txt'), 'w') as ff:
        for line in probs:
            ff.write(line+'\n')
