"""
Read label box export and convert to VOC format. For more information regarding LabelBox export format visit:

https://docs.labelbox.com/data-model/en/index-en#label

date: 8/5/21
author: @eor314
"""
import json
import argparse
import os
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
from utils.img_proc import get_dim


def parse_row(obj):
    """
    parse label-box row
    :param obj: dictionary from "Label" entry in labelbox export
    :return: pared down dictionary with bbox info
    """

    tmp = pd.DataFrame(columns=['value', 'ymin', 'xmin', 'ymax', 'xmax'])
    for item in obj['objects']:
        if 'bbox' in item.keys():
            ii = tmp.shape[0]
            tmp.at[ii, 'value'] = item['value']
            tmp.at[ii, 'ymin'] = item['bbox']['top']
            tmp.at[ii, 'xmin'] = item['bbox']['left']
            tmp.at[ii, 'ymax'] = item['bbox']['top'] + item['bbox']['height']
            tmp.at[ii, 'xmax'] = item['bbox']['left'] + item['bbox']['width']

    return tmp


if __name__ == '__main__':

    # define parser
    parser = argparse.ArgumentParser(description='read LabelBox export and convert to VOC')

    parser.add_argument('labelbox', metavar='labelbox', help='Absolute path to label box export [JSON]')
    parser.add_argument('out_dir', metavar='out_dir', help='Where to put output')

    args = parser.parse_args()

    # load the labelbox export as dataframe
    labelbox = pd.read_json(args.labelbox)

    #