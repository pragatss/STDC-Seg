#!/usr/bin/python
# -*- encoding: utf-8 -*-
"""Turn the DatasetNinja/Supervisely RUGD export into trainId label pngs.

The export stores each image's ground truth as <split>/ann/<name>.png.json
containing one zlib+base64 encoded bitmap per class instance. Training reads
plain single-channel pngs instead, so decode each annotation once into
<split>/label/<name>.png with pixel values = RUGD_CLASSES index and 255 for
void (any pixel no annotation covers).
"""

import argparse
import base64
import io
import json
import os
import os.path as osp
import zlib
from multiprocessing import Pool

import numpy as np
from PIL import Image

from rugd import RUGD_CLASSES

IGNORE_LB = 255
TITLE_TO_TRAINID = {t: i for i, t in enumerate(RUGD_CLASSES)}


def decode_bitmap(bitmap):
    """Supervisely bitmap -> (bool mask, x origin, y origin)."""
    raw = zlib.decompress(base64.b64decode(bitmap['data']))
    mask = np.array(Image.open(io.BytesIO(raw))) > 0
    if mask.ndim == 3:  # RGBA encoded masks carry the mask in the alpha channel
        mask = mask[:, :, -1]
    x, y = bitmap['origin']
    return mask, x, y


def convert_one(job):
    annpth, outpth = job
    with open(annpth) as f:
        ann = json.load(f)
    H, W = ann['size']['height'], ann['size']['width']
    label = np.full((H, W), IGNORE_LB, dtype=np.uint8)
    unknown = set()
    for obj in ann['objects']:
        title = obj['classTitle']
        if title not in TITLE_TO_TRAINID:
            unknown.add(title)
            continue
        if obj.get('geometryType') != 'bitmap':
            unknown.add(title)
            continue
        mask, x, y = decode_bitmap(obj['bitmap'])
        h, w = mask.shape
        ## clip in case a mask runs past the image border
        h, w = min(h, H - y), min(w, W - x)
        if h <= 0 or w <= 0:
            continue
        window = label[y:y + h, x:x + w]
        window[mask[:h, :w]] = TITLE_TO_TRAINID[title]
    Image.fromarray(label).save(outpth)
    return unknown


def main():
    parse = argparse.ArgumentParser()
    parse.add_argument('--rootpth', type=str, default='./data/rugd')
    parse.add_argument('--splits', type=str, nargs='+', default=['train', 'val', 'test'])
    parse.add_argument('--n_workers', type=int, default=8)
    args = parse.parse_args()

    ## sanity-check the class list against the export's own meta.json
    metapth = osp.join(args.rootpth, 'meta.json')
    if osp.isfile(metapth):
        with open(metapth) as f:
            titles = [c['title'] for c in json.load(f)['classes']]
        assert titles == RUGD_CLASSES, \
            'meta.json classes {} do not match RUGD_CLASSES {}'.format(titles, RUGD_CLASSES)

    for split in args.splits:
        annpth = osp.join(args.rootpth, split, 'ann')
        outdir = osp.join(args.rootpth, split, 'label')
        if not osp.isdir(outdir):
            os.makedirs(outdir)
        jobs = []
        for fn in sorted(os.listdir(annpth)):
            if not fn.endswith('.json'):
                continue
            jobs.append((osp.join(annpth, fn), osp.join(outdir, fn[:-len('.json')])))
        pool = Pool(args.n_workers)
        unknown = set()
        try:
            for i, u in enumerate(pool.imap_unordered(convert_one, jobs, chunksize=16)):
                unknown |= u
                if (i + 1) % 200 == 0:
                    print('{} {}/{}'.format(split, i + 1, len(jobs)))
        finally:
            pool.close()
            pool.join()
        print('{}: wrote {} labels to {}'.format(split, len(jobs), outdir))
        if unknown:
            print('  WARNING skipped unmapped classes: {}'.format(sorted(unknown)))


if __name__ == '__main__':
    main()
