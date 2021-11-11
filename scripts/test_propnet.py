#!/usr/bin/env python
# FIXME
import sys

sys.path.append("src")

import argparse
import pathlib
import pickle

import dill

from Networks import *
from Test import *


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', '-d', default='Data/DATASET_FIXED_ONLY/9Objects/')
    parser.add_argument('--checkpoint', '-c', default='Models/PN_fixed.hdf5')
    parser.add_argument('--dataset-scaler', '-s', type=pathlib.Path, default='Models/Dataset_Simple_scaler.pickle')

    args = parser.parse_args()

    dill._dill._reverse_typemap['ObjectType'] = object

    with args.dataset_scaler.open('rb') as handle:
        dataset_scaler = pickle.load(handle, encoding="latin1")

    my_dataset9 = MyDataset(path=args.dataset_dir,
                            n_of_scene=5,
                            n_of_exp=1,
                            n_of_obj=9,
                            f_size=8,
                            n_of_rel_type=1,
                            fr_size=240,
                            scaler=dataset_scaler)
    my_dataset9.divideDataset(0.0, 0.0)

    # Network Creation
    Pns = PropagationNetwork()
    # why!?!?
    Pns.getModel(9)
    Pns.setModel(9, args.checkpoint)

    # ### Running Model on Test Set Sparse
    frame_len = 200
    xy_origin_pos, xy_calculated_pos, r, edge = Test(my_dataset9, Pns, frame_len, dataset_scaler.relation_threshold)

    outdir = pathlib.Path('results')
    outdir.mkdir(exist_ok=True)
    make_fixed_video_overlayed(true=xy_origin_pos, pred=xy_calculated_pos, r=r, edge=edge, outdir=outdir)
    make_fixed_video(pos=xy_calculated_pos, r=r, edge=edge, outdir=outdir)


if __name__ == '__main__':
    main()
