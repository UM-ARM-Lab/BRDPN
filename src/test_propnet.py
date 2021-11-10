#!/usr/bin/env python
import pathlib
import pickle

import dill

from Networks import *
from Test import *

dill._dill._reverse_typemap['ObjectType'] = object

with open('../Models/Dataset_Simple_scaler.pickle', 'rb') as handle:
    dataset_scaler = pickle.load(handle, encoding="latin1")

TEST_PATH = '../Test_Results/Prop-Network-Fixed'

my_dataset9 = MyDataset(PATH='../Data/DATASET_FIXED_ONLY/9Objects/',
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
Pns.setModel(9, '../Models/PN_fixed.hdf5')

# ### Running Model on Test Set Sparse
frame_len = 200
xy_origin_pos, xy_calculated_pos, r, edge = Test(my_dataset9, Pns, frame_len, dataset_scaler.relation_threshold)

outdir = pathlib.Path(TEST_PATH) / 'TestVideos' / '9-Sparse'
make_fixed_video_overlayed(true=xy_origin_pos, pred=xy_calculated_pos, r=r, edge=edge, outdir=outdir)
make_fixed_video(pos=xy_calculated_pos, r=r, edge=edge, outdir=outdir)
