#!/usr/bin/env python
import argparse
import pathlib
import pickle
import re
from time import time

import dill
from tensorflow.keras.callbacks import ModelCheckpoint

from Callbacks import Test_My_Metrics_Callback, PlotLosses
from DataGenerator import DataGenerator
from DatasetLoader import MyDataset
from Networks import *


def main():
    parser = argparse.ArgumentParser()

    args = parser.parse_args()

    dill._dill._reverse_typemap['ObjectType'] = object

    root = pathlib.Path('../Test_Results/Prop-Network-Fixed')
    i = -1
    for f in root.iterdir():
        if f.is_dir():
            m = re.match(r'(\d+)-.*', f.name)
            if m:
                i = max(int(m.group(1)), i)
    i += 1
    unique_dir = f'{i}-{int(time())}'
    outdir = root / unique_dir
    outdir.mkdir(parents=True)

    with open('../Models/Dataset_Simple_scaler.pickle', 'rb') as handle:
        dataset_scaler = pickle.load(handle, encoding="latin1")

    # for debugging
    tf.config.run_functions_eagerly(True)

    my_dataset9 = MyDataset(path='../Data/DATASET_FIXED_ONLY/9Objects/',
                            n_of_scene=1100,
                            n_of_exp=4,
                            n_of_obj=9,
                            f_size=8,
                            n_of_rel_type=1,
                            fr_size=240,
                            scaler=dataset_scaler)
    my_dataset9.divideDataset(9.0 / 11, 1.5 / 11)

    TrainDg9_PN = DataGenerator(n_objects=10, n_of_rel_type=1, num_of_traj=240, number_of_frames=3600,
                                dataset=my_dataset9.data_tr,
                                dataRelation=my_dataset9.r_i_tr,
                                relation_threshold=my_dataset9.scaler.relation_threshold,
                                isTrain=True,
                                batch_size=64)

    Pns = PropagationNetwork()
    Pn1 = Pns.getModel(n_objects=10, object_dim=6, relation_dim=1)

    test_metrics = Test_My_Metrics_Callback(Pns,
                                            n_of_dataset=1,
                                            n_of_rel=1,
                                            scaler=my_dataset9.scaler,
                                            dataset_0=my_dataset9)

    csv_path = outdir / 'Networks_logs.csv'
    ckpt_path = outdir / 'saved_models/weights.{epoch:02d}.hdf5'
    plt_callback = PlotLosses(csv_path.as_posix(), n_of_dataset=1, num_of_objects=[10, 7, 13])
    save_model = ModelCheckpoint(ckpt_path.as_posix(), monitor='val_loss', verbose=0,
                                 save_best_only=False, save_weights_only=False, mode='auto', period=1)

    Pn1.fit(TrainDg9_PN,
            epochs=250,
            use_multiprocessing=False,
            workers=32,
            callbacks=[test_metrics, plt_callback, save_model],
            verbose=1)


if __name__ == '__main__':
    main()
