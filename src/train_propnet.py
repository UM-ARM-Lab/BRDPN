#!/usr/bin/env python
import pickle

import dill
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

from Callbacks import Change_Noise_Callback, Test_My_Metrics_Callback, PlotLosses
from DataGenerator import DataGenerator
from DatasetLoader import MyDataset
from Networks import *

dill._dill._reverse_typemap['ObjectType'] = object

TEST_PATH = '../Test_Results/Prop-Network-Fixed'

with open('../Models/Dataset_Simple_scaler.pickle', 'rb') as handle:
    dataset_scaler = pickle.load(handle, encoding="latin1")

# for debugging
tf.config.run_functions_eagerly(True)

my_dataset9 = MyDataset(PATH='../Data/DATASET_FIXED_ONLY/9Objects/',
                        n_of_scene=150,
                        n_of_exp=4,
                        n_of_obj=9,
                        f_size=8,
                        n_of_rel_type=1,
                        fr_size=240,
                        scaler=dataset_scaler)
my_dataset9.divideDataset(9.0 / 11, 1.5 / 11)

TrainDg9_PN = DataGenerator(n_objects=10, n_of_rel_type=1, num_of_traj=100, number_of_frames=100,
                            dataset=my_dataset9.data_tr,
                            dataRelation=my_dataset9.r_i_tr,
                            relation_threshold=my_dataset9.scaler.relation_threshold,
                            isTrain=True,
                            batch_size=64)

Pns = PropagationNetwork()
Pn1 = Pns.getModel(n_objects=10, object_dim=6, relation_dim=1)

gauss_callback = Change_Noise_Callback(TrainDg9_PN)
test_metrics = Test_My_Metrics_Callback(Pns,
                                        n_of_dataset=1,
                                        n_of_rel=1,
                                        scaler=my_dataset9.scaler,
                                        dataset_0=my_dataset9)

plt_callback = PlotLosses(csv_path=TEST_PATH + '/Networks_logs.csv', n_of_dataset=1, num_of_objects=[10, 7, 13])
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8, verbose=1, patience=20, mode='auto', cooldown=20)
save_model = ModelCheckpoint(TEST_PATH + '/saved_models/weights.{epoch:02d}.hdf5', monitor='val_loss', verbose=0,
                             save_best_only=False, save_weights_only=False, mode='auto', period=1)

Pn1.fit(TrainDg9_PN,
        epochs=250,
        use_multiprocessing=False,
        workers=32,
        callbacks=[reduce_lr, test_metrics, plt_callback, save_model],
        verbose=1)
