import pickle
import re
import sys
import time

import keras.callbacks

from Callbacks import *
from DataGenerator import *
from Networks import *


class DebuggingCheckpoint(keras.callbacks.Callback):
    def on_batch_end(self, *args, **kwargs):
        pass


def main():
    with open('../Models/Dataset_Simple_scaler.pickle', 'rb') as handle:
        if sys.version_info.major == 3:
            dataset_scaler = pickle.load(handle, encoding='latin1')
        else:
            dataset_scaler = pickle.load(handle)

    print(dataset_scaler)

    TEST_PATH = '../Test_Results/Prop-Network-Fixed'

    my_dataset9 = MyDataset(PATH='../Data/DATASET_FIXED_ONLY/9Objects/', n_of_scene=500, n_of_exp=4, n_of_obj=9,
                            f_size=8,
                            n_of_rel_type=1, fr_size=240, scaler=dataset_scaler)
    my_dataset6 = MyDataset(PATH='../Data/DATASET_FIXED_ONLY/6Objects/', n_of_scene=10, n_of_exp=4, n_of_obj=6,
                            f_size=8,
                            n_of_rel_type=1, fr_size=240, scaler=dataset_scaler)
    my_dataset12 = MyDataset(PATH='../Data/DATASET_FIXED_ONLY/12Objects/', n_of_scene=10, n_of_exp=4, n_of_obj=12,
                             f_size=8,
                             n_of_rel_type=1, fr_size=240, scaler=dataset_scaler)
    my_dataset9.divideDataset(9.0 / 11, 1.5 / 11)
    my_dataset6.divideDataset(0.0, 0.5)
    my_dataset12.divideDataset(0.0, 0.5)

    # Network Creation

    Pns = PropagationNetwork()
    Pn1 = Pns.getModel(10, 6, 1)

    Pns.setModel(10, '../Models/PN_fixed.hdf5')
    _ = Pns.getModel(10, 6, 1)

    # Training
    TrainDg9_PN = DataGenerator(n_objects=10, n_of_rel_type=1, number_of_frames=240, num_of_traj=500,
                                dataset=my_dataset9.data_tr,
                                dataRelation=my_dataset9.r_i_tr,
                                relation_threshold=my_dataset9.scaler.relation_threshold,
                                shuffle=True,
                                batch_size=64)
    valDg9_PN = DataGenerator(n_objects=10, n_of_rel_type=1, number_of_frames=240, num_of_traj=200,
                              dataset=my_dataset9.data_val,
                              dataRelation=my_dataset9.r_i_val,
                              relation_threshold=my_dataset9.scaler.relation_threshold,
                              shuffle=False,
                              batch_size=128)

    Pns = PropagationNetwork()
    _ = Pns.getModel(10, 6, 1)

    test_metrics = Test_My_Metrics_Callback(Pns, 3, 1, my_dataset9.scaler, dataset_0=my_dataset9, dataset_1=my_dataset6,
                                            dataset_2=my_dataset12)

    i = -1
    for f in os.listdir(TEST_PATH):
        parts = os.path.split(f)
        m = re.match(r'(\d+)-.*', parts[-1])
        if m:
            i = max(int(m.group(1)), i)
    i += 1
    unique_dir = "{0}-{1}".format(i, int(time.time()))
    outdir = os.path.join(TEST_PATH, unique_dir)
    print(outdir)
    if os.path.exists(outdir):
        print(Exception('This directory already exists'))
    else:
        os.mkdir(outdir)
        os.mkdir(outdir + '/saved_models')

    plt_callback = PlotLosses(outdir + '/Networks_logs.csv', 3, [10, 7, 13])
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8, verbose=1, patience=20, mode='auto',
                                                  cooldown=20)
    save_model = keras.callbacks.ModelCheckpoint(outdir + '/saved_models/weights.{epoch:02d}.hdf5', monitor='val_loss',
                                                 verbose=0, save_best_only=False, save_weights_only=False, mode='auto',
                                                 period=1)
    debugging_ckpt = DebuggingCheckpoint()

    Pn1.fit_generator(generator=TrainDg9_PN,
                      validation_data=valDg9_PN,
                      epochs=3,
                      use_multiprocessing=True,
                      workers=16,
                      callbacks=[debugging_ckpt, reduce_lr, test_metrics, plt_callback, save_model],
                      verbose=1)


if __name__ == '__main__':
    main()
