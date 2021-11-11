import pickle

from Callbacks import *
from DataGenerator import *
from Networks import *

with open('../Models/Dataset_Simple_scaler.pickle', 'rb') as handle:
    dataset_scaler = pickle.load(handle)

TEST_PATH = '../Test_Results/Prop-Network-Fixed'
if os.path.exists(TEST_PATH):
    print(Exception('This directory already exists'))
else:
    os.mkdir(TEST_PATH)
    os.mkdir(TEST_PATH + '/saved_models')

my_dataset9 = MyDataset(PATH='../Data/DATASET_FIXED_ONLY/9Objects/', n_of_scene=1100, n_of_exp=4, n_of_obj=9, f_size=8,
                        n_of_rel_type=1, fr_size=240, scaler=dataset_scaler)
my_dataset6 = MyDataset(PATH='../Data/DATASET_FIXED_ONLY/6Objects/', n_of_scene=50, n_of_exp=4, n_of_obj=6, f_size=8,
                        n_of_rel_type=1, fr_size=240, scaler=dataset_scaler)
my_dataset12 = MyDataset(PATH='../Data/DATASET_FIXED_ONLY/12Objects/', n_of_scene=50, n_of_exp=4, n_of_obj=12, f_size=8,
                         n_of_rel_type=1, fr_size=240, scaler=dataset_scaler)
my_dataset9.divideDataset(9.0 / 11, 1.5 / 11)
my_dataset6.divideDataset(0.0, 0.5)
my_dataset12.divideDataset(0.0, 0.5)

my_dataset_test6 = MyDataset2(PATH='../Data/DATASET_TEST_Simple/6Objects/', n_of_scene=50, n_of_exp=4, n_of_groups=7,
                              n_of_obj=6, f_size=5, n_of_rel_type=1, fr_size=50, scaler=dataset_scaler)
my_dataset_test6.divideDataset(0, 0)
my_dataset_test8 = MyDataset2(PATH='../Data/DATASET_TEST_Simple/8Objects/', n_of_scene=50, n_of_exp=4, n_of_groups=10,
                              n_of_obj=8, f_size=5, n_of_rel_type=1, fr_size=50, scaler=dataset_scaler)
my_dataset_test8.divideDataset(0, 0)
my_dataset_test9 = MyDataset2(PATH='../Data/DATASET_TEST_Simple/9Objects/', n_of_scene=50, n_of_exp=4, n_of_groups=12,
                              n_of_obj=9, f_size=5, n_of_rel_type=1, fr_size=50, scaler=dataset_scaler)
my_dataset_test9.divideDataset(0, 0)

# ### Network Creation

Pns = PropagationNetwork()
Pn1 = Pns.getModel(10, 6, 1)

Pns.setModel(10, '../Models/PN_fixed.hdf5')
_ = Pns.getModel(10, 6, 1)

# ## Training
TrainDg9_PN = DataGenerator(10, 1, 240, 3600, my_dataset9.data_tr, my_dataset9.r_i_tr,
                            my_dataset9.scaler.relation_threshold, True, 64)
valDg9_PN = DataGenerator(10, 1, 240, 600, my_dataset9.data_val, my_dataset9.r_i_val,
                          my_dataset9.scaler.relation_threshold, False, 128)
testDg_PN = DataGenerator(10, 1, 240, 200, my_dataset9.data_test, my_dataset9.r_i_test,
                          my_dataset9.scaler.relation_threshold, False, 100, False)

Pns = PropagationNetwork()
_ = Pns.getModel(10, 6, 1)

gauss_callback = Change_Noise_Callback(TrainDg9_PN)
test_metrics = Test_My_Metrics_Callback(Pns, 3, 1, my_dataset9.scaler, dataset_0=my_dataset9, dataset_1=my_dataset6,
                                        dataset_2=my_dataset12)
plt_callback = PlotLosses(TEST_PATH + '/Networks_logs.csv', 3, [10, 7, 13])
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.8, verbose=1, patience=20, mode='auto',
                                              cooldown=20)
save_model = keras.callbacks.ModelCheckpoint(TEST_PATH + '/saved_models/weights.{epoch:02d}.hdf5', monitor='val_loss',
                                             verbose=0, save_best_only=False, save_weights_only=False, mode='auto',
                                             period=1)

Pn1.fit_generator(generator=TrainDg9_PN,
                  validation_data=valDg9_PN,
                  epochs=50,
                  use_multiprocessing=True,
                  workers=32,
                  callbacks=[reduce_lr, test_metrics, plt_callback, save_model],
                  verbose=1)
