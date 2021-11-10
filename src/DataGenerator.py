import random

import numpy as np
import tensorflow as tf


class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, n_objects, n_of_rel_type, number_of_frames, num_of_traj, dataset, dataRelation,
                 relation_threshold, isTrain=True, batch_size=100, shuffle=True):
        'Initialization'
        self.n_objects = n_objects
        self.relation_threshold = relation_threshold
        self.batch_size = batch_size
        self.n_of_features = 6
        self.num_of_traj = num_of_traj
        self.n_of_rel_type = n_of_rel_type
        self.n_relations = n_objects * (n_objects - 1)  # number of edges in fully connected graph
        self.shuffle = shuffle
        self.number_of_frames = number_of_frames
        self.currEpoch = 0
        self.data = dataset
        self.dataRelation = dataRelation
        self.indexes = 1 + np.arange(self.number_of_frames - 1)
        for i in range(1, self.num_of_traj):
            self.indexes = np.concatenate(
                [self.indexes, (i * self.number_of_frames + 1 + np.arange(self.number_of_frames - 1))])
        self.std_dev_pos = 0.05 * np.std(self.data[:, :, 2:4])
        self.std_dev_vel = 0.05 * np.std(self.data[:, :, 4:6])
        self.add_gaus = 0.20
        self.propagation = np.zeros((self.batch_size, self.n_objects, 100), dtype=float);
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.num_of_traj * (self.number_of_frames - 1) / self.batch_size) / 2)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch

        data_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        # Find list of IDs
        # Generate data
        X, y = self.__data_generation(data_indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.currEpoch = self.currEpoch + 1
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def changeGauss(self, addGauss):
        self.add_gaus = addGauss

    def __data_generation(self, data_indexes):
        """

        """
        'Generates data containing batch_size samples'  # X : (n_samples, *dim, n_channels)
        # Initialization
        temp_data = self.data[data_indexes, :, :].copy()
        data_x_vel_indexes = [idx - 1 for idx in data_indexes]
        # In here we know velocity of objects on previous timesteps, but not this one. So velocity input it robot hands velocity and last velocities of objects.
        temp_data[:, 1:, 4:6] = self.data[data_x_vel_indexes, 1:, 4:6]
        if self.add_gaus > 0:
            for i in range(self.batch_size):
                for j in range(self.n_objects):
                    if (random.random() < self.add_gaus):
                        temp_data[i, j, 2] = temp_data[i, j, 2] + np.random.normal(0, self.std_dev_pos)
                    if (random.random() < self.add_gaus):
                        temp_data[i, j, 3] = temp_data[i, j, 3] + np.random.normal(0, self.std_dev_pos)
                    if (random.random() < self.add_gaus):
                        temp_data[i, j, 4] = temp_data[i, j, 4] + np.random.normal(0, self.std_dev_vel)
                    if (random.random() < self.add_gaus):
                        temp_data[i, j, 5] = temp_data[i, j, 5] + np.random.normal(0, self.std_dev_vel)

        cnt = 0
        x_receiver_relations = np.zeros((self.batch_size, self.n_objects, self.n_relations), dtype=float);
        x_sender_relations = np.zeros((self.batch_size, self.n_objects, self.n_relations), dtype=float);

        x_relation_info = np.zeros((self.batch_size, self.n_relations, self.n_of_rel_type))
        for i in range(self.n_objects):
            for j in range(self.n_objects):
                if (i != j):

                    inzz = np.linalg.norm(temp_data[:, i, 2:4] - temp_data[:, j, 2:4], axis=1) < self.relation_threshold
                    if self.n_of_rel_type == 1:
                        x_relation_info[:, cnt, :] = self.dataRelation[data_indexes, i * self.n_objects + j, :]
                    else:
                        x_relation_info[:, cnt, 1:] = self.dataRelation[data_indexes, i * self.n_objects + j, :]
                        x_relation_info[np.sum(x_relation_info[:, cnt, 1:self.n_of_rel_type], axis=1) == 0, cnt, 0] = 1
                    x_receiver_relations[inzz, j, cnt] = 1.0
                    x_sender_relations[inzz, i, cnt] = 1.0
                    cnt += 1
        x_object = temp_data
        y = self.data[data_indexes, 1:, 4:6]
        return {'objects':            x_object, 'sender_relations': x_sender_relations, \
                'receiver_relations': x_receiver_relations, 'relation_info': x_relation_info, \
                'propagation':        self.propagation}, {'target': y}
