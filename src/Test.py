import copy

import matplotlib.animation as manimation
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

from DatasetLoader import *

color = ['blue', 'red', 'green', 'yellow', 'purple', 'cyan', 'orange', 'black', 'pink']


def Test(dataset, Ins, frame_len, relation_threshold):
    n_objects = dataset.n_of_obj + 1
    num_of_rel_type = dataset.num_of_rel_type
    if num_of_rel_type > 1:
        num_of_rel_type = num_of_rel_type + 1
    n_relations = n_objects * (n_objects - 1)
    In = Ins.getModel(n_objects)
    GroundData = dataset.data[dataset.test_traj_start:]
    n_of_traj = GroundData.shape[0]

    xy_origin_pos = copy.deepcopy(GroundData[:, :frame_len, :, 2:4])
    xy_origin_vel = copy.deepcopy(GroundData[:, :frame_len, :, 4:6])

    dataToModel = np.zeros([n_of_traj, frame_len, n_objects, 6])
    dataToModel[:, 0, :, :] = copy.deepcopy(GroundData[:, 0, :, :])
    dataToModel[:, :, 0, :] = copy.deepcopy(GroundData[:, :frame_len, 0, :])
    dataToModel[:, :, :, :2] = copy.deepcopy(GroundData[:, :frame_len, :, :2])
    dataToModel[:, 0, 1:, 4:6] = 0
    r = dataToModel[:, 0, :, 0]
    dataToModel = dataset.scaler.transform(dataToModel)
    val_receiver_relations = np.zeros((n_of_traj, n_objects, n_relations), dtype=float)
    val_sender_relations = np.zeros((n_of_traj, n_objects, n_relations), dtype=float)
    val_relation_info = np.zeros((n_of_traj, n_relations, num_of_rel_type))
    propagation = np.zeros((n_of_traj, n_objects, 100))  # "h" in the propnet paper, initialized to zeros
    cnt = 0
    for m in range(n_objects):
        for j in range(n_objects):
            if (m != j):
                inzz = np.linalg.norm(dataToModel[:, 0, m, 2:4] - dataToModel[:, 0, j, 2:4],
                                      axis=1) < relation_threshold
                val_receiver_relations[inzz, j, cnt] = 1.0
                val_sender_relations[inzz, m, cnt] = 1.0
                if num_of_rel_type > 1:
                    val_relation_info[:, cnt, 1:] = dataset.r_i[dataset.test_traj_start:, 0, m * n_objects + j, :]
                    val_relation_info[np.sum(val_relation_info[:, cnt, 1:], axis=1) == 0, cnt, 0] = 1
                else:
                    val_relation_info[:, cnt, :] = dataset.r_i[dataset.test_traj_start:, 0, m * n_objects + j, :]
                cnt += 1
    edges = val_relation_info  # [t=200, n_rel=90, n_relations_types=1]
    for i in range(1, frame_len):
        velocities = In.predict({'objects':            dataToModel[:, i - 1, :, :],
                                 'sender_relations':   val_sender_relations,
                                 'receiver_relations': val_receiver_relations, 'relation_info': val_relation_info,
                                 'propagation':        propagation})
        dataToModel[:, i, 1:, 2:4] = dataToModel[:, i - 1, 1:, 2:4]
        dataToModel[:, i, 1:, 4:6] = velocities[:, :, :]
        dataToModel[:, i, 1:, :] = PositionCalculateNext(dataToModel[:, i, 1:, :], dataset.scaler)
        val_receiver_relations = np.zeros((n_of_traj, n_objects, n_relations), dtype=float)
        val_sender_relations = np.zeros((n_of_traj, n_objects, n_relations), dtype=float)
        cnt = 0
        for m in range(n_objects):
            for j in range(n_objects):
                if (m != j):
                    inzz = np.linalg.norm(dataToModel[:, i, m, 2:4] - dataToModel[:, i, j, 2:4],
                                          axis=1) < relation_threshold
                    val_receiver_relations[inzz, j, cnt] = 1.0
                    val_sender_relations[inzz, m, cnt] = 1.0
                    cnt += 1
    pred_xy = dataset.scaler.inv_transform(dataToModel)

    xy_calculated_pos = pred_xy[:, :, :, 2:4]
    xy_calculated_vel = pred_xy[:, :, :, 4:6]
    print('mse-pos:',
          mean_squared_error(xy_calculated_pos[:, :, 1:].reshape(-1, 2), xy_origin_pos[:, :, 1:].reshape(-1, 2)))
    print('mse-vel:',
          mean_squared_error(xy_calculated_vel[:, :, 1:].reshape(-1, 2), xy_origin_vel[:, :, 1:].reshape(-1, 2)))
    return xy_origin_pos, xy_calculated_pos, r, edges


def my_make_video_Fixed(true, pred, r, edge, outdir):
    """
    r is radius [n trajs, n_objects]
    edge is for drawing the ground-truth joints [n trajs, n_rel, 1]
    """

    FFMpegWriter = manimation.writers['ffmpeg']
    metadata = dict(title='Movie Test', artist='Matplotlib',
                    comment='Movie support!')
    writer = FFMpegWriter(fps=15, metadata=metadata)

    fig = plt.figure(figsize=(8, 8))
    ax = plt.gca()
    clear_axes(ax)
    n_trajs, n_frames, n_objects, _ = true.shape
    for traj_idx in range(n_trajs):
        filename = outdir / f'test_traj_{traj_idx}.mp4'
        with writer.saving(fig, filename, n_frames):
            for t in range(n_frames):
                add_edges(true[traj_idx, t], ax, edge[traj_idx])
                add_to_frame(true[traj_idx, t], ax, r[traj_idx], fill=True)
                add_to_frame(pred[traj_idx, t], ax, r[traj_idx], fill=False)

                writer.grab_frame()

                clear_axes(ax)
        print(filename)


def clear_axes(ax):
    ax.cla()
    ax.set_xlim((-1.00, 1.00))
    ax.set_ylim((-1.00, 1.00))
    ax.axis('off')


def add_to_frame(states, ax, r, fill):
    for j in range(states.shape[0]):
        circle = plt.Circle((states[j, 1], states[j, 0]), r[j] / 2, color=color[j % 9], fill=fill, alpha=0.8, linewidth=3)
        ax.add_artist(circle)


def add_edges(states, ax, edge):
    cnt = 0
    for j in range(states.shape[0]):
        for k in range(states.shape[0]):
            if j != k:
                if edge[cnt, 0] == 1:
                    ax.plot([states[j, 1], states[k, 1]], [states[j, 0], states[k, 0]],
                            color='black', linestyle='solid', lw=5)
                cnt = cnt + 1
