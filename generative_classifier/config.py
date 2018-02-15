cfg = {'delta_t': 1e-1,
       'time_start': 0,
       'time_end': 10,
       'num_y': 6,
       'num_layers': 5,
       'gamma': 0.01,
       'zeta': 0.9,
       'eps': 1e-8,
       'lr': 0.1,  # 0.2 for DR_RNN_1, 0.1 for DR_RNN_2 and 3, ??? for DR_RNN_4,
       'num_epochs': 15 * 10,
       'batch_size': 16,
       'data_fn': './data/3dof_sys_l.mat',  # './data/problem1.npz'
       }
