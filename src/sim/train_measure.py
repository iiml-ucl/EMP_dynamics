"""
Adjusted to fit non-rectangle network training.

"""

import copy
import json
import time

import numpy as np
import torch
from tqdm import tqdm
import os
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import sys
import torch.multiprocessing as mp
sys.path.append('../../..')

import src.models.dnn as dnn
from src.util.load_dataset import recipes
import src.util.util_torch as util_torch
import src.util.util_log as util_log
from src.util.util_torch import save_ckpt, load_ckpt
from src.util.util_dm import EarlyStop


PROJECT_ROOT_PATH = '/scratch/uceelyu/pycharm_sync/deep_measure'

class DeepMeasure:
    def __init__(self,
                 dm_note: str,
                 hpara_recon: dict,
                 hpara_label: dict,
                 hpara_train_dm: dict,
                 main_model_ckpt_file_path: str,
                 main_result_path: str,
                 ) -> None:
        """
        Do the deep-measure on the pre-trained model (with its saved check-points during the training of the main
        model).
        The main model should be pre-trained, and the check-points at each epoch should be saved.

        :param dm_note: (str) The note for this deep-measure training. Will be part of the result folder
        :param hpara_recon: (dict) The dictionary for hyper-parameters of reconstruction network. Vary per
                            network model.
        :param hpara_label: (dict) The dictionary for hyper-parameters of label-decoding network. Vary per
                            network model.
        :param hpara_train_dm: (dict) The dictionary for the hyper-parameters of the training of the deep-measure
                                network.
        :param main_model_ckpt_file_path: (String) The path to the pre-trained model. The path should be directed to a
                                    folder and there should be a folder called 'ckpt_main' and a file
                                    'info_train_main.json' in it, containing the checkpoints of the main model.
        """
        seed = datetime.now().microsecond
        torch.manual_seed(seed)
        self.hpara_recon = hpara_recon
        self.hpara_label = hpara_label
        self.hpara_train_dm = hpara_train_dm

        self.rpm = util_log.ResultPathManager(PROJECT_ROOT_PATH)
        self.rpm.reset_root(os.path.join(main_result_path, f'dm,{dm_note}'))
        self.rpm.add_path('ith_repeat', 'root', f'{hpara_train_dm["timestamp"]}')
        self.rpm.add_path('info_dm.json', 'ith_repeat', 'info_dm.json')
        self.rpm.add_path('summary', 'ith_repeat', 'dm_summary')
        self.rpm.add_path('ckpt_dm', 'ith_repeat', 'ckpt_dm')
        self.rpm.add_path('dm_tape', 'ith_repeat', 'dm_tape')


        # self.result_home_path = os.path.join(main_result_path, f'dm,{dm_note}/{hpara_train_dm["timestamp"]}')
        # os.makedirs(self.result_home_path, exist_ok=True)
        self.main_model_ckpt_file_path = main_model_ckpt_file_path
        # self.dm_info_path = os.path.join(self.result_home_path, 'info_dm.json')
        with open(self.rpm('info_dm.json'), 'w') as f:
            json.dump({
                'dm_note': dm_note,
                'hpara_recon': hpara_recon,
                'hpara_label': hpara_label,
                'hpara_train_dm': hpara_train_dm,
                'main_model_ckpt_file_path': main_model_ckpt_file_path,
                'result_home_path': str(self.rpm('ith_repeat')),
                'seed': seed,

            }, f, indent=4)
        main_info_path = os.path.join(main_result_path, 'info_train_main.json')
        with open(main_info_path, 'r') as f:
            self.main_info = json.load(f)
        network_hpara_main = self.main_info['hpara_main']
        main_model_repre_dict_keys = self.main_info['repre_dict'].keys()
        '''
        keys = [
            'block_0,recon',
            'block_0,label',
            'block_1,recon',
            'block_1,label',
            'block_2,recon',
            'block_2,label',
            ...
        ]
        '''
        self.keys = []

        dm_layer_list = main_model_repre_dict_keys
        for dm_layer_name in dm_layer_list:
            for branch_type in ['recon', 'label']:
                self.keys.append(f'{dm_layer_name},{branch_type}')

        self.network_hpara_dm = []
        for key in self.keys:
            if key.split(',')[1] == 'recon':
                i_network_hpara = copy.copy(self.hpara_recon)
            else:
                i_network_hpara = copy.copy(self.hpara_label)
            if isinstance(network_hpara_main['width'], int):
                if key.split(',')[0] == f'block_{network_hpara_main["depth"] - 1}':
                    i_network_hpara['dim_in'] = network_hpara_main['dim_out']
                else:
                    i_network_hpara['dim_in'] = network_hpara_main['width']
            elif isinstance(network_hpara_main['width'], list):
                temp_width_list = network_hpara_main['width'] + [network_hpara_main['dim_out']]
                i_network_hpara['dim_in'] = temp_width_list[int(key.split(',')[0].split('_')[1])]
            self.network_hpara_dm.append(i_network_hpara)

    def train(self):
        dm_device = util_torch.auto_select_GPU(dwell=1, random_delay=10, candidate_list=[0, 1, 2, 3, 4, 5, 6, 7]) ########################################################################
        # main_model = dnn.TishbyNet(network_hpara=self.main_info['hpara_main'])
        main_model = dnn.DNN(network_hpara=self.main_info['hpara_main'])
        main_model, _, _, _ = load_ckpt(self.main_model_ckpt_file_path, main_model, None, device=dm_device)
        main_model.to(dm_device).eval()
        global_mini_step = int(self.main_model_ckpt_file_path.split('/')[-1].split('.')[0].split('_')[1])
        # summary_folder = os.path.join(self.result_home_path, 'dm_summary')
        # os.makedirs(summary_folder, exist_ok=True)
        writer_path = os.path.join(self.rpm('summary'), f'step_{global_mini_step:012}')
        if os.path.isdir(writer_path):
            print('\nRecord exist, skip!\n')
            return
        writer = SummaryWriter(writer_path)
        self.task = self.main_info['hpara_train']['task']
        self.dataloaders = recipes(self.task,
                                   path_raw_dataset=os.path.join(PROJECT_ROOT_PATH, f'dataset/{self.task.split()[0]}'),
                                   batch_size=self.hpara_train_dm['batch_size'],
                                   dataset_kwargs={'train_size': self.hpara_train_dm['train_size']})
        dm_model_list = [dnn.DNN(network_hpara=i).to(dm_device) for i in self.network_hpara_dm]
        IF_LABEL_list = ['label' in i for i in self.keys]
        optimizer_list = [torch.optim.Adam(params=i.parameters(), lr=self.hpara_train_dm['lr']) for i in dm_model_list]
        loss_fn_list = []
        for IF_LABEL in IF_LABEL_list:
            if IF_LABEL:
                loss_fn_list.append(util_torch.select_loss('cross_entropy')[0])
            else:
                loss_fn_list.append(util_torch.select_loss('mse')[0])
        m_dict_list = []
        for idx, IF_LABEL in enumerate(IF_LABEL_list):
            if IF_LABEL:
                m_dict = {
                    'loss/train': util_log.Measurement(f'{self.keys[idx]}/loss/train'),
                    'loss/val': util_log.Measurement(f'{self.keys[idx]}/loss/train'),
                    'loss/test': util_log.Measurement(f'{self.keys[idx]}/loss/train'),
                    'acc/train': util_log.Measurement(f'{self.keys[idx]}/acc/train'),
                    'acc/val': util_log.Measurement(f'{self.keys[idx]}/acc/train'),
                    'acc/test': util_log.Measurement(f'{self.keys[idx]}/acc/train'),
                }
            else:
                m_dict = {
                    'loss/train': util_log.Measurement(f'{self.keys[idx]}/loss/train'),
                    'loss/val': util_log.Measurement(f'{self.keys[idx]}/loss/train'),
                    'loss/test': util_log.Measurement(f'{self.keys[idx]}/loss/train'),
                }
            m_dict_list.append(m_dict)

        # Load initialize ckpt, if not exist, do not load
        # ckpt_init_path = os.path.join(self.rpm('ith_repeat'), 'ckpt_dm')
        # os.makedirs(ckpt_init_path, exist_ok=True)
        ckpt_init_path_list = [os.path.join(self.rpm('ckpt_dm'), f'{i}.pt') for i in self.keys]
        for idx, ckpt_init_path in enumerate(ckpt_init_path_list):
            if os.path.isfile(ckpt_init_path):
                dm_model_list[idx], optimizer_list[idx], _, _ = \
                    load_ckpt(ckpt_init_path,
                              dm_model_list[idx],
                              optimizer_list[idx], device=None)
                dm_model_list[idx].to(dm_device)
            else:
                save_ckpt(ckpt_init_path, 0, dm_model_list[idx], optimizer_list[idx], 0.)

        early_stop_list = [EarlyStop(name=f'global_step_{global_mini_step}-{key}',
                                     window_size=self.hpara_train_dm['early_stop_window_size']) for key in self.keys]

        # Start training
        tq = tqdm(total=self.hpara_train_dm['max_ep'], dynamic_ncols=True, unit='dm-ep', position=0)
        ep = 0
        while(True):
            if self.hpara_train_dm['max_ep'] != -1 and ep > self.hpara_train_dm['max_ep']:
                break
            ep += 1

        # for ep in range(1, self.hpara_train_dm['max_ep'] + 1):
            for phase in ['val', 'train', 'test']:
                tq.set_description(f'dm,{phase},global_step_{global_mini_step}')
                # Clear measurements
                for idx, m_dict in enumerate(m_dict_list):
                    m_dict[f'loss/{phase}'].clear()
                    if IF_LABEL_list[idx]:
                        m_dict[f'acc/{phase}'].clear()

                for mini_step, batch_sample in enumerate(self.dataloaders[phase]):
                    batch_inputs = batch_sample[0][0].to(dm_device)
                    batch_labels = batch_sample[1][0].to(dm_device)
                    _ = main_model(batch_inputs)

                    for optimizer in optimizer_list:
                        optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        batch_repre_list = [main_model.repre_dict[i.split(',')[0]] for i in self.keys]
                        batch_dm_output_list = [None] * len(self.keys)
                        for idx, (dm_model, batch_repre) in enumerate(zip(dm_model_list, batch_repre_list)):
                            if early_stop_list[idx]():
                                batch_dm_output_list[idx] = dm_model(batch_repre)
                        dm_label_list = [None] * len(self.keys)
                        for idx in range(len(self.keys)):
                            if early_stop_list[idx]():
                                if IF_LABEL_list[idx]:
                                    dm_label_list[idx] = batch_labels
                                else:
                                    dm_label_list[idx] = torch.reshape(batch_inputs, batch_dm_output_list[idx].shape)

                        loss_list = [0.] * len(self.keys)
                        for idx in range(len(self.keys)):
                            if early_stop_list[idx]():
                                loss_list[idx] = loss_fn_list[idx](batch_dm_output_list[idx], dm_label_list[idx])
                        loss_value_list = [float(loss) for loss in loss_list]

                        for idx, (early_stop, m_dict, loss_value, key) in \
                            enumerate(zip(early_stop_list, m_dict_list, loss_value_list, self.keys)):
                            if early_stop():
                                m_dict[f'loss/{phase}'].update(loss_value)
                                if 'label' in key:
                                    acc = torch.sum((torch.argmax(batch_dm_output_list[idx], dim=-1) ==
                                                     batch_labels).type(torch.float32)) / len(batch_labels)
                                    acc_value = float(acc)
                                    m_dict[f'acc/{phase}'].update(acc_value)

                        if phase == 'train':
                            loss_sum = torch.tensor(0.).to(dm_device)
                            for i in loss_list:
                                if i != 0.:
                                    loss_sum += i
                            if loss_sum != 0.:
                                loss_sum.backward()
                            for early_stop, optimizer in zip(early_stop_list, optimizer_list):
                                if early_stop():
                                    if loss_sum != 0.:
                                        optimizer.step()
                                    else:
                                        early_stop.set_terminate()

                    # if (phase == 'test') and (mini_step == 0) and ((ep-1) % 10 == 0):
                    #     visualize(batch_inputs[:10], batch_predicts_recon[:10], writer, ep)
                for m_dict, early_stop, key in zip(m_dict_list, early_stop_list, self.keys):
                    if early_stop():
                        m_dict[f'loss/{phase}'].write_tape(plot_index=ep)
                        writer.add_scalar(f'in-dm_{key}/loss/{phase}', m_dict[f'loss/{phase}'].average, ep)
                        if 'label' in key:
                            m_dict[f'acc/{phase}'].write_tape(plot_index=ep)
                            writer.add_scalar(f'in-dm_{key}/acc/{phase}', m_dict[f'acc/{phase}'].average, ep)

            early_stop_all = False
            for early_stop, m_dict in zip(early_stop_list, m_dict_list):
                early_stop.update(m_dict['loss/val'].average)
                early_stop_all |= early_stop()
            if not early_stop_all:
                break
            tq.update()
        # All deep-measure finished
        # Save final dm ckpt to file, next dm test will load from this ckpt and save time
        # ckpt_init_path = os.path.join(self.result_home_path, 'ckpt_dm')
        # os.makedirs(ckpt_init_path, exist_ok=True)


        # Save dict to result_home_path
        # m_dict_folder_path = os.path.join(self.rpm('ckpt_dm'), 'dm_tape')
        # os.makedirs(m_dict_folder_path, exist_ok=True)
        for m_dict, key in zip(m_dict_list, self.keys):
            m_dict_path = os.path.join(self.rpm('dm_tape'), f'step_{global_mini_step:012},{key}.json')
            # m_dict_path = os.path.join(m_dict_folder_path, f'step_{global_mini_step:012},{key}.json')
            temp_save = {}
            for name, i_dict in m_dict.items():
                temp_save[name] = i_dict.tape
            util_log.save_json(m_dict_path, temp_save)
            # np.save(m_dict_path, temp_save)


def process_job(q):
    while not q.empty():
        print(f'Queue size: {q.qsize()}')

        main_model_ckpt_file_path,\
        main_result_path,\
        timestamp = q.get()
        main_info_path = os.path.join(main_result_path, 'info_train_main.json')
        with open(main_info_path, 'r') as f:
            main_info = json.load(f)
        # Setup deep-measure networks
        hpara_dm_recon = {'dim_in': 0,
                          'dim_out': main_info['hpara_main']['dim_in'],
                          'depth': 1,  # 1
                          'width': 0,  # 0
                          'act': 'relu'}
        hpara_dm_label = {'dim_in': 0,
                          'dim_out': main_info['hpara_main']['dim_out'],
                          'depth': 1,  # 1
                          'width': 0,  # 0
                          'act': 'relu'}
        # -------------
        # hpara_dm_recon = {'dim_in': 0,
        #                   'dim_out': main_info['hpara_main']['dim_in'],
        #                   'depth': 2,  # 1
        #                   'width': 64,  # 0
        #                   'act': 'relu'}
        # hpara_dm_label = {'dim_in': 0,
        #                   'dim_out': main_info['hpara_main']['dim_out'],
        #                   'depth': 2,  # 1
        #                   'width': 32,  # 0
        #                   'act': 'relu'}
        # -------------
        # hpara_dm_recon = {'dim_in': 0,
        #                   'dim_out': main_info['hpara_main']['dim_in'],
        #                   'depth': 3,  # 1
        #                   'width': 64,  # 0
        #                   'act': 'relu'}
        # hpara_dm_label = {'dim_in': 0,
        #                   'dim_out': main_info['hpara_main']['dim_out'],
        #                   'depth': 3,  # 1
        #                   'width': 32,  # 0
        #                   'act': 'relu'}
        hpara_train_dm = {
            'timestamp': timestamp,
            'task': 'mnist_cl',
            'max_ep': 600,
            'batch_size': 200,
            'train_size': main_info['hpara_train']['train_size'],
            'lr': 1e-2,
            'early_stop_window_size': 20
        }
        dm_note = f'cold_start,depth={hpara_dm_recon["depth"]}-net'

        dm = DeepMeasure(dm_note,
                         hpara_dm_recon,
                         hpara_dm_label,
                         hpara_train_dm,
                         main_model_ckpt_file_path,
                         main_result_path)
        dm.train()


if __name__ == '__main__':
    time_start = datetime.now()

    main_result_path_list = [
        # Path to the folder of pre-trained record (with a timestamp).
        # eg: '/results/sim/sim_mnist/results,act=relu,opt=sgd,train_size=6000,width=64/mnist_cl,2022-05-11_18-11-37_295524'
    ]

    for main_result_path in main_result_path_list:
        main_ckpt_folder = os.path.join(main_result_path, 'ckpt_main')
        main_ckpt_list = sorted([os.path.join(main_ckpt_folder, i) for i in os.listdir(main_ckpt_folder)])

        # Do deep measure every n epochs
        every_n = 1
        main_ckpt_list_new = []
        for idx in range(len(main_ckpt_list)):
            if (idx % every_n) == 0:
            # if (idx % every_n) == 0 or idx < 35:
                main_ckpt_list_new.append(main_ckpt_list[idx])
            else:
                continue

        main_ckpt_list = main_ckpt_list_new

        timestamp = util_log.get_formatted_time()
        args = zip(
            main_ckpt_list[1:],
            [main_result_path] * (len(main_ckpt_list) - 1),
            [timestamp] * (len(main_ckpt_list) - 1),
        )

        q = mp.Queue()
        for i in args:
            q.put(i)

        processes = []

        for rank in range(15):
            p = mp.Process(target=process_job, args=(q,), daemon=False)
            time.sleep(30)
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        time_cost = datetime.now() - time_start

        print(time_cost)





