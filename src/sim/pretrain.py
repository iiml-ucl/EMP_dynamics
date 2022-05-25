"""
Train the main network that to be evaluated by the deep-measure algorithm.

For DNN, MNIST classification.

This code was built based on pretrain.py
Adjusted to fit non-rectangle network training.

"""
import time

import numpy as np
import torch
from tqdm import tqdm
import os
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import json
import torch.multiprocessing as mp
import sys
sys.path.append('../../..')
import src.models.dnn as dnn
from src.util.load_dataset import recipes
import src.util.util_torch as util_torch
from src.util.util_torch import save_ckpt
import src.util.util_log as util_log

PROJECT_ROOT_PATH = os.path.join(os.getcwd(), '..')


class SaveCkptConditionChecker:
    """
    The condition to save a checkpoint for deep-measure
    """
    def __init__(self, setup):
        """

        :param setup: (list) [before_n-th_epoch, mini-step_step_size]. -1-th epoch means any epoch.
            e.g.:
                [[1, 10], [5, 20], [10, 60], [-1, 100]] means:
                    before 1-th epoch, save every 10 mini-steps;
                    before 5-th epoch, save every 20 mini-steps;
                    before 10-th epoch, save every 60 mini-steps;
                    in any epoch, save every 100 mini-steps.
        """
        self.setup = setup

    def __call__(self, epoch, global_mini_steps):
        """
        Get the flag whether to save the checkpoint.

        :param epoch: (int) current epoch number (start from 1).
        :param global_mini_steps: (int) current global mini-steps (start from 0).
        :return: (bool) True: save ckpt;
        """
        for condition in self.setup:
            assert isinstance(condition, list)
            if condition[0] == -1:  # In any epoch.
                if global_mini_steps % condition[1] == 0:
                    return True
                else:
                    return False
            elif epoch-1 > condition[0]:  # Current epoch exceed this condition's epoch.
                continue
            else:
                if global_mini_steps % condition[1] == 0:
                    return True
                else:
                    return False
        return False


class TrainMain:
    def __init__(self, sim_note: str, hpara_main: dict, hpara_train: dict, optimizer_type='adam') -> None:
        """
        Train the main network, and save its checkpoints to be analyzed by deep-measure.


        :param sim_note: (String) The note for this train/setup. will be part of the result folder.
        :param hpara_main: (dict) The dictionary for hyper-parameters of the main network. Keys and values vary per
                            network model.
        :param hpara_train: (dict) The dictionary for the hyper-parameters of the training of the main network.
        :param optimizer_type: (Optional, String) Select from {'sgd', 'adam'}.
        """
        seed = datetime.now().microsecond
        torch.manual_seed(seed)


        self.device = util_torch.auto_select_GPU(dwell=1)
        self.hpara_main = hpara_main

        self.timestamp = hpara_train['timestamp']
        self.task = hpara_train['task']
        self.tot_ep = hpara_train['tot_ep']
        self.batch_size = hpara_train['batch_size']
        self.train_size = hpara_train['train_size']
        self.save_ckpt_setup = hpara_train['save_ckpt_setup']
        self.reg = hpara_train['reg']

        self.rpm = util_log.ResultPathManager(PROJECT_ROOT_PATH)
        self.rpm.add_path('sim', 'root', f'results,{sim_note}')
        self.rpm.add_path('ith_repeat', 'sim', f'{self.task},{self.timestamp}')
        self.rpm.add_path('ckpt', 'ith_repeat', 'ckpt_main')
        self.rpm.add_path('info.json', 'ith_repeat', 'info_train_main.json')
        self.rpm.add_path('main_tape.json', 'ith_repeat', 'main_tape.json')

        # result_home_path = os.path.join(os.getcwd(), f'results,{sim_note}')
        print(f'\n\n{datetime.now()} I Open TensorBoard with:\n'
              f'tensorboard --logdir={self.rpm("root")} --host=0.0.0.0 --port=6007\n\n')
        # self.result_path = os.path.join(result_home_path, f'{self.task},{self.timestamp}')
        # os.makedirs(self.result_path)
        self.writer = SummaryWriter(self.rpm('ith_repeat'))

        self.m_dict = {
            'loss/train': util_log.Measurement('loss/train'),
            'loss/val': util_log.Measurement('loss/val'),
            'loss/test': util_log.Measurement('loss/test'),
            'acc/train': util_log.Measurement('acc/train'),
            'acc/val': util_log.Measurement('acc/val'),
            'acc/test': util_log.Measurement('acc/test'),
        }

        self.model = dnn.DNN(network_hpara=hpara_main).to(self.device)

        self.dataloaders = recipes(self.task,
                                   path_raw_dataset=os.path.join(self.rpm('project_root'), f'dataset/{self.task.split()[0]}'),
                                   batch_size=self.batch_size,
                                   dataset_kwargs={'train_size': self.train_size})

        self.loss_fn, _ = util_torch.select_loss('cross_entropy')
        self.optimizer_type = optimizer_type
        self.optimizer_class = util_torch.select_optimizer(self.optimizer_type)

        if self.reg is None:
            self.optimizer = self.optimizer_class(self.model.parameters(), hpara_train['lr'])
        elif self.reg[0] == 'weight_decay':
            self.optimizer = self.optimizer_class(self.model.parameters(), hpara_train['lr'], weight_decay=self.reg[1])
        else:
            self.optimizer = self.optimizer_class(self.model.parameters(), hpara_train['lr'])

        # self.ckpt_folder_path = os.path.join(self.result_path, 'ckpt_main')
        # os.makedirs(self.ckpt_folder_path, exist_ok=True)
        # self.info_path = os.path.join(self.result_path, 'info_train_main.json')
        info = {
            'sim_note': sim_note,
            'hpara_main': hpara_main,
            'hpara_train': hpara_train,
            'repre_dict': self.model.repre_dict,
            'seed': seed,
        }
        with open(self.rpm('info.json'), 'w') as f:
            json.dump(info, f, indent=4)

        self.save_ckpt_condition_checker = SaveCkptConditionChecker(self.save_ckpt_setup)

    def run(self):
        global_mini_step = 0
        tq = tqdm(total=self.tot_ep, dynamic_ncols=True, unit='ep', position=1)
        for ep in range(1, self.tot_ep + 1):
            # Validation phase
            for phase in ['val', 'train', 'test']:
                tq.set_description(f'train,{phase}')
                if phase == 'test' and ep != self.tot_ep:
                    continue
                self.m_dict[f'loss/{phase}'].clear()
                self.m_dict[f'acc/{phase}'].clear()
                for mini_step, batch_sample in enumerate(self.dataloaders[phase]):

                    # Save ckpt for Deep-measure
                    if phase == 'train':
                        if self.save_ckpt_condition_checker(ep, global_mini_step):
                            save_ckpt(self.rpm('ckpt').joinpath(f'step_{global_mini_step:012}.pt'),
                                      ep, self.model, self.optimizer, 0)
                        global_mini_step += 1

                    batch_inputs = batch_sample[0][0].to(self.device)
                    batch_labels = batch_sample[1][0].to(self.device)
                    self.optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        batch_predicts = self.model(batch_inputs)
                        loss = self.loss_fn(batch_predicts, batch_labels)
                        acc = torch.sum((torch.argmax(batch_predicts, dim=-1) ==
                                         batch_labels).type(torch.float32)) / self.batch_size
                        loss_value = float(loss)
                        acc_value = float(acc)
                        if phase == 'train':
                            loss.backward()
                            self.optimizer.step()
                    self.m_dict[f'loss/{phase}'].update(loss_value)
                    self.m_dict[f'acc/{phase}'].update(acc_value)
                tq.set_postfix(loss=loss_value, acc=acc_value)
                self.m_dict[f'loss/{phase}'].write_tape(global_mini_step)
                self.m_dict[f'acc/{phase}'].write_tape(global_mini_step)
                self.writer.add_scalar(f'loss/{phase}', self.m_dict[f'loss/{phase}'].average, global_mini_step)
                self.writer.add_scalar(f'acc/{phase}', self.m_dict[f'acc/{phase}'].average, global_mini_step)
                self.writer.add_scalar(f'loss/{phase}_ep', self.m_dict[f'loss/{phase}'].average, ep)
                self.writer.add_scalar(f'acc/{phase}_ep', self.m_dict[f'acc/{phase}'].average, ep)
            tq.update()

        # Summarize the m_dict's tapes and save it
        temp_save = {}
        for name, i_dict in self.m_dict.items():
            temp_save[name] = i_dict.tape
        # path_tape = os.path.join(self.rpm(''), 'main_tape.npy')
        with open(self.rpm('main_tape.json'), 'w') as f:
            json.dump(temp_save, f, indent=4)
        # np.save(path_tape, temp_save)
        print(f'{datetime.now()} I Training finished. Timestamp: {self.timestamp}')


def visualize(inputs, predicts_recon, writer, epoch):
    inputs = inputs.detach().cpu().numpy()
    predicts = predicts_recon.detach().cpu().numpy()
    predicts = predicts.reshape(inputs.shape)
    inputs = np.concatenate([inputs, inputs, inputs], axis=1)
    predicts = np.concatenate([predicts, predicts, predicts], axis=1)
    inputs = np.concatenate(inputs, -1)
    predicts = np.concatenate(predicts, -1)
    img = np.concatenate([inputs, predicts], 1).clip(0, 1)
    writer.add_image('inputs vs recons', img, epoch)


def process_job(q):
    while not q.empty():
        idx_repeat = q.get()  # Consume a job in queue by a thread

        hpara_main = {'dim_in': 28 * 28,
                      'dim_out': 10,
                      'depth': 4,
                      'width': 16,  # [16, 64, 512]
                      'act': 'relu'}
        hpara_train = {
            'timestamp': util_log.get_formatted_time(),
            'task': 'mnist_cl',
            'tot_ep': 4800,  # sgd: 2000; adam: 300;
            'batch_size': 200,  # 60,000 / 200 = 300 mini-steps per epoch
            'train_size': 6000,
            'save_ckpt_setup': [[-1, 4800 * 6000 / 200 / 300]],  # sgd: [[-1, 300]]; adam: [[-1, 30]]  tot_ep * train_size / batch_size / i = num_points
            'opt': 'sgd',  # sgd; adam
            'lr': 1e-2,  # sgd: 1e-2, adam: 1e-4
            'reg': None,  # ('weight_decay', 1e-3)
        }

        sim_note = f'act={hpara_main["act"]},opt={hpara_train["opt"]},train_size={hpara_train["train_size"]},width={hpara_main["width"]}'
        train = TrainMain(sim_note, hpara_main, hpara_train, optimizer_type=hpara_train['opt'])
        train.run()

        del train


if __name__ == '__main__':
    repeat = 5

    q = mp.Queue()
    args = zip(
        list(range(repeat))
    )
    for i in args:
        q.put(i)

    processes = []

    for rank in range(5):
        p = mp.Process(target=process_job, args=(q,), daemon=True)
        time.sleep(30)
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

